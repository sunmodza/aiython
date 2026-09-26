"""Guard ordinary Python filesystem APIs during AI execution; not a sandbox."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import sys

from .models import AiythonError


class SourceWriteError(AiythonError, PermissionError):
    pass


_ACTIVE = ContextVar('aiython_source_guard', default=None)
SOURCE_SUFFIXES = {'.py', '.pyi', '.pyw', '.js', '.jsx', '.ts', '.tsx', '.c', '.h', '.cpp', '.rs', '.go', '.java'}


def protected(path, manager):
    if isinstance(path, int):
        try:
            path = os.readlink(f'/proc/self/fd/{path}')
        except OSError:
            return True
    if path is None:
        return False
    path = Path(os.fsdecode(path))
    resolved = path.resolve()
    # Check both names: a source-named symlink and a symlink to source are protected.
    return (path.suffix.lower() in SOURCE_SUFFIXES or resolved.suffix.lower() in SOURCE_SUFFIXES
            or str(resolved) in manager.units)


def audit(event, args):
    manager = _ACTIVE.get()
    if manager is None:
        return
    paths = []
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            paths = [path]
    elif event in {'os.remove', 'os.rmdir', 'os.chmod', 'os.truncate', 'os.utime'}:
        paths = [args[0]]
    elif event in {'os.rename', 'os.link', 'os.symlink'}:
        paths = list(args[:2])
    elif event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.spawn', 'ctypes.dlopen'}:
        raise SourceWriteError('AI execution cannot launch subprocesses or load native libraries to bypass source protection. Use live Python runtime tools.')
    if any(protected(path, manager) for path in paths):
        raise SourceWriteError('AI must not create, rewrite, delete or replace source files. Execute the requested semantics in the live frame; do not fix source on disk.')
    # Moving/removing directories could indirectly move source files.
    if event in {'os.rename', 'os.rmdir'} and any(Path(os.fsdecode(p)).is_dir() for p in paths if not isinstance(p, int)):
        raise SourceWriteError('AI cannot move or remove directories containing potential source files.')


# Python otherwise hides audit hooks from tracers, including coverage.
audit.__cantrace__ = True
sys.addaudithook(audit)


@contextmanager
def protect_source(manager):
    token = _ACTIVE.set(manager)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
