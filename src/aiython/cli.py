from __future__ import annotations

import argparse
import atexit
import builtins
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
import runpy
import sys
import tokenize
import types
from pathlib import Path
from time import perf_counter

from .config import describe, resolve
from .frontend import RUNTIME_NAME, parse
from .models import AiythonError, ConfigError
from .runtime import Runtime


def read_source(path):
    try:
        with tokenize.open(path) as file:
            return file.read()
    except OSError as exc:
        raise AiythonError(f"Cannot read source file: {path} ({type(exc).__name__})") from None


class ProjectLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, runtime):
        super().__init__(name, path)
        self.runtime = runtime

    def exec_module(self, module):
        module.__dict__[RUNTIME_NAME] = self.runtime
        exec(self.runtime.compile_source(read_source(self.path), self.path), module.__dict__)


class ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.config.project_root.resolve()
        self.aiython_root = Path(__file__).parent.resolve()

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if not spec or not isinstance(spec.loader, importlib.machinery.SourceFileLoader) or not spec.origin:
            return None
        location = Path(spec.origin).resolve()
        if not location.is_relative_to(self.root) or location.is_relative_to(self.aiython_root):
            return None
        relative = location.relative_to(self.root)
        if any(part in {".venv", "venv", "site-packages", "dist-packages", "__pycache__"} for part in relative.parts):
            return None
        spec.loader = ProjectLoader(fullname, spec.origin, self.runtime)
        return spec


def interpreter_arguments():
    original = sys.orig_argv
    for index in range(len(original) - 1):
        if original[index] == "-m" and original[index + 1] in ("aiython", "aiython.__main__"):
            return original[:index]
    return [sys.executable]


def module_source(name, arguments=()):
    old_main = sys.modules.get("__main__")
    old_argv, old_orig_argv = sys.argv, sys.orig_argv
    interpreter_args = interpreter_arguments()
    initial_main = types.ModuleType("__main__")
    initial_main.__loader__ = importlib.machinery.BuiltinImporter
    initial_main.__builtins__ = builtins
    if sys.version_info < (3, 14):
        initial_main.__annotations__ = {}
    try:
        sys.modules["__main__"] = initial_main
        sys.argv = ["-m", *arguments]
        sys.orig_argv = [*interpreter_args, "-m", name, *arguments]
        if not sys.flags.safe_path:
            sys.path[:1] = [str(Path.cwd())]
        actual_name, spec, code = runpy._get_module_details(name)
    except ImportError as exc:
        raise AiythonError(str(exc)) from None
    finally:
        sys.argv = old_argv
        sys.orig_argv = old_orig_argv
        if old_main is not None:
            sys.modules["__main__"] = old_main
        else:
            sys.modules.pop("__main__", None)
    loader = spec.loader
    source = getattr(loader, "get_source", lambda _: None)(actual_name)
    return spec, source, code, initial_main


def run_script(path: Path, arguments=(), *, config=None, agent_factory=None, stats=False, trace_plan=False,
               config_path=None, profile=None, force_profile=None, restore_state=True,
               source=None, module_spec=None, module_invocation=None, compiled_code=None, initial_main=None):
    started = perf_counter()
    argv0 = str(path) if module_spec is None else module_spec.origin
    display_path = path.absolute() if module_spec is None else path
    path = path.resolve()
    config_source = Path.cwd() / "__main__.py" if module_spec else path
    config = config or resolve(config_source, config_path=config_path, profile=profile, force_profile=force_profile)
    config_seconds = perf_counter() - started
    runtime = Runtime(config, agent_factory=agent_factory, stats=stats, trace_plan=trace_plan)
    code = (compiled_code if compiled_code is not None else
            runtime.compile_source(read_source(path) if source is None else source, str(display_path), entry=True))
    module = initial_main or types.ModuleType("__main__")
    module.__dict__.update({"__file__": str(display_path) if module_spec is None else module_spec.origin,
                            "__package__": module_spec.parent if module_spec else None,
                            "__spec__": module_spec,
                            "__cached__": module_spec.cached if module_spec else None,
                            "__loader__": module_spec.loader if module_spec else
                                          importlib.machinery.SourceFileLoader("__main__", str(display_path)),
                            RUNTIME_NAME: runtime,
                            "__builtins__": builtins})
    if sys.version_info < (3, 14):
        module.__dict__.setdefault("__annotations__", {})
    old_main = sys.modules.get("__main__")
    old_argv, old_orig_argv, old_path = sys.argv, sys.orig_argv, sys.path[:]
    interpreter_args = interpreter_arguments()
    old_spawn_entry = os.environ.get("AIYTHON_SPAWN_ENTRY")
    finder = ProjectFinder(runtime)
    execution_started = None
    def finish():
        execution_seconds = perf_counter() - execution_started if execution_started is not None else 0
        runtime.capabilities.close()
        if stats:
            runtime.stats.run = {'total_seconds': perf_counter() - started,
                                 'config_seconds': config_seconds,
                                 'execution_seconds': execution_seconds}
        runtime.stats.report()
    if not restore_state:
        # Register before user code so its atexit callbacks run while the
        # script's argv, import path, and runtime are still available.
        atexit.register(finish)
    try:
        sys.modules["__main__"] = module
        sys.argv = [argv0, *arguments]
        sys.orig_argv = ([*interpreter_args, "-m", module_invocation, *arguments] if module_spec else
                         [*interpreter_args, argv0, *arguments])
        if module_spec is None and not sys.flags.safe_path:
            sys.path.insert(0, str(path.parent))
        sys.meta_path.insert(0, finder)
        os.environ["AIYTHON_SPAWN_ENTRY"] = str(display_path)
        execution_started = perf_counter()
        exec(code, module.__dict__)
        return module.__dict__
    finally:
        if restore_state:
            sys.argv = old_argv
            sys.orig_argv = old_orig_argv
            sys.path[:] = old_path
            sys.meta_path.remove(finder)
            if old_spawn_entry is None:
                os.environ.pop("AIYTHON_SPAWN_ENTRY", None)
            else:
                os.environ["AIYTHON_SPAWN_ENTRY"] = old_spawn_entry
            if old_main is not None:
                sys.modules["__main__"] = old_main
            else:
                sys.modules.pop("__main__", None)
            finish()
        else:
            # CPython removes these entry-script attributes before waiting for
            # non-daemon threads and running atexit callbacks.
            if module_spec is None:
                module.__dict__.pop("__file__", None)
                module.__dict__.pop("__cached__", None)


def parser():
    result = argparse.ArgumentParser(
        prog="aiython", description="Python with project-scoped AI execution",
        epilog="Get started: aiython setup; aiython --explain script.py; aiython script.py; aiython -m package.module. "
               "Other commands: aiython config show; aiython jobs list.")
    result.add_argument("--version", action="version",
                        version=f"%(prog)s {importlib.metadata.version('aiython')}")
    result.add_argument("--config")
    group = result.add_mutually_exclusive_group()
    group.add_argument("--profile", help="Default profile; source directives may override it")
    group.add_argument("--force-profile", help="Use this profile for every AI invocation")
    result.add_argument("--explain", action="store_true", help="Show blocks/checkpoints without executing code")
    result.add_argument("--trace-plan", action="store_true", help="Trace actual capability routes, cache and timings")
    result.add_argument("--stats", action="store_true", help="Report model calls, tools, request bytes and timings on stderr")
    result.add_argument("-m", "--module", dest="module_args", nargs=argparse.REMAINDER,
                        help="Run a Python module as __main__")
    result.add_argument("script", nargs="?")
    result.add_argument("args", nargs=argparse.REMAINDER)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    original_path = sys.path[:] if argv is not None else None
    try:
        if not arguments:
            parser().print_help()
            return
        if arguments[:1] == ["setup"]:
            from .setup import setup
            setup(arguments[1:])
            return
        if arguments[:1] == ["jobs"]:
            command = argparse.ArgumentParser(prog="aiython jobs")
            command.add_argument("action", choices=("list", "resume"))
            command.add_argument("operation", nargs="?")
            command.add_argument("--script", default="main.py")
            command.add_argument("--config")
            command.add_argument("--profile")
            args = command.parse_args(arguments[1:])
            config = resolve(Path(args.script), config_path=args.config, profile=args.profile)
            runtime = Runtime(config)
            name = config.cli_profile or config.default_profile
            if name not in config.profiles:
                raise ConfigError("Select a configured profile for jobs")
            profile = config.profiles[name]
            runtime.capabilities.require(profile, "read_asset")
            if args.action == "list":
                print(json.dumps(runtime.capabilities.store.jobs(), indent=2))
            elif not args.operation:
                command.error("resume requires an operation ID")
            else:
                print(runtime.capabilities.resume_job(profile, args.operation).path)
            return
        if arguments[:2] == ["config", "show"]:
            command = argparse.ArgumentParser(prog="aiython config show")
            command.add_argument("--script", default="main.py")
            command.add_argument("--config")
            args = command.parse_args(arguments[2:])
            print(json.dumps(describe(resolve(Path(args.script), config_path=args.config)), indent=2, ensure_ascii=False))
            return
        argument_parser = parser()
        args = argument_parser.parse_args(arguments)
        module_spec = source = module_invocation = compiled_code = initial_main = None
        if args.module_args is not None:
            if not args.module_args:
                argument_parser.error("-m requires a module name")
            module_invocation, *script_args = args.module_args
            module_spec, source, original_code, initial_main = module_source(module_invocation, script_args)
            path = Path(module_spec.origin or original_code.co_filename)
            if source is None:
                compiled_code = original_code
        else:
            if args.script is None:
                argument_parser.error("a script path or -m module is required")
            path = Path(args.script)
            script_args = args.args
        if args.explain:
            if source is None and compiled_code is not None:
                raise AiythonError("Cannot explain a module without Python source")
            config_source = Path.cwd() / "__main__.py" if module_spec else path
            config = resolve(config_source, config_path=args.config, profile=args.profile,
                             force_profile=args.force_profile)
            filename = str(path if module_spec else path.resolve())
            unit = parse(read_source(path.resolve()) if source is None else source, filename)
            runtime = Runtime(config)
            runtime.prepare(unit, entry=True)
            print(json.dumps({"config": describe(config), "blocks": [
                {"statement": b.statement, "span": vars(b.span), "expression": b.expression, "output_type": b.output_type,
                 "plan": "requires runtime intent resolution", "cache": "unknown", "cost": "unknown",
                 "directive": vars(unit.directives.at(b.span.line))} for b in unit.blocks.values()],
                "checkpoints": [{"span": vars(c.span), "statement": c.statement}
                                for c in runtime.checkpoints.values()]}, ensure_ascii=False, indent=2))
            return
        run_script(path, script_args, config_path=args.config, profile=args.profile,
                   force_profile=args.force_profile, stats=args.stats, trace_plan=args.trace_plan,
                   restore_state=argv is not None, source=source,
                   module_spec=module_spec, module_invocation=module_invocation,
                   compiled_code=compiled_code, initial_main=initial_main)
    except AiythonError as exc:
        # Keep the original runtime cause visible without leaking provider internals.
        if exc.__cause__:
            import traceback
            traceback.print_exception(exc.__cause__)
        print(f"aiython: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if original_path is not None:
            sys.path[:] = original_path
