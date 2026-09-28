"""Execution routing plus optional VM type hooks for original CPython code.

The normal runtime uses this bridge to select native code or the existing
boundary compiler for contracts, typed yields and recovery. The VM hook API
needs the optional CPython patches in ``native/patches``.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import linecache
import sys

from .type_constraints import TypeViolation
from .typed_runtime import TypedTransformer, TypeRuntime


@dataclass(frozen=True)
class _FrameTypes:
    declarations: dict[str, str]
    parameters: dict[str, tuple[str, str]]
    returns: str | None
    global_names: frozenset[str]
    outer: _FrameTypes | None = None


def _parameter_types(arguments):
    result = {}
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        if argument.annotation is not None:
            result[argument.arg] = (ast.unparse(argument.annotation), 'value')
    for argument, mode in ((arguments.vararg, 'args'), (arguments.kwarg, 'kwargs')):
        if argument is not None and argument.annotation is not None:
            result[argument.arg] = (ast.unparse(argument.annotation), mode)
    return result


def _global_names(body):
    names = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Global):
            names.update(node.names)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in body:
        visit(statement)
    return frozenset(names)


class NativeTypeBridge:
    """Choose CPython bytecode or Aiython boundaries and validate VM hooks.

    ``prepare_unit`` is the normal runtime entry point. It compiles plain
    source directly with CPython, while annotated source, typed yields and AI
    recovery use the existing Aiython boundary compiler. A bridge bound to a
    Runtime also exposes this behavior through ``compile_source``. Without a
    Runtime, ``compile_source`` and ``installed`` expose patched-VM hooks.
    """

    _HOOKS = ('_aiython_before_store', '_aiython_before_mutation',
              '_aiython_on_call', '_aiython_on_return')

    def __init__(self, types: TypeRuntime | None = None, *, manager=None):
        self.types = types or TypeRuntime()
        self.manager = manager
        self.frames: dict[tuple[str, str, int], _FrameTypes] = {}
        self.module_types: dict[str, _FrameTypes] = {}

    def native_source(self, unit, *, flags=0, display_last_expr=False,
                      recovery_metadata=False):
        """Keep runtime checks when source can introduce typed objects."""
        manager = self.manager
        if manager is None:
            raise RuntimeError('Native source selection requires a Runtime')
        if (flags or display_last_expr or recovery_metadata or manager.config.profiles
                or unit.blocks or unit.directives.annotations
                or self.types.classes or self.types._module_scopes):
            return False
        for node in ast.walk(unit.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            if isinstance(node, ast.Name) and node.id in ('__import__', '__builtins__'):
                return False
            if isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name) and node.func.id in
                    ('__import__', 'exec', 'eval', 'compile', 'getattr', 'vars', 'globals', 'locals')
                    or isinstance(node.func, ast.Attribute) and node.func.attr in
                    ('import_module', 'run_module', 'run_path')):
                return False
            if isinstance(node, ast.AnnAssign):
                return False
            if isinstance(node, ast.arg) and node.annotation is not None:
                return False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                return False
            if getattr(node, 'type_params', ()):
                return False
            if getattr(ast, 'TypeAlias', None) is not None and isinstance(node, ast.TypeAlias):
                return False
        return True

    def prepare_unit(self, unit, *, entry=False, flags=0,
                     display_last_expr=False, recovery_metadata=False):
        """Return (code, native); recovery and typed yields remain executable."""
        manager = self.manager
        if manager is None:
            raise RuntimeError('Source preparation requires a Runtime')
        if not self.native_source(unit, flags=flags,
                                  display_last_expr=display_last_expr,
                                  recovery_metadata=recovery_metadata):
            return manager.prepare(unit, entry=entry, flags=flags,
                                   display_last_expr=display_last_expr,
                                   recovery_metadata=recovery_metadata), False
        manager.register(unit)
        linecache.cache[unit.filename] = (
            len(unit.source), None, unit.source.splitlines(True), unit.filename)
        return compile(unit.source, unit.filename, 'exec', dont_inherit=True), True

    def compile_source(self, source: str, filename: str, *, entry=False):
        if self.manager is not None:
            from .frontend import parse
            unit = parse(source, filename)
            code, native = self.prepare_unit(unit, entry=entry)
            return code if native else self.manager._bind_compiled(code, unit.runtime_name)
        tree = ast.parse(source, filename)
        module = _FrameTypes(TypedTransformer.declarations_in(tree.body), {}, None,
                             frozenset())
        self.module_types[filename] = module
        self.frames[(filename, '<module>', 1)] = module

        def collect(node, prefix, outer):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                info = _FrameTypes(
                    TypedTransformer.declarations_in(node.body),
                    _parameter_types(node.args),
                    ast.unparse(node.returns) if node.returns is not None else None,
                    _global_names(node.body), outer)
                for line in {node.lineno, *(item.lineno for item in node.decorator_list)}:
                    self.frames[(filename, name, line)] = info
                for statement in node.body:
                    collect(statement, name + '.<locals>.', info)
            elif isinstance(node, ast.ClassDef):
                name = prefix + node.name
                info = _FrameTypes(TypedTransformer.declarations_in(node.body),
                                   {}, None, _global_names(node.body), outer)
                for line in {node.lineno, *(item.lineno for item in node.decorator_list)}:
                    self.frames[(filename, name, line)] = info
                for statement in node.body:
                    # Methods cannot capture names from a class namespace.
                    collect(statement, name + '.', outer)
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child, prefix, outer)

        for statement in tree.body:
            collect(statement, '', module)
        # Compile the exact source. Compiling an AST can produce a different
        # code object, even when the tree is not explicitly transformed.
        return compile(source, filename, 'exec', dont_inherit=True)

    def _frame_types(self, frame):
        code = frame.f_code
        return self.frames.get((code.co_filename, code.co_qualname,
                                code.co_firstlineno))

    def before_store(self, frame, name, value):
        info = self._frame_types(frame)
        if info is None:
            return
        source = info.declarations.get(name)
        if source is None and name in info.parameters:
            parameter_source, mode = info.parameters[name]
            if mode == 'value':
                source = parameter_source
        if source is None and name in frame.f_code.co_freevars:
            outer = info.outer
            while outer is not None:
                source = outer.declarations.get(name)
                if source is None and name in outer.parameters:
                    source, mode = outer.parameters[name]
                    if mode != 'value':
                        source = None
                if source is not None:
                    break
                outer = outer.outer
        if name in info.global_names:
            source = self.module_types[frame.f_code.co_filename].declarations.get(name)
        if source:
            contract = self.types.contract(source, self.types.namespace(frame))
            if contract.marker == 'Final':
                values = frame.f_globals if name in info.global_names else frame.f_locals
                if name in values:
                    raise TypeViolation(f'{name}: Final binding cannot be reassigned')
            contract.validate(value, name)

    def before_mutation(self, frame, kind, owner, key, value):
        if kind == 'attr' and self._frame_types(frame) is not None:
            self.types.validate_attribute(frame, owner, key, value)

    def on_call(self, frame):
        info = self._frame_types(frame)
        if info is None or not info.parameters:
            return
        self.types._initialize(frame, info.declarations, info.parameters, info.returns)

    def on_return(self, frame, value):
        info = self._frame_types(frame)
        if info is not None and info.returns:
            contract = self.types.contract(info.returns, self.types.namespace(frame))
            if (frame.f_code.co_flags & (inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR)
                    and contract.kind in ('generator', 'async_generator')):
                contract = contract.args[2]
            contract.validate(value, 'return')

    @contextmanager
    def installed(self):
        """Install callbacks for a scoped experiment, restoring prior hooks."""
        missing = object()
        previous = {name: getattr(sys, name, missing) for name in self._HOOKS}
        callbacks = (self.before_store, self.before_mutation,
                     self.on_call, self.on_return)
        try:
            for name, callback in zip(self._HOOKS, callbacks):
                setattr(sys, name, callback)
            observed = False

            def probe(frame, name, value):
                nonlocal observed
                if frame.f_code.co_filename == '<aiython-vm-probe>':
                    observed = True

            sys._aiython_before_store = probe
            try:
                exec(compile('probe_value = 1', '<aiython-vm-probe>', 'exec'), {})
            finally:
                sys._aiython_before_store = self.before_store
            if not observed:
                raise RuntimeError('NativeTypeBridge requires a CPython build with Aiython VM hooks')
            yield self
        finally:
            for name, value in previous.items():
                if value is missing:
                    vars(sys).pop(name, None)
                else:
                    setattr(sys, name, value)
