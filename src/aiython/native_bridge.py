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
import threading
import weakref

from .type_constraints import Contract, TypeViolation, compile_contract
from .typed_runtime import TypedTransformer, TypeRuntime, unconstrained_variadic


_VM_LOCK = threading.RLock()
_VM_AVAILABLE = None
_VM_CODES = {}
_VM_HOOK_INSTALLED = False
_VM_PREVIOUS_STORE = None
_VM_PREVIOUS_CALL = None


def vm_available():
    """Probe the store callback once; stock CPython ignores the sys attribute."""
    global _VM_AVAILABLE
    if _VM_AVAILABLE is not None:
        return _VM_AVAILABLE
    with _VM_LOCK:
        if _VM_AVAILABLE is None:
            missing = object()
            previous = getattr(sys, '_aiython_before_store', missing)
            observed = False

            def probe(frame, name, value):
                nonlocal observed
                if frame.f_code.co_filename == '<aiython-vm-probe>':
                    observed = True

            try:
                sys._aiython_before_store = probe
                exec(compile('probe_value = 1', '<aiython-vm-probe>', 'exec'), {})
            finally:
                if previous is missing:
                    vars(sys).pop('_aiython_before_store', None)
                else:
                    sys._aiython_before_store = previous
            _VM_AVAILABLE = observed
    return _VM_AVAILABLE


def _vm_entry(code):
    with _VM_LOCK:
        entry = _VM_CODES.get(id(code))
        return entry if entry is not None and entry[0]() is code else None


def _vm_before_store(frame, name, value):
    if _VM_LOCK is None:
        return
    entry = _vm_entry(frame.f_code)
    if entry is not None:
        entry[1].before_store(frame, name, value)
    if _VM_PREVIOUS_STORE is not None:
        _VM_PREVIOUS_STORE(frame, name, value)


def _vm_on_call(frame):
    if _VM_LOCK is None:
        return
    entry = _vm_entry(frame.f_code)
    if entry is not None:
        entry[1].on_call(frame)
    if _VM_PREVIOUS_CALL is not None:
        _VM_PREVIOUS_CALL(frame)


def _register_vm_code(code, bridge, info):
    global _VM_HOOK_INSTALLED, _VM_PREVIOUS_STORE, _VM_PREVIOUS_CALL
    with _VM_LOCK:
        if not _VM_HOOK_INSTALLED:
            _VM_PREVIOUS_STORE = getattr(sys, '_aiython_before_store', None)
            _VM_PREVIOUS_CALL = getattr(sys, '_aiython_on_call', None)
            sys._aiython_before_store = _vm_before_store
            sys._aiython_on_call = _vm_on_call
            _VM_HOOK_INSTALLED = True
        key = id(code)

        def discard(reference):
            with _VM_LOCK:
                current = _VM_CODES.get(key)
                if current is not None and current[0] is reference:
                    _VM_CODES.pop(key, None)

        _VM_CODES[key] = (weakref.ref(code, discard), bridge, info)


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

    ``prepare_unit`` is the normal runtime entry point. Plain source keeps
    CPython code objects. Patched CPython uses VM callbacks for simple typed
    module assignments, with statement checkpoints when recovery is enabled.
    Other typed source and recovery use Aiython's boundary compiler. A bridge
    bound to a Runtime also exposes this behavior through ``compile_source``.
    Without a Runtime, ``compile_source`` and ``installed`` expose VM hooks.
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

    def vm_source(self, unit):
        """Use VM stores for simple annotated module assignments."""
        manager = self.manager
        if (manager is None or unit.blocks or unit.directives.annotations or self.types.classes
                or self.types._module_scopes):
            return False
        found_annotation = False
        for statement in unit.tree.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                found_annotation = True
                if not isinstance(statement.annotation, ast.Name):
                    return False
                if statement.annotation.id not in ('int', 'str', 'bool', 'float', 'bytes'):
                    return False
                value = statement.value
            elif isinstance(statement, ast.Assign) and all(
                    isinstance(target, ast.Name) for target in statement.targets):
                value = statement.value
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            elif isinstance(statement, ast.Pass):
                continue
            else:
                return False
            if value is not None and not isinstance(value, ast.Name):
                try:
                    ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    return False
        return found_annotation and vm_available()

    def prepare_unit(self, unit, *, entry=False, flags=0,
                     display_last_expr=False, recovery_metadata=False):
        """Return (code, native); recovery and typed yields remain executable."""
        manager = self.manager
        if manager is None:
            raise RuntimeError('Source preparation requires a Runtime')
        if not (flags or display_last_expr or recovery_metadata) and self.vm_source(unit):
            code = (self._compile_vm_recovery(unit) if manager.config.profiles else
                    self._compile_vm_source(unit.source, unit.filename))
            _register_vm_code(code, self, self.module_types[unit.filename])
            if not manager.config.profiles:
                manager.register(unit)
            linecache.cache[unit.filename] = (
                len(unit.source), None, unit.source.splitlines(True), unit.filename)
            return code, True
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

    def uses_vm(self, code):
        entry = _vm_entry(code)
        return entry is not None and entry[1] is self

    def compile_source(self, source: str, filename: str, *, entry=False):
        if self.manager is not None:
            from .frontend import parse
            unit = parse(source, filename)
            code, native = self.prepare_unit(unit, entry=entry)
            return code if native else self.manager._bind_compiled(code, unit.runtime_name)
        return self._compile_vm_source(source, filename)

    def _index_vm_source(self, source, filename):
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
        return tree

    def _compile_vm_source(self, source, filename):
        self._index_vm_source(source, filename)
        # Compile the exact source. Compiling an AST can produce a different
        # code object, even when the tree is not explicitly transformed.
        return compile(source, filename, 'exec', dont_inherit=True)

    def _compile_vm_recovery(self, unit):
        """Keep VM type callbacks while adding statement retry boundaries."""
        from .runtime import RuntimeReferences, bind_runtime, install_checkpoint

        manager = self.manager
        tree = self._index_vm_source(unit.source, unit.filename)
        manager.register(unit)
        body = []
        for index, statement in enumerate(tree.body):
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                key = f'{unit.filename}:vm-checkpoint:{index}'
                statement = install_checkpoint(manager, unit, statement, key,
                                               scoped_retries=True)
            body.append(statement)
        tree.body = body
        ast.fix_missing_locations(tree)
        tree = RuntimeReferences(unit.runtime_name).visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, unit.filename, 'exec', dont_inherit=True)
        return bind_runtime(code, unit.runtime_name, manager)

    def _frame_types(self, frame):
        code = frame.f_code
        entry = _vm_entry(code)
        if entry is not None and entry[1] is self:
            return entry[2]
        return self.frames.get((code.co_filename, code.co_qualname,
                                code.co_firstlineno))

    def before_store(self, frame, name, value):
        info = self._frame_types(frame)
        if info is None:
            return
        if frame.f_code.co_name == '<module>':
            retained = self.types._module_scopes.get(id(frame.f_globals))
            if retained is None or retained[0] is not frame.f_globals:
                scope = self.types._initialize(frame, info.declarations, {}, None)
                self.types._module_scopes[id(frame.f_globals)] = (frame.f_globals, scope)
        source = info.declarations.get(name)
        variadic = None
        if source is None and name in info.parameters:
            parameter_source, mode = info.parameters[name]
            if mode == 'value':
                source = parameter_source
            else:
                variadic = (parameter_source, mode)
        if source is None and name in frame.f_code.co_freevars:
            outer = info.outer
            while outer is not None:
                source = outer.declarations.get(name)
                if source is None and name in outer.parameters:
                    parameter_source, mode = outer.parameters[name]
                    if mode == 'value':
                        source = parameter_source
                    else:
                        variadic = (parameter_source, mode)
                if source is not None or variadic is not None:
                    break
                outer = outer.outer
        if name in info.global_names:
            source = self.module_types[frame.f_code.co_filename].declarations.get(name)
        if source or variadic is not None:
            namespace = self.types.namespace(frame)
            if variadic is not None:
                parameter_source, mode = variadic
                contract = (Contract('any', 'Any') if unconstrained_variadic(
                    parameter_source, mode, namespace) else
                    self.types.contract(parameter_source, namespace))
                if mode == 'args':
                    contract = Contract('tuple_many', parameter_source, (contract,))
                else:
                    contract = (contract.args[0] if contract.kind == 'unpack_typeddict' else
                                Contract('dict', parameter_source,
                                         (compile_contract('str', namespace), contract)))
            else:
                contract = self.types.contract(source, namespace)
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
        if info is None:
            return
        if frame.f_code.co_name == '<module>':
            scope = self.types._initialize(frame, info.declarations, {}, None)
            self.types._module_scopes[id(frame.f_globals)] = (frame.f_globals, scope)
            return
        if not info.parameters:
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

    def on_yield(self, code, offset, value):
        """Validate the value before CPython exposes it to the generator caller."""
        frame = inspect.currentframe().f_back
        try:
            info = self._frame_types(frame)
            if info is None or not info.returns:
                return
            contract = self.types.contract(info.returns, self.types.namespace(frame))
            if contract.kind not in ('generator', 'async_generator'):
                raise TypeViolation('Generator return annotation must describe yielded values')
            contract.args[0].validate(value, 'yield')
        finally:
            del frame

    @contextmanager
    def _yield_monitor(self):
        monitoring = getattr(sys, 'monitoring', None)
        if monitoring is None or not hasattr(monitoring.events, 'PY_YIELD'):
            yield
            return
        tool = next((number for number in reversed(range(6))
                     if monitoring.get_tool(number) is None), None)
        if tool is None:
            raise RuntimeError('No free sys.monitoring tool ID for typed yields')
        monitoring.use_tool_id(tool, 'aiython-native-yield')
        try:
            monitoring.register_callback(tool, monitoring.events.PY_YIELD, self.on_yield)
            monitoring.set_events(tool, monitoring.events.PY_YIELD)
            try:
                yield
            finally:
                monitoring.set_events(tool, 0)
                monitoring.register_callback(tool, monitoring.events.PY_YIELD, None)
        finally:
            monitoring.free_tool_id(tool)

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
            with self._yield_monitor():
                yield self
        finally:
            for name, value in previous.items():
                if value is missing:
                    vars(sys).pop(name, None)
                else:
                    setattr(sys, name, value)
