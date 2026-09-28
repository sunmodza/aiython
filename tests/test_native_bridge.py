"""Bridge contracts that also run without a patched CPython interpreter."""

import ast
import sys
import gc
import weakref
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from aiython.models import ProfileConfig, RecoveryDecision, ResolvedConfig
from aiython.frontend import parse
from aiython import native_bridge
from aiython.native_bridge import NativeTypeBridge, vm_available
from aiython.runtime import Runtime
from aiython.type_constraints import TypeViolation


class NativeBridgeTests(unittest.TestCase):
    def test_callback_store_initializes_module_scope_without_call_event(self):
        bridge = NativeTypeBridge()
        source = ('value: int = 1\n'
                  'bridge.before_store(__import__("inspect").currentframe(), '
                  '"value", "invalid")\n')
        namespace = {'bridge': bridge}
        with self.assertRaises(TypeViolation):
            exec(bridge.compile_source(source, '<store-without-call>'), namespace)
        self.assertIs(bridge.types._module_scopes[id(namespace)][0], namespace)
        bridge.before_store(__import__('inspect').currentframe(), 'unknown', 1)
        bridge.on_call(__import__('inspect').currentframe())
        bridge.on_return(__import__('inspect').currentframe(), 1)

    def test_native_source_detects_dynamic_type_boundaries(self):
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        try:
            for source in ('answer = __builtins__\n',
                           'def answer(value: int):\n    return value\n'):
                with self.subTest(source=source):
                    self.assertFalse(runtime.bridge.native_source(parse(source, '<dynamic>')))
            if hasattr(__import__('ast'), 'TypeAlias'):
                self.assertFalse(runtime.bridge.native_source(
                    parse('type Answer = int\n', '<type-alias>')))
            parameterized = parse('pass\n', '<type-parameters>')
            parameterized.tree.body[0].type_params = [object()]
            self.assertFalse(runtime.bridge.native_source(parameterized))
            with patch.object(ast, 'TypeAlias', ast.Pass, create=True):
                self.assertFalse(runtime.bridge.native_source(
                    parse('pass\n', '<type-alias-probe>')))
        finally:
            runtime.capabilities.close()

    def test_vm_probe_and_scoped_hook_lifecycle_with_emulated_event(self):
        def emit_probe(code, namespace):
            unrelated = SimpleNamespace(f_code=SimpleNamespace(co_filename='<other>'))
            sys._aiython_before_store(unrelated, 'other', 1)
            frame = SimpleNamespace(f_code=SimpleNamespace(co_filename='<aiython-vm-probe>'))
            sys._aiython_before_store(frame, 'probe_value', 1)

        previous_availability = native_bridge._VM_AVAILABLE
        previous_hook = getattr(sys, '_aiython_before_store', None)
        try:
            sys._aiython_before_store = lambda frame, name, value: None
            with patch.object(native_bridge, '_VM_AVAILABLE', None), \
                    patch.object(native_bridge, 'exec', side_effect=emit_probe, create=True):
                self.assertTrue(native_bridge.vm_available())
                bridge = NativeTypeBridge()
                with bridge.installed():
                    self.assertIs(sys._aiython_before_store.__self__, bridge)
        finally:
            native_bridge._VM_AVAILABLE = previous_availability
            if previous_hook is None:
                vars(sys).pop('_aiython_before_store', None)
            else:
                sys._aiython_before_store = previous_hook

    def test_vm_availability_double_check_handles_concurrent_probe(self):
        class ConcurrentProbe:
            def __enter__(self):
                native_bridge._VM_AVAILABLE = True

            def __exit__(self, kind, error, traceback):
                return False

        with patch.object(native_bridge, '_VM_AVAILABLE', None), \
                patch.object(native_bridge, '_VM_LOCK', ConcurrentProbe()):
            self.assertTrue(native_bridge.vm_available())

    def test_unregistered_vm_events_do_not_call_previous_handlers(self):
        frame = SimpleNamespace(f_code=compile('pass', '<unknown-vm-code>', 'exec'))
        with patch.object(native_bridge, '_VM_PREVIOUS_STORE', None), \
                patch.object(native_bridge, '_VM_PREVIOUS_CALL', None):
            native_bridge._vm_on_call(frame)
            native_bridge._vm_before_store(frame, 'value', 1)

    def test_expired_code_cannot_remove_a_new_registry_entry(self):
        bridge = NativeTypeBridge()
        newer = compile('pass', '<newer-code>', 'exec')

        def register_then_replace():
            old = compile('pass', '<older-code>', 'exec')
            with patch.object(native_bridge, '_VM_HOOK_INSTALLED', True):
                native_bridge._register_vm_code(old, bridge, 'old')
            key = id(old)
            original_reference = native_bridge._VM_CODES[key][0]
            native_bridge._VM_CODES[key] = (weakref.ref(newer), bridge, 'new')
            return key, original_reference

        key, original_reference = register_then_replace()
        try:
            gc.collect()
            self.assertIsNone(original_reference())
            self.assertEqual(native_bridge._VM_CODES[key][2], 'new')
        finally:
            native_bridge._VM_CODES.pop(key, None)

    def test_variadic_rebinding_is_checked_locally_and_through_closures(self):
        source = '''def outer(*values: int):
    def middle():
        def inner():
            nonlocal values
            bridge.before_store(__import__('inspect').currentframe(), 'values', ('new',))
            values = ('new',)
            return values
        return inner
    return middle()
def rebind_args(*values: int):
    bridge.before_store(__import__('inspect').currentframe(), 'values', ('bad',))
def rebind_kwargs(**named: int):
    bridge.before_store(__import__('inspect').currentframe(), 'named', {'x': 'bad'})
def grandparent():
    value: int = 1
    def middle():
        def inner():
            nonlocal value
            bridge.before_store(__import__('inspect').currentframe(), 'value', 2)
            value = 2
            return value
        return inner
    return middle()
def typed_outer(value: int):
    def inner():
        nonlocal value
        bridge.before_store(__import__('inspect').currentframe(), 'value', 'bad')
    return inner()
def untyped_outer(value):
    def inner():
        nonlocal value
        bridge.before_store(__import__('inspect').currentframe(), 'value', 'allowed')
        value = 'allowed'
        return value
    return inner()
def no_parameters():
    bridge.on_call(__import__('inspect').currentframe())
'''
        bridge = NativeTypeBridge()
        namespace = {'bridge': bridge}
        exec(bridge.compile_source(source, '<outer-variadic>'), namespace)
        for action in (lambda: namespace['outer'](1)(),
                       lambda: namespace['rebind_args'](1),
                       lambda: namespace['rebind_kwargs'](x=1),
                       lambda: namespace['typed_outer'](1)):
            with self.assertRaises(TypeViolation):
                action()
        self.assertEqual(namespace['grandparent']()(), 2)
        self.assertEqual(namespace['untyped_outer'](1), 'allowed')
        namespace['no_parameters']()
        bridge.before_mutation(__import__('inspect').currentframe(), 'item', {}, 'key', 1)

    def test_first_final_store_is_allowed_before_binding_exists(self):
        source = '''from typing import Final
bridge.before_store(__import__('inspect').currentframe(), 'fixed', 1)
fixed: Final[int] = 1
'''
        bridge = NativeTypeBridge()
        namespace = {'bridge': bridge}
        exec(bridge.compile_source(source, '<first-final-store>'), namespace)
        self.assertEqual(namespace['fixed'], 1)

    def test_vm_yield_monitor_fallback_and_exhausted_tool_ids(self):
        bridge = NativeTypeBridge()
        with patch.object(sys, 'monitoring', None, create=True):
            with bridge._yield_monitor():
                pass
        busy = SimpleNamespace(events=SimpleNamespace(PY_YIELD=1),
                               get_tool=lambda number: 'busy')
        with patch.object(sys, 'monitoring', busy, create=True):
            with self.assertRaisesRegex(RuntimeError, 'No free sys.monitoring tool ID'):
                with bridge._yield_monitor():
                    pass
        events = []
        free = SimpleNamespace(
            events=SimpleNamespace(PY_YIELD=8),
            get_tool=lambda number: None,
            use_tool_id=lambda number, name: events.append(('use', number, name)),
            register_callback=lambda number, event, callback: events.append(('callback', callback)),
            set_events=lambda number, event: events.append(('events', event)),
            free_tool_id=lambda number: events.append(('free', number)),
        )
        with patch.object(sys, 'monitoring', free, create=True):
            with bridge._yield_monitor():
                self.assertEqual(events[0][0], 'use')
        self.assertEqual([entry[0] for entry in events],
                         ['use', 'callback', 'events', 'events', 'callback', 'free'])

    def test_vm_route_registers_exact_code_for_each_runtime(self):
        runtimes = [Runtime(ResolvedConfig(None, Path.cwd()))]
        profile = ProfileConfig('default', 'fake', 'fake')
        runtimes.append(Runtime(ResolvedConfig(None, Path.cwd(), 'default',
                                               {'default': profile})))
        hooks = {name: getattr(sys, name, None) for name in
                 ('_aiython_before_store', '_aiython_on_call')}
        installed = native_bridge._VM_HOOK_INSTALLED
        previous = (native_bridge._VM_PREVIOUS_STORE, native_bridge._VM_PREVIOUS_CALL)
        try:
            with patch.object(native_bridge, 'vm_available', return_value=True):
                for runtime in runtimes:
                    code = runtime.compile_source('value: int = 1\n', '<vm-default-route>')
                    self.assertTrue(runtime.bridge.uses_vm(code))
                    self.assertEqual(runtime.bridge._frame_types(
                        SimpleNamespace(f_code=code)).declarations, {'value': 'int'})
                    self.assertEqual(runtime.units['<vm-default-route>'].source,
                                     'value: int = 1\n')
        finally:
            for runtime in runtimes:
                runtime.capabilities.close()
            native_bridge._VM_HOOK_INSTALLED = installed
            native_bridge._VM_PREVIOUS_STORE, native_bridge._VM_PREVIOUS_CALL = previous
            for name, value in hooks.items():
                if value is None:
                    vars(sys).pop(name, None)
                else:
                    setattr(sys, name, value)

    def test_bridge_requires_runtime_to_select_managed_code(self):
        bridge = NativeTypeBridge()
        unit = parse('value: int = 1\n', '<no-runtime>')
        with self.assertRaisesRegex(RuntimeError, 'requires a Runtime'):
            bridge.native_source(unit)
        with self.assertRaisesRegex(RuntimeError, 'requires a Runtime'):
            bridge.prepare_unit(unit)

    def test_scoped_native_hooks_restore_prior_sys_callbacks(self):
        bridge = NativeTypeBridge()
        source = 'value: int = 1\nvalue = "invalid"\n'
        code = bridge.compile_source(source, '<scoped-vm-hooks>')
        missing = object()
        previous = {name: getattr(sys, name, missing) for name in bridge._HOOKS}
        if vm_available():
            with bridge.installed():
                with self.assertRaises(TypeViolation):
                    exec(code, {})
        else:
            with self.assertRaisesRegex(RuntimeError, 'requires a CPython build'):
                with bridge.installed():
                    pass
        for name, value in previous.items():
            self.assertIs(getattr(sys, name, missing), value)

    def test_vm_dispatch_during_interpreter_shutdown_is_safe(self):
        frame = SimpleNamespace(f_code=compile('pass', '<shutdown>', 'exec'))
        with patch.object(native_bridge, '_VM_LOCK', None):
            native_bridge._vm_on_call(frame)
            native_bridge._vm_before_store(frame, 'value', 1)

    def test_yield_and_generator_return_callbacks_enforce_contracts(self):
        source = '''from typing import Generator
def yielded(value) -> Generator[int, None, None]:
    bridge.on_yield(None, None, value)
def bad_annotation(value) -> int:
    bridge.on_yield(None, None, value)
def returned(value) -> Generator[int, None, int]:
    bridge.on_return(__import__('inspect').currentframe(), value)
    yield 1
'''
        bridge = NativeTypeBridge()
        namespace = {'bridge': bridge}
        exec(bridge.compile_source(source, '<yield-callbacks>'), namespace)
        namespace['yielded'](1)
        with self.assertRaises(TypeViolation):
            namespace['yielded']('invalid')
        with self.assertRaises(TypeViolation):
            namespace['bad_annotation'](1)
        self.assertEqual(list(namespace['returned'](1)), [1])
        with self.assertRaises(TypeViolation):
            list(namespace['returned']('invalid'))
        bridge.on_yield(None, None, 'outside tracked code')

    def test_callbacks_reject_invalid_parameters_returns_globals_and_mutations(self):
        source = '''from typing import Final, Generator
answer: int = 1
fixed: Final[int] = 1
class Box:
    value: int
    def __init__(self):
        self.value = 1
def parameter(value: int):
    bridge.on_call(__import__('inspect').currentframe())
    bridge.before_store(__import__('inspect').currentframe(), 'value', 'invalid')
def returned(value) -> int:
    bridge.on_return(__import__('inspect').currentframe(), value)
def change_global(value):
    global answer
    bridge.before_store(__import__('inspect').currentframe(), 'answer', value)
    answer = value
def change_final():
    global fixed
    bridge.before_store(__import__('inspect').currentframe(), 'fixed', 2)
    fixed = 2
def change_attribute(box, value):
    bridge.before_mutation(__import__('inspect').currentframe(), 'attr', box, 'value', value)
    box.value = value
def outer():
    current: int = 1
    def change(value):
        nonlocal current
        bridge.before_store(__import__('inspect').currentframe(), 'current', value)
        current = value
    return change
'''
        bridge = NativeTypeBridge()
        namespace = {'bridge': bridge}
        exec(bridge.compile_source(source, '<manual-contracts>'), namespace)
        with self.assertRaises(TypeViolation):
            namespace['parameter']('invalid')
        with self.assertRaises(TypeViolation):
            namespace['parameter'](1)
        with self.assertRaises(TypeViolation):
            namespace['returned']('invalid')
        namespace['returned'](1)
        with self.assertRaises(TypeViolation):
            namespace['change_global']('invalid')
        namespace['change_global'](2)
        self.assertEqual(namespace['answer'], 2)
        with self.assertRaises(TypeViolation):
            namespace['change_final']()
        box = namespace['Box']()
        with self.assertRaises(TypeViolation):
            namespace['change_attribute'](box, 'invalid')
        namespace['change_attribute'](box, 2)
        self.assertEqual(box.value, 2)
        setter = namespace['outer']()
        with self.assertRaises(TypeViolation):
            setter('invalid')
        setter(3)

    def test_vm_indexes_decorated_nested_class_and_variadic_frames(self):
        source = '''def identity(function):
    return function
@identity
def checked(value: int, /, *rest: int, named: int, **options: int) -> int:
    local: int = value
    return local
class Box:
    field: int
    def method(self, value: int) -> int:
        return value
async def async_checked(value: int) -> int:
    return value
def outer():
    current: int = 1
    def inner(value: int):
        nonlocal current
        current = value
    return inner
def change_global(value: int):
    global answer
    answer = value
'''
        bridge = NativeTypeBridge()
        code = bridge.compile_source(source, '<indexed-frames>')
        self.assertEqual(code, compile(source, '<indexed-frames>', 'exec', dont_inherit=True))
        frames = bridge.frames
        self.assertIs(frames[('<indexed-frames>', 'checked', 3)],
                      frames[('<indexed-frames>', 'checked', 4)])
        self.assertEqual(frames[('<indexed-frames>', 'checked', 4)].parameters,
                         {'value': ('int', 'value'), 'rest': ('int', 'args'),
                          'named': ('int', 'value'), 'options': ('int', 'kwargs')})
        self.assertEqual(frames[('<indexed-frames>', 'Box.method', 9)].returns, 'int')
        self.assertIn(('<indexed-frames>', 'async_checked', 11), frames)
        self.assertEqual(frames[('<indexed-frames>', 'outer.<locals>.inner', 15)]
                         .outer.declarations['current'], 'int')
        self.assertEqual(frames[('<indexed-frames>', 'change_global', 19)]
                         .global_names, frozenset({'answer'}))

    def test_vm_routing_only_selects_supported_module_shapes(self):
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        try:
            supported = (
                'value: int = 1\n',
                'value: int\nvalue = 1\n',
                '"module docstring"\npass\nvalue: bytes = b"ok"\n',
                'previous = 1\nvalue: int = previous\n',
            )
            unsupported = (
                'value = 1\n',
                'value: list[int] = []\n',
                'value: Other = 1\n',
                'value: int = int("1")\n',
                'value: int = 1\nif True: value = 2\n',
                'value: int = 1\nvalue, other = 2, 3\n',
            )
            with patch.object(native_bridge, 'vm_available', return_value=True):
                for source in supported:
                    with self.subTest(source=source):
                        self.assertTrue(runtime.bridge.vm_source(parse(source, '<vm-routing>')))
                for source in unsupported:
                    with self.subTest(source=source):
                        self.assertFalse(runtime.bridge.vm_source(parse(source, '<vm-routing>')))
            with patch.object(native_bridge, 'vm_available', return_value=False):
                self.assertFalse(runtime.bridge.vm_source(
                    parse('value: int = 1\n', '<vm-routing>')))
        finally:
            runtime.capabilities.close()

    def test_vm_recovery_compiler_retries_original_statement(self):
        class Agent:
            def recover(self, request, bridge):
                self.exception = request.exception
                return RecoveryDecision('complete', 42, True)

        agent = Agent()
        profile = ProfileConfig('default', 'fake', 'fake')
        config = ResolvedConfig(None, Path.cwd(), 'default', {'default': profile})
        runtime = Runtime(config, agent_factory=lambda _: agent)
        try:
            source = ('bridge.on_call(__import__("inspect").currentframe())\n'
                      'value: int = 1\nanswer = missing_value\n')
            unit = parse(source, '<vm-recovery-compiler>')
            code = runtime.bridge._compile_vm_recovery(unit)
            namespace = {'bridge': runtime.bridge}
            exec(code, namespace)
            self.assertEqual((namespace['value'], namespace['answer']), (1, 42))
            self.assertIsInstance(agent.exception, NameError)
        finally:
            runtime.capabilities.close()

    def test_managed_vm_dispatch_is_bound_to_code_identity(self):
        first_code = compile('value: int = 1', '<first-vm-dispatch>', 'exec')
        second_code = compile('value: int = 1', '<first-vm-dispatch>', 'exec')
        self.assertEqual(first_code, second_code)
        self.assertIsNot(first_code, second_code)
        target_code = weakref.ref(first_code)
        bridge = Mock()
        previous_events = []

        def previous_store(frame, name, value):
            if frame.f_code is target_code():
                previous_events.append(('store', name, value))

        def previous_call(frame):
            if frame.f_code is target_code():
                previous_events.append(('call',))
        original_store = getattr(sys, '_aiython_before_store', None)
        original_call = getattr(sys, '_aiython_on_call', None)
        installed = native_bridge._VM_HOOK_INSTALLED
        saved_previous = (native_bridge._VM_PREVIOUS_STORE,
                          native_bridge._VM_PREVIOUS_CALL)
        try:
            native_bridge._VM_HOOK_INSTALLED = False
            sys._aiython_before_store = previous_store
            sys._aiython_on_call = previous_call
            native_bridge._register_vm_code(first_code, bridge, 'info')
            self.assertIs(sys._aiython_before_store, native_bridge._vm_before_store)
            self.assertIs(sys._aiython_on_call, native_bridge._vm_on_call)
            self.assertEqual(native_bridge._vm_entry(first_code)[2], 'info')
            self.assertIsNone(native_bridge._vm_entry(second_code))
            frame = SimpleNamespace(f_code=first_code)
            native_bridge._vm_on_call(frame)
            native_bridge._vm_before_store(frame, 'value', 1)
            bridge.on_call.assert_called_once_with(frame)
            bridge.before_store.assert_called_once_with(frame, 'value', 1)
            self.assertEqual(previous_events, [('call',), ('store', 'value', 1)])
            native_bridge._vm_on_call(SimpleNamespace(f_code=second_code))
            native_bridge._vm_before_store(SimpleNamespace(f_code=second_code), 'value', 2)
            self.assertEqual(bridge.on_call.call_count, 1)
            self.assertEqual(bridge.before_store.call_count, 1)
            bridge.reset_mock()
            del frame, first_code
            gc.collect()
            self.assertFalse(any(entry[1] is bridge for entry in native_bridge._VM_CODES.values()))
        finally:
            native_bridge._VM_HOOK_INSTALLED = installed
            native_bridge._VM_PREVIOUS_STORE, native_bridge._VM_PREVIOUS_CALL = saved_previous
            if original_store is None:
                vars(sys).pop('_aiython_before_store', None)
            else:
                sys._aiython_before_store = original_store
            if original_call is None:
                vars(sys).pop('_aiython_on_call', None)
            else:
                sys._aiython_on_call = original_call

    def test_callbacks_validate_module_function_closure_and_return(self):
        source = '''from typing import Final, Generator
answer: int = 1
fixed: Final[int] = 1
class Box:
    value: int
def typed(value: int) -> int:
    return value
def change_global(value):
    global answer
    answer = value
def outer():
    current: int = 1
    def change(value):
        nonlocal current
        current = value
    return change
def yields() -> Generator[int, None, None]:
    yield 1
'''
        bridge = NativeTypeBridge()
        code = bridge.compile_source(source, '<callback-contracts>')
        self.assertEqual(code, compile(source, '<callback-contracts>', 'exec', dont_inherit=True))
        namespace = {}
        exec(code, namespace)
        self.assertIn('<callback-contracts>', bridge.module_types)

        # A module code object is needed for the same frame metadata as VM hooks.
        probe = compile('bridge.on_call(__import__("inspect").currentframe())\n'
                        'bridge.before_store(__import__("inspect").currentframe(), '
                        '"answer", "invalid")', '<callback-contracts>', 'exec')
        with self.assertRaises(TypeViolation):
            exec(probe, {'bridge': bridge})

        function_code = namespace['typed'].__code__
        self.assertEqual(bridge.frames[(function_code.co_filename,
                                        function_code.co_qualname,
                                        function_code.co_firstlineno)].returns, 'int')

        # Use the exact compiled function frame while testing callbacks on stock Python.
        callback_source = '''def typed(value: int) -> int:
    bridge.on_call(__import__('inspect').currentframe())
    bridge.before_store(__import__('inspect').currentframe(), 'value', value)
    bridge.on_return(__import__('inspect').currentframe(), value)
'''
        callback_bridge = NativeTypeBridge()
        callback_namespace = {'bridge': callback_bridge}
        exec(callback_bridge.compile_source(callback_source, '<manual-callbacks>'), callback_namespace)
        callback_namespace['typed'](1)
        with self.assertRaises(TypeViolation):
            callback_namespace['typed']('invalid')

    @unittest.skipUnless(vm_available(), 'requires Aiython VM hooks')
    def test_vm_store_and_ai_recovery_share_the_default_execution_path(self):
        class Agent:
            def __init__(self):
                self.errors = []

            def recover(self, request, bridge):
                self.errors.append(request.exception)
                return RecoveryDecision('complete', 42, True)

        agent = Agent()
        profile = ProfileConfig('default', 'fake', 'fake')
        config = ResolvedConfig(None, Path.cwd(), 'default', {'default': profile})
        runtime = Runtime(config, agent_factory=lambda _: agent)
        try:
            code = runtime.compile_source(
                'value: int = 1\nanswer = missing_value\n', '<vm-recovery>', entry=True)
            self.assertTrue(runtime.bridge.uses_vm(code))
            namespace = {}
            exec(code, namespace)
            self.assertEqual((namespace['value'], namespace['answer']), (1, 42))
            self.assertEqual(len(agent.errors), 1)
            self.assertIsInstance(agent.errors[0], NameError)
            checking = Runtime(config, agent_factory=lambda _: agent)
            try:
                bad = checking.compile_source(
                    'value: int = 1\nvalue = "invalid"\n', '<vm-type-error>', entry=True)
                self.assertTrue(checking.bridge.uses_vm(bad))
                with self.assertRaises(TypeViolation):
                    exec(bad, {})
                self.assertEqual(len(agent.errors), 1)
            finally:
                checking.capabilities.close()
        finally:
            runtime.capabilities.close()

    @unittest.skipUnless(vm_available(), 'requires Aiython VM hooks')
    def test_default_vm_store_callbacks_keep_runtimes_separate(self):
        source = 'value: int = 1\nvalue = "invalid"\n'
        filename = '<managed-vm-store>'
        first = Runtime(ResolvedConfig(None, Path.cwd()))
        second = Runtime(ResolvedConfig(None, Path.cwd()))
        try:
            first_code = first.compile_source(source, filename)
            second_code = second.compile_source(source, filename)
            self.assertEqual(first_code, compile(source, filename, 'exec', dont_inherit=True))
            self.assertIsNot(first_code, second_code)
            self.assertTrue(first.bridge.uses_vm(first_code))
            self.assertTrue(second.bridge.uses_vm(second_code))
            with patch.object(first.bridge, 'before_store', wraps=first.bridge.before_store) as first_hook, \
                    patch.object(second.bridge, 'before_store', wraps=second.bridge.before_store) as second_hook:
                with self.assertRaises(TypeViolation):
                    exec(first_code, {})
                first_calls = first_hook.call_count
                self.assertGreater(first_calls, 0)
                self.assertEqual(second_hook.call_count, 0)
                with self.assertRaises(TypeViolation):
                    exec(second_code, {})
                self.assertEqual(first_hook.call_count, first_calls)
                self.assertGreater(second_hook.call_count, 0)
        finally:
            first.capabilities.close()
            second.capabilities.close()

    @unittest.skipUnless(hasattr(sys, 'monitoring'), 'requires Python 3.12+')
    def test_yield_monitor_rejects_value_and_runs_generator_finally(self):
        source = '''from typing import Generator
cleanup = []
def values() -> Generator[int, None, None]:
    try:
        yield from (1, "invalid")
    finally:
        cleanup.append("closed")
'''
        filename = '<bridge-yield-monitor>'
        bridge = NativeTypeBridge()
        code = bridge.compile_source(source, filename)
        self.assertEqual(code, compile(source, filename, 'exec', dont_inherit=True))
        namespace = {}
        before = [sys.monitoring.get_tool(number) for number in range(6)]
        with bridge._yield_monitor():
            exec(code, namespace)
            with self.assertRaisesRegex(TypeViolation, 'yield'):
                list(namespace['values']())
        self.assertEqual(namespace['cleanup'], ['closed'])
        self.assertEqual([sys.monitoring.get_tool(number) for number in range(6)], before)
