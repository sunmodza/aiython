import ast
from collections import OrderedDict
import inspect
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiython import runtime as rt
from aiython.frontend import parse
from aiython.models import (
    AgentRequest, AiythonError, ProfileConfig, RecoveryDecision, ResolvedConfig, SourceSpan,
)
from aiython.type_constraints import TypeViolation


class FrameNamespaceEdgeTests(unittest.TestCase):
    def test_namespace_global_local_set_delete_and_iteration(self):
        frame = SimpleNamespace(f_locals={'local': 1}, f_globals={'global_value': 2})
        sync = Mock()
        with patch.object(rt, '_LOCALS_TO_FAST', sync):
            namespace = rt.FrameNamespace(frame, {'global_value'})
            self.assertEqual(dict(namespace), {'local': 1, 'global_value': 2})
            self.assertEqual(len(namespace), 2)
            namespace['local'] = 3
            namespace['global_value'] = 4
            del namespace['local']
            del namespace['global_value']
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(frame.f_globals, {})


class RuntimeBridgeEdgeTests(unittest.TestCase):
    def setUp(self):
        self.manager = rt.Runtime(ResolvedConfig(None, Path.cwd()))
        self.bridge = rt.RuntimeBridge(inspect.currentframe(), self.manager)

    def test_eval_code_object_binding_and_unknown_handle_validation(self):
        self.assertEqual(self.bridge.eval(compile('1 + 2', '<test>', 'eval')), 3)
        for name in ('not a name',):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'non-reserved'):
                self.bridge.set(name, 1)
        self.bridge.set('__aiython_runtime__', 1)
        self.assertEqual(self.bridge.eval('__aiython_runtime__'), 1)
        with self.assertRaisesRegex(ValueError, 'Unknown object handle'):
            self.bridge.dereference('missing')
        for depth, limit in ((0, 1), (1, 101), (True, 1)):
            with self.subTest(depth=depth, limit=limit), self.assertRaisesRegex(ValueError, 'inspect requires'):
                self.bridge.inspect('missing', depth=depth, limit=limit)

    def test_inspect_budget_and_user_object_metadata(self):
        mapping = {str(index): list(range(20)) for index in range(100)}
        handle = self.bridge.handle(mapping)['handle']
        result = self.bridge.inspect(handle, depth=3, limit=100)
        self.assertTrue(result['truncated'])
        values = [[{'value': index} for index in range(20)] for _ in range(100)]
        result = self.bridge.inspect(self.bridge.handle(values)['handle'], depth=3, limit=100)
        self.assertTrue(result['truncated'])

        class Slot:
            __slots__ = ('value',)
            def __init__(self): self.value = 3
        result = self.bridge.inspect(self.bridge.handle(Slot())['handle'])
        self.assertIn('attributes', result)

        class WeirdDescriptor:
            def __get__(self, instance, owner): return 5
        class Weird:
            __dict__ = WeirdDescriptor()
        with patch.object(rt.types, 'GetSetDescriptorType', WeirdDescriptor):
            result = self.bridge.inspect(self.bridge.handle(Weird())['handle'])
        self.assertIn('attributes', result)

    def test_traceback_frame_listing_and_source_selection(self):
        frame = inspect.currentframe()
        self.bridge.frame = frame
        self.manager.units[frame.f_code.co_filename] = None
        self.bridge.traceback = SimpleNamespace(tb_frame=frame, tb_lineno=frame.f_lineno,
                                                tb_next=None)
        listed = self.bridge.frames()
        self.assertEqual(len(listed), 1)
        self.assertTrue(listed[0]['active'])
        self.assertIs(self.bridge._source_frame('origin'), frame)
        unregistered = SimpleNamespace(f_code=SimpleNamespace(co_filename='<unregistered>'))
        self.bridge.traceback = SimpleNamespace(tb_frame=unregistered, tb_lineno=1,
            tb_next=SimpleNamespace(tb_frame=frame, tb_lineno=frame.f_lineno, tb_next=None))
        self.assertEqual(len(self.bridge.frames()), 1)
        self.assertIs(self.bridge._source_frame('origin'), frame)
        with self.assertRaisesRegex(ValueError, 'active or origin'):
            self.bridge._source_frame('invalid')
        self.bridge.traceback = None
        self.assertIsNone(self.bridge._source_frame('origin'))
        self.assertFalse(self.bridge.get_frame_code('origin')['available'])


class RuntimePreparationEdgeTests(unittest.TestCase):
    def setUp(self):
        self.manager = rt.Runtime(ResolvedConfig(None, Path.cwd()))

    def test_large_source_revision_and_bounded_revision_cache(self):
        large = 'x' * (256 * 1024 + 1)
        self.assertEqual(len(self.manager.source_revision(large)), 64)
        for index in range(129):
            self.manager.source_revision(f'value-{index}')
        self.assertEqual(len(self.manager._source_revisions), 128)
        self.assertNotIn('value-0', self.manager._source_revisions)

    def test_large_source_skips_preparation_cache_and_small_cache_evicts(self):
        filename = str(Path.cwd() / 'runtime-cache-test.py')
        with patch.object(rt, '_PREPARED', OrderedDict()), patch.object(rt, '_PREPARED_LIMIT', 0):
            self.manager.compile_source('x = 1', filename)
            self.assertFalse(rt._PREPARED)
        large = 'x = 1\n' + ('# padding\n' * 30000)
        with patch.object(self.manager, 'prepare', return_value=compile('pass', filename, 'exec')), \
                patch('aiython.frontend.parse', return_value=parse('x = 1', filename)):
            self.manager.compile_source(large, filename)
        self.assertTrue(large.startswith('x = 1'))

    def test_bound_code_cache_can_be_bypassed_and_evicted(self):
        source = compile('answer = 1', '<bound-code-cache>', 'exec')
        self.manager._bind_compiled(source, '__aiython_runtime__', cache=False)
        with patch.object(rt, '_PREPARED_LIMIT', 0):
            self.manager._bind_compiled(source, '__aiython_runtime__')
            self.assertFalse(self.manager._bound_codes)

    def test_dynamic_name_fallback_and_local_lookup(self):
        transformer = rt.DynamicNames(parse('x = 1', 'test.py'))
        node = ast.Name('x', ast.Load())
        self.assertIs(transformer.in_scope('absent', 1, [node])[0], node)
        def outer():
            secret = 42
            def middle():
                return next(self.manager.lookup('secret') for _ in [0])
            return [middle()]
        self.assertEqual(outer(), [42])
        def builtins_from_generator():
            return next(self.manager.lookup('len') for _ in [0])
        self.assertIs(builtins_from_generator(), len)
        with self.assertRaisesRegex(NameError, 'missing'):
            self.manager.lookup('missing')

    def test_ast_transformers_cover_nested_match_lambda_and_comprehensions(self):
        source = '''async def work():
    pick = lambda value=1, *, extra=None: value
    a = {item for item in [1]}
    b = {item: item for item in [1]}
    c = (item for item in [1])
    match 1:
        case 1:
            print('one')
'''
        unit = parse(source, 'runtime-transformers.py')
        rt.DynamicNames(unit).visit(ast.parse(source))
        rt.AsyncCalls().visit(ast.parse(source))
        transformed = rt.NestedCheckpoints(self.manager, unit).visit(ast.parse(source))
        self.assertIsInstance(transformed, ast.Module)
        transformer = rt.DynamicNames(parse('x = 1', 'test.py'))
        transformer.table = SimpleNamespace(get_type=lambda: 'function',
                                            lookup=Mock(side_effect=KeyError('missing')))
        self.assertEqual(transformer.visit_Name(ast.Name('missing', ast.Load())).id, 'missing')
        without_child = rt.DynamicNames(parse('x = 1', 'test.py'))
        without_child.comprehension(ast.parse('{x for x in [1]}', mode='eval').body,
                                    'setcomp')

    def test_legacy_cpython_frame_support_import(self):
        for version, expected in (((3, 12), True), ((3, 14), False)):
            module = types.ModuleType('aiython._runtime_compatibility')
            module.__package__ = 'aiython'
            module.__file__ = rt.__file__
            sys.modules[module.__name__] = module
            try:
                with patch.object(sys, 'version_info', version):
                    exec(compile(Path(rt.__file__).read_text(), rt.__file__, 'exec'), module.__dict__)
            finally:
                sys.modules.pop(module.__name__)
            self.assertEqual(module._LOCALS_TO_FAST is not None, expected)


class RuntimeRecoveryEdgeTests(unittest.TestCase):
    def setUp(self):
        profile = ProfileConfig('test', 'fake', 'model')
        self.manager = rt.Runtime(ResolvedConfig(None, Path.cwd(), 'test', {'test': profile}))
        self.unit = parse('answer = 1 / 0', 'recovery-test.py')
        self.span = SourceSpan(self.unit.filename, 1, 0, 1, 14)
        self.manager.units[self.unit.filename] = self.unit
        self.manager.checkpoints['failure'] = rt.Checkpoint(
            self.unit, self.span, 'answer = 1 / 0', 'answer')
        self.request = AgentRequest('answer = 1 / 0', self.unit.source, {}, {},
                                    self.span, profile, ())
        self.agent = SimpleNamespace(recover=Mock(return_value=RecoveryDecision('complete')))

    def recover(self, error, decision, *, attempt=1, target='answer'):
        self.manager.checkpoints['failure'].target = target
        self.agent.recover.return_value = decision
        with patch.object(self.manager, 'request', return_value=(self.request, self.agent)):
            return self.manager.recover('failure', error, attempt)

    def test_nonrecoverable_error_gets_location_once_and_missing_origin_passes_through(self):
        first = TypeViolation('wrong type')
        with self.assertRaises(TypeViolation) as caught:
            self.manager.recover('failure', first)
        self.assertIs(caught.exception, first)

        self.manager.units[__file__] = parse('pass', __file__)
        try:
            raise TypeViolation('invalid value')
        except TypeViolation as error:
            with self.assertRaises(TypeViolation):
                self.manager.recover('failure', error)
            self.assertTrue(error._aiython_location)
            self.assertIn(__file__, str(error))

    def test_unscoped_checkpoint_and_python_try_handler(self):
        node = self.unit.tree.body[0]
        checkpoint = rt.install_checkpoint(self.manager, self.unit, node,
                                           'unscoped', scoped_retries=False)
        tree = ast.fix_missing_locations(ast.Module(body=[checkpoint], type_ignores=[]))
        code = compile(tree, self.unit.filename, 'exec')
        namespace = {self.unit.runtime_name: self.manager}
        with patch.object(self.manager, 'recover', return_value=False) as recover:
            exec(code, namespace)
        self.assertEqual(recover.call_args.args[2], 1)

        source = '''try:
    answer = 1
except ValueError:
    answer = probe()
else:
    answer = probe()
finally:
    cleaned = True
'''
        filename = '<protected-try-handler>'
        self.manager.units[filename] = parse(source, filename)

        def probe():
            return self.manager.caller_has_python_handler(inspect.currentframe())

        namespace = {'probe': probe}
        exec(compile(source, filename, 'exec'), namespace)
        self.assertTrue(namespace['answer'])
        self.assertTrue(namespace['cleaned'])

    def test_recovery_without_profile_reraises_and_retained_scope_is_found(self):
        from aiython.typed_runtime import Scope

        runtime = rt.Runtime(ResolvedConfig(None, Path.cwd()))
        error = ValueError('original')
        with self.assertRaises(ValueError) as caught:
            runtime.recover('missing', error)
        self.assertIs(caught.exception, error)
        frame = inspect.currentframe()
        scope = Scope()
        runtime.types._module_scopes[id(frame.f_globals)] = (frame.f_globals, scope)
        self.assertIs(runtime.recovery_scope(frame), scope)
        runtime.capabilities.close()

    def test_fallback_attempt_counter_and_origin_traceback(self):
        self.assertTrue(self.recover(ValueError('failed'), RecoveryDecision('retry'), attempt=None))
        try:
            raise ValueError('origin failed')
        except ValueError as error:
            self.manager.units[__file__] = parse('pass', __file__)
            self.assertFalse(self.recover(error, RecoveryDecision('complete')))
            self.assertEqual(self.agent.recover.call_args.args[0].origin.filename, __file__)
        try:
            json.loads('{')
        except json.JSONDecodeError as error:
            self.assertFalse(self.recover(error, RecoveryDecision('complete')))

    def test_invalid_decision_reraise_note_and_replacement_guard(self):
        with self.assertRaisesRegex(AiythonError, 'invalid recovery decision'):
            self.recover(ValueError('bad'), object())
        error = ValueError('original')
        with self.assertRaisesRegex(ValueError, 'original'):
            self.recover(error, RecoveryDecision('reraise', explanation='try another source'))
        self.assertIn('try another source', error.__notes__)
        with self.assertRaisesRegex(ValueError, 'without note'):
            self.recover(ValueError('without note'), RecoveryDecision('reraise'))
        with self.assertRaisesRegex(AiythonError, 'single-name assignment'):
            self.recover(ValueError('bad'), RecoveryDecision('complete', 42, True), target=None)
        self.assertFalse(self.recover(ValueError('bad'), RecoveryDecision('complete', 42, True)))


if __name__ == '__main__':
    unittest.main()
