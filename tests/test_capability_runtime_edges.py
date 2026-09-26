"""Capability contracts, cache recovery, and local index boundaries."""

import contextlib
import io
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiython.assets import Document, Image, VectorIndex
from aiython.capabilities import (
    CapabilityError, CapabilityRequest, CapabilityResult, CapabilityRuntime, CapabilitySpec,
    Embeddings, InvocationError, LocalProvider, Registry, Store, decode, encode,
)
from aiython.models import ConfigError, ProfileConfig, ResolvedConfig
from aiython.stats import CURRENT_STATS, InvocationStats


class CapabilityRuntimeEdgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = ProfileConfig('default', 'fake', 'model', routes={
            'reasoning': [{'provider': 'fake', 'model': 'model'}],
            'embedding': [{'provider': 'fake', 'model': 'model'}],
            'vision': [{'provider': 'fake', 'model': 'model'}],
        })
        self.config = ResolvedConfig(None, self.root, 'default', {'default': self.profile})
        self.caps = CapabilityRuntime(self.config)
        self.adapter = Mock(version='test', api_base=None)
        self.adapter.capabilities.return_value = {'reasoning', 'embedding', 'vision'}
        self.caps.adapters[('default', 'fake')] = self.adapter
        self.addCleanup(self.caps.close)

    def test_embedding_and_input_contracts_reject_invalid_shapes(self):
        for vectors in ([], [[]], [[1.0], [1.0, 2.0]], [[math.inf]], [[True]]):
            with self.subTest(vectors=vectors), self.assertRaises(CapabilityError):
                Embeddings(vectors, 'space')
        specs = Registry().specs
        cases = [
            ('video', {'mode': 'understand', 'prompt': 'describe'}, 'requires assets'),
            ('reranking', {'query': 1, 'documents': ['a']}, 'query must be text'),
            ('embedding', {'inputs': ['a'], 'dimensions': 0}, 'positive integer'),
            ('vision', {'prompt': 'describe', 'assets': []}, 'nonempty list'),
            ('vision', {'prompt': 'describe', 'assets': [object()]}, 'paths or Asset'),
            ('embedding', {'inputs': [object()]}, 'embedding inputs'),
            ('indexing', {'embeddings': 'wrong', 'documents': ['a']}, 'must be Embeddings'),
            ('semantic_search', {'index': 'wrong', 'query': Embeddings([[1]], 'space')},
             'VectorIndex'),
            ('reasoning', {}, 'missing required'),
            ('reasoning', {'prompt': 'hi', 'extra': True}, 'unknown input'),
        ]
        for name, params, message in cases:
            with self.subTest(name=name, params=params), self.assertRaisesRegex(CapabilityError, message):
                specs[name].validate_inputs(params)
        specs['reasoning'].validate_inputs({'prompt': {'$binding': 'prompt'}}, references=True)

    def test_registry_plugins_and_cache_encoding_are_validated(self):
        plugin = CapabilitySpec('custom', 'Custom', ('prompt',))
        entry = Mock()
        entry.load.return_value = lambda: plugin
        with patch('aiython.capabilities.importlib.metadata.entry_points', return_value=[entry]):
            registry = Registry()
        self.assertIn('custom', registry.specs)
        with self.assertRaisesRegex(ConfigError, 'unique CapabilitySpec'):
            registry.register(plugin)
        with self.assertRaisesRegex(ConfigError, 'unique CapabilitySpec'):
            Registry().register(object())
        with self.assertRaises(TypeError):
            encode({'$reserved': 1})
        with self.assertRaises(TypeError):
            encode(object())
        self.assertEqual(decode([{'public': [1, None]}]), [{'public': [1, None]}])
        self.assertEqual(decode(encode(Embeddings([[1.0, 2.0]], 'space'))).space, 'space')
        path = self.root / 'asset.txt'
        path.write_text('original')
        saved = encode(Document(path))
        self.assertEqual(decode(saved).path, path)
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            decode(saved)

    def test_store_invalid_cache_and_fork_cleanup(self):
        store = Store(self.root)
        self.addCleanup(store.close)
        self.assertEqual(store.jobs(), [])
        store.put('uncacheable', object())
        self.assertEqual(store.cached('uncacheable'), (False, None))
        store.put('valid', {'value': 42})
        self.assertEqual(store.cached('valid'), (True, {'value': 42}))
        with store.connect() as db:
            db.execute('UPDATE cache SET value=? WHERE key=?', ('{bad json', 'valid'))
        self.assertEqual(store.cached('valid'), (False, None))
        store.job('job-1', {'status': 'pending'})
        self.assertEqual(store.jobs(), [{'id': 'job-1', 'status': 'pending'}])
        self.assertIsNone(store.job('missing'))
        self.assertTrue(store._connections)
        with patch('aiython.capabilities.os.getpid', return_value=store._pid + 1):
            store._after_fork()
        self.assertEqual(store._connections, {})

    def test_store_closes_failed_schema_initialization(self):
        store = Store(self.root)
        self.addCleanup(store.close)
        connection = Mock()
        connection.execute.side_effect = sqlite3.OperationalError('schema failure')
        with patch('aiython.capabilities.sqlite3.connect', return_value=connection):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'schema failure'):
                with store.connect():
                    pass
        connection.close.assert_called_once()

    def test_local_documents_and_index_validation(self):
        local = LocalProvider()
        document = self.root / 'policy.txt'
        document.write_text('\n'.join(f'line {n}' for n in range(41)))
        request = lambda capability, params: CapabilityRequest(capability, 'local', params)
        with self.assertRaisesRegex(InvocationError, 'support extraction'):
            local.invoke(request('document_understanding', {'assets': [Document(document)],
                                                           'prompt': 'answer'}), self.caps)
        pdf = self.root / 'policy.pdf'
        pdf.write_text('text')
        with self.assertRaisesRegex(InvocationError, 'accepts .txt/.md'):
            local.invoke(request('document_understanding', {'assets': [Document(pdf)]}), self.caps)
        chunks = local.invoke(request('document_understanding', {'assets': [Document(document)]}),
                              self.caps).value
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1]['source'].endswith(':41'))
        vectors = Embeddings([[1.0, 0.0]], 'space')
        for params, message in [
            ({'embeddings': vectors, 'documents': []}, 'One embedding per document'),
            ({'embeddings': vectors, 'documents': [42]}, 'Documents must be text'),
            ({'embeddings': vectors, 'documents': [{'text': 'x', 'source': 42}]},
             'source must be a string'),
        ]:
            with self.subTest(params=params), self.assertRaisesRegex(CapabilityError, message):
                local.invoke(request('indexing', params), self.caps)
        index = local.invoke(request('indexing', {'embeddings': vectors, 'documents': ['x']}),
                             self.caps).value
        with self.assertRaisesRegex(CapabilityError, 'one query embedding'):
            local.invoke(request('semantic_search', {'index': index,
                        'query': Embeddings([[1.0, 0.0], [0.0, 1.0]], 'space')}), self.caps)
        other = VectorIndex(self.root / 'other.sqlite', index.name, index.space, index.dimensions)
        with self.assertRaisesRegex(CapabilityError, 'different project'):
            local.invoke(request('semantic_search', {'index': other,
                        'query': Embeddings([[1.0, 0.0]], 'space')}), self.caps)
        missing = VectorIndex(index.path, 'missing', index.space, index.dimensions)
        with self.assertRaisesRegex(CapabilityError, 'Index missing'):
            local.invoke(request('semantic_search', {'index': missing,
                        'query': Embeddings([[1.0, 0.0]], 'space')}), self.caps)
        restored = VectorIndex(index.path, index.name, index.space, index.dimensions)
        hits = local.invoke(request('semantic_search', {'index': restored,
                            'query': Embeddings([[0.0, 0.0]], 'space')}), self.caps).value
        self.assertEqual(hits[0]['score'], 0.0)
        self.assertEqual(hits[0]['object'], {'text': 'x', 'source': ''})

    def test_runtime_route_permissions_and_asset_conversion(self):
        invalid = ProfileConfig('invalid', 'fake', 'model', routes={'unknown': [{'provider': 'x'}]})
        with self.assertRaisesRegex(ConfigError, 'unregistered capability'):
            CapabilityRuntime(ResolvedConfig(None, self.root, profiles={'invalid': invalid}))
        self.assertEqual(self.caps.routes(ProfileConfig('legacy', 'fake', 'model'), 'reasoning')[0]['provider'],
                         '$legacy')
        with self.assertRaisesRegex(CapabilityError, 'Unknown job operation ID'):
            self.caps.resume_job(self.profile, 'unknown')
        self.assertIsNone(self.caps.suggest_missing_route(self.profile, 'reasoning'))
        with self.assertRaisesRegex(CapabilityError, 'Asset must be a path'):
            self.caps.asset(42)
        relative = Image(Path('photo.png'))
        self.assertEqual(self.caps.asset(relative).path, self.root / 'photo.png')
        cyclic = []
        cyclic.append(cyclic)
        self.caps.assets({'context': cyclic}, self.profile)
        with self.assertRaisesRegex(CapabilityError, 'nonempty list'):
            self.caps.assets({'assets': []}, self.profile)

    def test_runtime_trace_and_fork_reset(self):
        traced = CapabilityRuntime(self.config, trace=True)
        self.addCleanup(traced.close)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            traced.event(capability='embedding', status='ok')
        self.assertIn('aiython plan:', output.getvalue())
        traced._pool = Mock()
        with patch('aiython.capabilities.os.getpid', return_value=traced._pid + 1):
            traced._after_fork()
        self.assertIsNone(traced._pool)

    def test_runtime_rejects_invalid_invocation_inputs_and_adapter_results(self):
        with self.assertRaisesRegex(ConfigError, 'Unregistered capability'):
            self.caps.invoke(self.profile, 'unknown', {})
        from dataclasses import replace
        restricted = replace(self.profile, permissions=('network',))
        with self.assertRaisesRegex(CapabilityError, 'permission denied'):
            self.caps.invoke(restricted, 'video', {'mode': 'generate', 'prompt': 'clip'})
        text = self.root / 'notes.txt'
        text.write_text('plain text')
        with self.assertRaisesRegex(CapabilityError, 'requires image assets'):
            self.caps.invoke(self.profile, 'vision', {'prompt': 'describe', 'assets': [text]})
        image = self.root / 'photo.png'
        image.write_bytes(b'image')
        with self.assertRaisesRegex(CapabilityError, 'requires audio or video'):
            self.caps.invoke(self.profile, 'speech_to_text', {'assets': [image]})
        with self.assertRaisesRegex(CapabilityError, 'video mode'):
            self.caps.invoke(self.profile, 'video', {'mode': 'invalid', 'prompt': 'clip'})
        self.adapter.capabilities.return_value = set()
        with self.assertRaisesRegex(ConfigError, 'does not support vision'):
            self.caps.invoke(self.profile, 'vision', {'prompt': 'describe', 'assets': [image]})
        self.adapter.capabilities.return_value = {'reasoning', 'embedding', 'vision'}
        self.adapter.invoke.return_value = 'not a CapabilityResult'
        with self.assertRaisesRegex(CapabilityError, 'must return CapabilityResult'):
            self.caps.invoke(self.profile, 'reasoning', {'prompt': 'hi'})
        with self.assertRaisesRegex(CapabilityError, 'invalid adapter result type'):
            self.caps.validate_result('video', {'mode': 'generate'}, 'not a Video')
        with self.assertRaisesRegex(CapabilityError, 'invalid adapter result type'):
            self.caps.validate_result('reasoning', {'prompt': 'hi'}, 42)
        with self.assertRaisesRegex(CapabilityError, 'result count'):
            self.caps.validate_result('embedding', {'inputs': ['one', 'two']},
                                      Embeddings([[1.0]], 'space'))
        class LooseSpec(CapabilitySpec):
            def validate_inputs(self, params, *, references=False):
                pass
        self.caps.registry.specs['custom_limit'] = LooseSpec(
            'custom_limit', 'Plugin capability with its own input parser', (), ('limit',))
        with self.assertRaisesRegex(CapabilityError, 'limit must be a positive integer'):
            self.caps.invoke(self.profile, 'custom_limit', {'limit': 0})

    def test_runtime_cache_disabled_by_uncacheable_context_or_permissions(self):
        self.adapter.invoke.return_value = CapabilityResult('done')
        self.assertEqual(self.caps.invoke(self.profile, 'reasoning',
                         {'prompt': 'hi', 'context': {'live': object()}}, cache=True), 'done')
        from dataclasses import replace
        restricted = replace(self.profile, permissions=('network',))
        stats = InvocationStats('test.py', 1, 'syntax')
        token = CURRENT_STATS.set(stats)
        try:
            self.assertEqual(self.caps.invoke(restricted, 'reasoning', {'prompt': 'hi'}, cache=True),
                             'done')
        finally:
            CURRENT_STATS.reset(token)
        self.assertGreaterEqual(stats.cache_seconds, 0)

    def test_runtime_builds_configured_adapter_on_demand(self):
        route = {'provider': 'external', 'model': 'model'}
        factory_result = Mock(version='test')
        with patch('aiython.capability_providers.make_adapter', return_value=factory_result) as factory:
            self.assertIs(self.caps.adapter(route, self.profile), factory_result)
            self.assertIs(self.caps.adapter(route, self.profile), factory_result)
        factory.assert_called_once()

    @staticmethod
    def bridge(bindings=None):
        bindings = bindings or {}
        return SimpleNamespace(frame=SimpleNamespace(f_globals={}),
                               namespace=lambda: bindings,
                               dereference=lambda handle: bindings[handle])

    def test_plan_references_reject_malformed_shapes_before_execution(self):
        valid = {'id': 's', 'capability': 'reasoning', 'params': {'prompt': 'hello'}}
        cases = [
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$ref': 'other', 'extra': 1}}}], {'$ref': 's'}, 'Invalid step reference'),
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$ref': 'other', 'path': 'bad'}}}], {'$ref': 's'}, 'Reference path'),
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$asset': 42}}}], {'$ref': 's'}, 'Asset reference'),
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$handle': 'h', 'extra': 1}}}], {'$ref': 's'}, 'Invalid runtime reference'),
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$binding': 'not valid'}}}], {'$ref': 's'}, 'must be identifiers'),
            ([{'id': 's', 'capability': 'reasoning', 'params': {
                'prompt': {'$binding': 'missing'}}}], {'$ref': 's'}, 'does not exist'),
            ([None], {'$ref': 's'}, 'Invalid plan step'),
            ([valid, valid], {'$ref': 's'}, 'nonempty and unique'),
            ([{**valid, 'cache': 'yes'}], {'$ref': 's'}, 'cache must be boolean'),
            ([valid], {'$ref': 'other'}, 'unknown step'),
        ]
        for steps, output, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(CapabilityError, message):
                self.caps.run_plan(self.profile, steps, output, self.bridge())
        self.adapter.invoke.assert_not_called()

    def test_plan_directives_and_video_mode_validate_before_execution(self):
        step = {'id': 's', 'capability': 'reasoning', 'params': {'prompt': 'hi'}}
        with self.assertRaisesRegex(CapabilityError, 'Plan must contain'):
            self.caps.run_plan(self.profile, [], {'$ref': 's'}, self.bridge())
        with self.assertRaisesRegex(CapabilityError, 'provider directive requires'):
            self.caps.run_plan(self.profile, [step], {'$ref': 'missing'}, self.bridge(),
                               provider='fake')
        with self.assertRaisesRegex(CapabilityError, 'must include the directive capability'):
            self.caps.run_plan(self.profile, [step], {'$ref': 's'}, self.bridge(),
                               capability='vision')
        video = {'id': 'v', 'capability': 'video',
                 'params': {'mode': 'bad', 'prompt': 'make a clip'}}
        with self.assertRaisesRegex(CapabilityError, 'literal understand or generate'):
            self.caps.run_plan(self.profile, [video], {'$ref': 'v'}, self.bridge())
        from dataclasses import replace
        restricted = replace(self.profile, permissions=('network',))
        video['params']['mode'] = 'generate'
        with self.assertRaisesRegex(CapabilityError, 'permission denied'):
            self.caps.run_plan(restricted, [video], {'$ref': 'v'}, self.bridge())
        self.adapter.capabilities.return_value = set()
        with self.assertRaisesRegex(ConfigError, 'does not support reasoning'):
            self.caps.run_plan(self.profile, [step], {'$ref': 's'}, self.bridge())
        self.adapter.capabilities.return_value = {'reasoning', 'embedding', 'vision'}

    def test_plan_runtime_handles_paths_and_completed_step_changes(self):
        step = {'id': 's', 'capability': 'reasoning',
                'params': {'prompt': {'$handle': 'prompt'}}}
        bridge = self.bridge({'prompt': 'hello'})
        self.adapter.invoke.return_value = CapabilityResult('done')
        self.assertEqual(self.caps.run_plan(self.profile, [step], {'$ref': 's'}, bridge), 'done')
        self.assertEqual(self.adapter.invoke.call_args.args[0].params['prompt'], 'hello')
        changed = {'id': 's', 'capability': 'reasoning', 'params': {'prompt': 'different'}}
        with self.assertRaisesRegex(CapabilityError, 'cannot be reused'):
            self.caps.run_plan(self.profile, [changed], {'$ref': 's'}, bridge)
        primitive = {'id': 'a', 'capability': 'reasoning', 'params': {'prompt': 'hello'}}
        with self.assertRaisesRegex(CapabilityError, 'can only traverse dict/list'):
            self.caps.run_plan(self.profile, [primitive], {'$ref': 'a', 'path': ['bad']},
                               self.bridge())

    def test_parallel_plan_collects_worker_failure(self):
        steps = [
            {'id': 'one', 'capability': 'reasoning', 'params': {'prompt': 'fail'}},
            {'id': 'two', 'capability': 'reasoning', 'params': {'prompt': 'succeed'}},
        ]
        def invoke(request, _context):
            if request.params['prompt'] == 'fail':
                raise InvocationError('provider failed')
            return CapabilityResult('done')
        self.adapter.invoke.side_effect = invoke
        with self.assertRaisesRegex(InvocationError, 'provider failed'):
            self.caps.run_plan(self.profile, steps, {'$ref': 'two'}, self.bridge())
        self.adapter.invoke.side_effect = lambda request, _context: CapabilityResult(
            request.params['prompt'])
        self.assertEqual(self.caps.run_plan(self.profile, steps, {'$ref': 'two'}, self.bridge()),
                         'succeed')

    def test_parallel_plan_propagates_unexpected_worker_failure(self):
        steps = [
            {'id': 'one', 'capability': 'reasoning', 'params': {'prompt': 'one'}},
            {'id': 'two', 'capability': 'reasoning', 'params': {'prompt': 'two'}},
        ]
        with patch.object(self.caps, 'invoke', side_effect=RuntimeError('worker failed')):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                self.caps.run_plan(self.profile, steps, {'$ref': 'two'}, self.bridge())
