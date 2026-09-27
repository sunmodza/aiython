"""Regression contracts for optimizations; no timing thresholds or live APIs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import gc
import io
import json
import multiprocessing
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch

from aiython.capabilities import Store, CapabilityResult, Embeddings
from aiython.cli import run_script
from aiython.models import ProfileConfig, ResolvedConfig
from aiython.providers import LiteLLMProvider
from aiython.runtime import Runtime
from aiython.type_constraints import ContractCache, TypeViolation, compile_contract
from aiython.stats import CURRENT_STATS, InvocationStats


class OverheadTests(unittest.TestCase):
    def test_primitive_containers_keep_strict_types_and_error_paths(self):
        for annotation, good, bad, path in [
            ('list[int]', [1, 2], [1, True], 'value[1]'),
            ('tuple[str, ...]', ('a', 'b'), ('a', 1), 'value[1]'),
            ('list[float]', [1.0], [1], 'value[0]'),
            ('list[bytes]', [b'a'], ['a'], 'value[0]'),
        ]:
            with self.subTest(annotation=annotation):
                contract = compile_contract(annotation, {})
                contract.validate(good)
                with self.assertRaises(TypeViolation) as error:
                    contract.validate(bad)
                self.assertIn(path, str(error.exception))

    def test_cached_types_follow_rebindings_and_class_annotation_changes(self):
        cache = ContractCache()
        first = cache.compile('list[int]', {})
        first.validate([1])
        cache.compile('list[int]', {'int': str}).validate(['now a string'])
        with self.assertRaises(TypeViolation):
            cache.compile('list[int]', {'int': str}).validate([1])
        Item = type('Item', (), {'__annotations__': {'value': int}})
        obj = Item()
        obj.value = 1
        cache.compile('Item', {'Item': Item}).validate(obj)
        Item.__annotations__['value'] = str
        with self.assertRaises(TypeViolation):
            cache.compile('Item', {'Item': Item}).validate(obj)
        # A string forward reference has dependencies beyond its outer AST.
        cache.compile('list["Alias"]', {'Alias': int}).validate([1])
        with self.assertRaises(TypeViolation):
            cache.compile('list["Alias"]', {'Alias': str}).validate([1])

    def test_preparation_cache_preserves_fresh_state_and_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'main.py'
            config = ResolvedConfig(None, path.parent)
            source = 'items: list[int] = []\nitems.append(1)\nanswer = items\n'
            first, second = Runtime(config, stats=True), Runtime(config, stats=True)
            code_a = first.compile_source(source, str(path), entry=True)
            code_b = second.compile_source(source, str(path), entry=True)
            self.assertIsNot(code_a, code_b)
            self.assertIs(code_a, first.compile_source(source, str(path), entry=True))
            self.assertEqual(second.stats.preparation_cache_hits, 1)
            self.assertIsNot(first.units[str(path)], second.units[str(path)])
            self.assertEqual(set(first.checkpoints), set(second.checkpoints))
            left, right = {}, {}
            exec(code_a, left)
            exec(code_b, right)
            self.assertFalse(any(name.startswith('__aiython_runtime') for name in left | right))
            self.assertIsNot(left['answer'], right['answer'])
            changed = Runtime(config)
            namespace = {}
            exec(changed.compile_source(source.replace('append(1)', 'append(2)'), str(path), entry=True), namespace)
            self.assertEqual(namespace['answer'], [2])
            self.assertEqual(left['answer'], [1])

    def test_cached_program_uses_current_agent_and_directives(self):
        class Agent:
            def __init__(self, value): self.value = value
            def execute(self, request, runtime):
                self.profile = request.profile.name
                return self.value
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'main.py'
            path.write_text('# aiython: profile="selected"\nanswer: int = choose a value\n')
            config = ResolvedConfig(None, path.parent, 'default', {
                name: ProfileConfig(name, 'fake', name) for name in ('default', 'selected')})
            for value in (1, 2):
                agent = Agent(value)
                result = run_script(path, config=config, agent_factory=lambda _: agent)
                self.assertEqual(result['answer'], value)
                self.assertEqual(agent.profile, 'selected')

    def test_warm_program_still_catches_foreign_alias_mutations(self):
        # Foreign helpers are not AST-instrumented. Every execution must inspect
        # the actual contents even if length, identity and source are unchanged.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'main.py'
            config = ResolvedConfig(None, path.parent)
            source = 'items: list[int] = [1, 2]\nalias = items\nmutate(alias)\nanswer = items\n'
            for _ in range(2):
                runtime = Runtime(config)
                namespace = {'mutate': lambda values: values.__setitem__(0, True)}
                with self.assertRaises(TypeViolation):
                    exec(runtime.compile_source(source, str(path), entry=True), namespace)

    def test_cached_tool_code_uses_live_values_and_rechecks_types(self):
        class Agent:
            def execute(self, request, bridge):
                bridge.exec('items.append(bridge_value)')
                return bridge.eval(' \t len(items) ')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'main.py'
            profile = ProfileConfig('default', 'fake', 'fake')
            config = ResolvedConfig(None, path.parent, 'default', {'default': profile})
            path.write_text('items: list[int] = []\nfor bridge_value in [1, 2]:\n    size: int = append the current value\n')
            result = run_script(path, config=config, agent_factory=lambda _: Agent())
            self.assertEqual(result['items'], [1, 2])
            self.assertEqual(result['size'], 2)
            path.write_text(path.read_text().replace('[1, 2]', '[1, True]'))
            with self.assertRaises(TypeViolation):
                run_script(path, config=config, agent_factory=lambda _: Agent())

    def test_store_reuses_connections_and_recovers_after_close(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder))
            with store.connect() as first:
                pass
            with store.connect() as second:
                self.assertIs(first, second)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: store.put(str(i), i), range(20)))
            self.assertEqual([store.cached(str(i)) for i in range(20)], [(True, i) for i in range(20)])
            store.close()
            self.assertEqual(store.cached('3'), (True, 3))
            store.close()

    def test_store_rolls_back_failed_transaction_and_detects_database_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder))
            with self.assertRaises(ValueError), store.connect() as db:
                db.execute('INSERT INTO cache VALUES (?, ?, ?)', ('failed', 0, '1'))
                raise ValueError('abort')
            self.assertEqual(store.cached('failed'), (False, None))
            store.put('old', 1)
            (store.root / 'runtime-v3.sqlite').unlink()
            self.assertEqual(store.cached('old'), (False, None))
            store.put('new', 2)
            self.assertEqual(store.cached('new'), (True, 2))
            store.close()

    def test_store_finalizer_closes_persistent_handle(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder))
            with store.connect() as db:
                pass
            del store
            gc.collect()
            with self.assertRaises(sqlite3.ProgrammingError):
                db.execute('SELECT 1')

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork unavailable')
    def test_store_reopens_in_child_without_inheriting_locked_mutex(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder))
            store.put('parent', 1)
            context = multiprocessing.get_context('fork')
            reader, writer = context.Pipe(duplex=False)
            def child():
                store.put('child', 2)
                writer.send(store.cached('parent'))
                store.close()
            # Simulate a lock held by a different thread at fork time.
            from threading import Event, Thread
            acquired, release = Event(), Event()
            def hold():
                with store._lock:
                    acquired.set()
                    release.wait(10)
            holder = Thread(target=hold)
            holder.start()
            self.assertTrue(acquired.wait(5))
            process = context.Process(target=child)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*multi-threaded.*')
                    process.start()
                self.assertTrue(reader.poll(5))
                self.assertEqual(reader.recv(), (True, 1))
            finally:
                release.set()
                holder.join(5)
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
                reader.close()
                writer.close()
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(store.cached('child'), (True, 2))
            store.close()

    def test_cache_hit_timing_includes_lookup_and_skips_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = ProfileConfig('default', 'fake', 'fake', routes={
                'embedding': [{'provider': 'fake', 'model': 'fake'}]})
            runtime = Runtime(ResolvedConfig(None, Path(folder), 'default', {'default': profile}))
            adapter = Mock(version='1', api_base=None)
            adapter.capabilities.return_value = {'embedding'}
            adapter.invoke.return_value = CapabilityResult(Embeddings([[1.]], 'space'))
            runtime.capabilities.adapters[('default', 'fake')] = adapter
            record = InvocationStats('test.py', 1, 'syntax')
            token = CURRENT_STATS.set(record)
            try:
                for _ in range(2):
                    runtime.capabilities.invoke(profile, 'embedding', {'inputs': ['one']})
                self.assertEqual(adapter.invoke.call_count, 1)
                event = record.capability_events[-1]
                self.assertEqual(event['cache'], 'hit')
                self.assertGreater(event['seconds'], 0)
                self.assertGreater(record.cache_seconds, 0)
                self.assertGreater(record.capability_provider_seconds, 0)
                self.assertGreater(record.capability_seconds, record.capability_provider_seconds)
            finally:
                CURRENT_STATS.reset(token)
                runtime.capabilities.close()

    def test_reusable_plan_workers_handle_nested_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = ProfileConfig('default', 'fake', 'fake', routes={
                'embedding': [{'provider': 'fake', 'model': 'fake'}]})
            runtime = Runtime(ResolvedConfig(None, Path(folder), 'default', {'default': profile}))
            caps = runtime.capabilities
            def steps(prefix, count):
                return [{'id': f'{prefix}{i}', 'capability': 'embedding',
                         'params': {'inputs': [f'{prefix}{i}']}, 'cache': False} for i in range(count)]
            class Adapter:
                version = '1'
                def capabilities(self): return {'embedding'}
                def invoke(self, request, context):
                    if request.params['inputs'][0].startswith('outer'):
                        context.run_plan(profile, steps('inner', 2), {'$ref': 'inner0'}, SimpleNamespace())
                    return CapabilityResult(Embeddings([[1.]], 'space'))
            caps.adapters[('default', 'fake')] = Adapter()
            try:
                for _ in range(2):
                    result = caps.run_plan(profile, steps('outer', 4), {'$ref': 'outer0'}, SimpleNamespace())
                    self.assertEqual(result.vectors, [[1.]])
            finally:
                caps.close()

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork unavailable')
    def test_plan_workers_reopen_after_fork(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = ProfileConfig('default', 'fake', 'fake', routes={
                'embedding': [{'provider': 'fake', 'model': 'fake'}]})
            runtime = Runtime(ResolvedConfig(None, Path(folder), 'default', {'default': profile}))
            caps = runtime.capabilities
            adapter = Mock(version='1', api_base=None)
            adapter.capabilities.return_value = {'embedding'}
            adapter.invoke.return_value = CapabilityResult(Embeddings([[1.]], 'space'))
            caps.adapters[('default', 'fake')] = adapter
            steps = [{'id': str(i), 'capability': 'embedding',
                      'params': {'inputs': [str(i)]}, 'cache': False} for i in range(2)]
            def run():
                return caps.run_plan(profile, steps, {'$ref': '0'}, SimpleNamespace()).vectors
            self.assertEqual(run(), [[1.]])
            context = multiprocessing.get_context('fork')
            reader, writer = context.Pipe(duplex=False)
            def child():
                writer.send(run())
                caps.close()
            process = context.Process(target=child)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*multi-threaded.*')
                    process.start()
                self.assertTrue(reader.poll(5), 'child reused an executor with no live workers')
                self.assertEqual(reader.recv(), [[1.]])
            finally:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
                reader.close()
                writer.close()
                caps.close()
            self.assertEqual(process.exitcode, 0)

    def test_disabled_stats_skip_request_serialization_sync_and_async(self):
        profile = ProfileConfig('default', 'fake', 'fake', routes={
            'reasoning': [{'provider': 'route', 'model': 'openai/test'}]})
        config = ResolvedConfig(None, Path.cwd(), 'default', {'default': profile})
        response = {'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]}
        async def complete(**kwargs): return response
        sdk = Mock(completion=Mock(return_value=response), acompletion=complete)
        provider = LiteLLMProvider(config, profile)
        with patch('aiython.providers.sdk', return_value=sdk), \
                patch('aiython.providers.json.dumps', side_effect=AssertionError('unneeded serialization')):
            self.assertEqual(provider.complete([], [])['content'], 'ok')
            self.assertEqual(asyncio.run(provider.acomplete([], []))['content'], 'ok')

    def test_stats_measure_whole_run_and_warm_cache_without_changing_stdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'main.py'
            path.write_text('print("done")\n')
            for expected_hits in (0, 1):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    run_script(path, stats=True)
                self.assertEqual(out.getvalue(), 'done\n')
                summary = json.loads(err.getvalue().split('aiython run stats: ')[1])
                self.assertEqual(summary['preparation_cache_hits'], expected_hits)
                self.assertGreaterEqual(summary['total_seconds'], summary['prepare_seconds'])
