"""Offline runtime diagnostics, independent of network/model latency.

Run: uv run python benchmarks/runtime_overhead.py --output result.json
Use --source-root /path/to/checkout/src to compare another source snapshot using
the same interpreter, dependencies, fixtures and timing procedure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1] / 'src')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('use at least three repeats')
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    from aiython.agent import ToolAgent
    from aiython.capabilities import CapabilityResult, Embeddings, Store
    from aiython.frontend import RUNTIME_NAME, parse
    from aiython.models import ProfileConfig, ResolvedConfig
    from aiython.runtime import Runtime

    def measure(fn):
        fn()
        samples = []
        for _ in range(args.repeats):
            start = perf_counter()
            fn()
            samples.append((perf_counter() - start) * 1000)
        return {'median_ms': statistics.median(samples), 'samples_ms': samples}

    class Provider:
        def complete(self, messages, tools):
            return {'role': 'assistant', 'tool_calls': [{
                'id': 'answer', 'type': 'function', 'function': {
                    'name': 'finish', 'arguments': '{"outcome":{"kind":"literal","value":1}}'}}]}

    cases = [
        ('untyped_scalar_10000', 'answer = 0\nfor i in range(10000):\n    answer += 1\n', 10000),
        ('typed_scalar_10000', 'answer: int = 0\nfor i in range(10000):\n    answer += 1\n', 10000),
        ('untyped_list_800', 'items = []\nfor i in range(800):\n    items.append(i)\nanswer = len(items)\n', 800),
        ('typed_list_400', 'items: list[int] = []\nfor i in range(400):\n    items.append(i)\nanswer = len(items)\n', 400),
        ('typed_list_800', 'items: list[int] = []\nfor i in range(800):\n    items.append(i)\nanswer = len(items)\n', 800),
        ('ai_invocations_100', 'answer = 0\nfor i in range(100):\n    value: int = choose one value\n    answer += value\n', 100),
    ]
    results = {'python': sys.version, 'platform': platform.platform(), 'repeats': args.repeats,
               'started_at': datetime.now(timezone.utc).isoformat(),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'dependencies': {name: importlib.metadata.version(name) for name in (
                   'typing-extensions', 'python-dotenv', 'pydantic', 'litellm')},
               'network': False, 'timing': 'milliseconds, one untimed warm-up, median of repetitions',
               'cases': {}}
    digest = hashlib.sha256()
    for path in sorted((source_root / 'aiython').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    results['source_sha256'] = digest.hexdigest()
    with tempfile.TemporaryDirectory(prefix='aiython-benchmark-') as directory:
        root = Path(directory)
        profile = ProfileConfig('default', 'fake', 'fake')
        config = ResolvedConfig(None, root, 'default', {'default': profile})
        for name, source, expected in cases:
            runtime = Runtime(config, agent_factory=lambda _: ToolAgent(Provider()))
            code = runtime.prepare(parse(source, str(root / (name + '.py'))), entry=True)
            def run():
                namespace = {RUNTIME_NAME: runtime, '__name__': '__main__'}
                exec(code, namespace)
                assert namespace['answer'] == expected
            results['cases'][name] = {**measure(run), 'source': source,
                                      'scope': 'execution only; preparation/imports excluded'}
            if hasattr(runtime.capabilities, 'close'):
                runtime.capabilities.close()

        source = ''.join(f'value_{i}: int = choose one value\n' for i in range(50))
        filename = str(root / 'prepare.py')
        def prepare():
            runtime = Runtime(config)
            if hasattr(runtime, 'compile_source'):
                runtime.compile_source(source, filename, entry=True)
            else:
                runtime.prepare(parse(source, filename), entry=True)
        results['cases']['prepare_repeated_source_50'] = {**measure(prepare),
            'scope': 'fresh Runtime, repeated identical source in one process'}
        preparations = 0
        def prepare_cold():
            nonlocal preparations
            preparations += 1
            runtime = Runtime(config)
            name = str(root / f'cold-{preparations}.py')
            if hasattr(runtime, 'compile_source'):
                runtime.compile_source(source, name, entry=True)
            else:
                runtime.prepare(parse(source, name), entry=True)
        results['cases']['prepare_cold_source_50'] = {**measure(prepare_cold),
            'scope': 'fresh Runtime and filename; preparation cache miss; imports warm'}

        store = Store(root)
        store.put('entry', {'value': 42})
        def read_cache():
            for _ in range(100):
                assert store.cached('entry') == (True, {'value': 42})
        results['cases']['sqlite_hits_100'] = {**measure(read_cache), 'scope': '100 existing cache lookups'}
        if hasattr(store, 'close'):
            store.close()

        routed = ProfileConfig('default', 'fake', 'fake', routes={
            'embedding': [{'provider': 'fake', 'model': 'fake'}]})
        runtime = Runtime(ResolvedConfig(None, root, 'default', {'default': routed}))
        class Adapter:
            version = '1'
            def capabilities(self): return {'embedding'}
            def invoke(self, request, context):
                return CapabilityResult(Embeddings([[1.]], 'space'))
        runtime.capabilities.adapters[('default', 'fake')] = Adapter()
        steps = [{'id': str(i), 'capability': 'embedding',
                  'params': {'inputs': [str(i)]}, 'cache': False} for i in range(2)]
        def run_plans():
            for _ in range(20):
                value = runtime.capabilities.run_plan(routed, steps, {'$ref': '0'}, SimpleNamespace())
                assert value.vectors == [[1.]]
        results['cases']['capability_plans_20'] = {**measure(run_plans),
            'scope': '20 plans with two independent immediate fake adapter calls each; cache disabled'}
        if hasattr(runtime.capabilities, 'close'):
            runtime.capabilities.close()

        script = root / 'plain.py'
        script.write_text('answer = 1\n')
        env = {**os.environ, 'PYTHONPATH': str(source_root)}
        def spawn(command):
            subprocess.run([sys.executable, *command], check=True, capture_output=True, cwd=root, env=env)
        results['cases']['startup_plain_python'] = {**measure(lambda: spawn([str(script)])),
            'scope': 'new Python process; filesystem caches may be warm'}
        results['cases']['startup_aiython'] = {**measure(lambda: spawn(['-m', 'aiython', str(script)])),
            'scope': 'new Aiython process, no project configuration; filesystem caches may be warm'}

    assert 'litellm' not in sys.modules
    content = json.dumps(results, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
