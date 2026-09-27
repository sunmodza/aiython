"""Check that Aiython compiles the current interpreter's standard library.

Run: uv run python benchmarks/stdlib_syntax.py
This checks syntax transformation only; it does not execute the modules.
"""

import argparse
import gc
from pathlib import Path
import sysconfig
import tokenize

from aiython.models import ProfileConfig, ResolvedConfig
from aiython.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdlib', type=Path, default=Path(sysconfig.get_path('stdlib')))
    parser.add_argument('--batch-size', type=int, default=40)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    root = args.stdlib.resolve()
    paths = sorted(path for path in root.rglob('*.py')
                   if not {'site-packages', 'test', 'tests'} & set(path.parts))
    profile = ProfileConfig('offline', 'fake', 'offline')
    config = ResolvedConfig(None, root, 'offline', {'offline': profile})
    failures = []
    runtime = None
    for index, path in enumerate(paths):
        if index % args.batch_size == 0:
            if runtime is not None:
                runtime.capabilities.close()
                del runtime
                gc.collect()
            runtime = Runtime(config)
        try:
            with tokenize.open(path) as file:
                source = file.read()
            compile(source, str(path), 'exec', dont_inherit=True)
            runtime.compile_source(source, str(path))
        except Exception as error:
            failures.append((path.relative_to(root), error))
        if (index + 1) % 100 == 0:
            print(f'Checked {index + 1}/{len(paths)}; failures: {len(failures)}', flush=True)
    if runtime is not None:
        runtime.capabilities.close()
    print(f'Checked {len(paths)} standard library files; failures: {len(failures)}')
    for path, error in failures[:30]:
        print(f'{path}: {type(error).__name__}: {error}')
    if len(failures) > 30:
        print(f'... and {len(failures) - 30} more failures')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
