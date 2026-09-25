# Offline overhead evidence

These fixtures measure local Aiython work, not live model quality or latency.
Run from the repository root with the same Python and installed dependencies for
both source snapshots:

```sh
uv run --locked python benchmarks/runtime_overhead.py \
  --source-root /path/to/before/src --output before-1.json
uv run --locked python benchmarks/runtime_overhead.py --output after-1.json
uv run --locked python benchmarks/runtime_overhead.py --output after-2.json
uv run --locked python benchmarks/runtime_overhead.py \
  --source-root /path/to/before/src --output before-2.json
uv run --locked python benchmarks/compare_overhead.py \
  --before before-1.json before-2.json --after after-1.json after-2.json
```

Each run has one untimed warm-up and seven measured repetitions. Run A-B-B-A
sequentially while other tests and benchmarks are idle. The comparison validates
runner hashes, source identity within each variant, fixtures, timing scopes,
Python/platform and dependency versions before calculating medians. These checks
do not control CPU load, filesystem state or other environmental noise.

## Recorded local result, 2026-09-25

CPython 3.14.4 on Linux/WSL2. The baseline is a snapshot of the working source
before these optimizations, including the existing Aiython rename; it is **not**
the repository's HEAD. Each table entry is the median of 14 raw samples per
variant. The two variants use the same interpreter, dependencies and runner.
No credentials, paid API calls or live models were used.

| Case | Before (ms) | After (ms) | Time reduced | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Untyped scalar loop, 10,000 iterations | 17.431 | 17.139 | 1.7% | 1.02x |
| Typed scalar loop, 10,000 iterations | 26.290 | 20.385 | 22.5% | 1.29x |
| Untyped list append, 800 items | 1.243 | 1.158 | 6.8% | 1.07x |
| Typed list append, 400 items | 62.725 | 2.621 | 95.8% | 23.93x |
| Typed list append, 800 items | 249.643 | 8.829 | 96.5% | 28.27x |
| 100 AI invocations, immediate fake provider | 15.336 | 9.656 | 37.0% | 1.59x |
| Prepare 50 AI expressions, warm source cache | 36.842 | 2.175 | 94.1% | 16.94x |
| Prepare 50 AI expressions, cache miss | 36.859 | 32.208 | 12.6% | 1.14x |
| SQLite cache, 100 reads | 5.235 | 0.736 | 85.9% | 7.11x |
| 20 capability plans, two immediate fake calls each | 14.828 | 6.497 | 56.2% | 2.28x |
| Plain Python process startup, control | 8.506 | 8.477 | 0.3% | 1.00x |
| Aiython process startup, no project config | 91.870 | 54.707 | 40.5% | 1.68x |

Raw evidence: [before 1](results/overhead-before-1.json),
[after 1](results/overhead-after-1.json), [after 2](results/overhead-after-2.json),
[before 2](results/overhead-before-2.json). The
[machine-readable comparison](results/overhead-comparison.json) retains the
source hashes and exact, unrounded calculations. Recreate it with:

```sh
uv run --locked python benchmarks/compare_overhead.py \
  --before benchmarks/results/overhead-before-1.json benchmarks/results/overhead-before-2.json \
  --after benchmarks/results/overhead-after-1.json benchmarks/results/overhead-after-2.json \
  --output benchmarks/results/overhead-comparison.json
```

Execution-only rows exclude parsing and imports. Preparation rows include fresh
runtime construction. The warm-source cache is process-local; a fresh CLI process
does not reuse it. Startup includes a fresh interpreter, while filesystem caches
may be warm. Capability-plan results use fake adapters with caching disabled.
The plain-Python control measures environmental variation, not an optimization.

Time reduction is `100 * (before - after) / before`; speedup is `before / after`.
Small differences, especially untyped loops, are not strong evidence from this
sample size. Do not average these percentages into a claim about user task speed.
Growing typed containers still require full scans at type checkpoints, and
model/network wait time is absent from all these fixtures. The prompt now
encourages existing single-response and batched tools; its effect on real model
rounds has not been measured here.

## Correctness checks

The offline suite ran 252 tests on each supported Python version: 3.11.16
(12 skipped), 3.12.14 (9 skipped), 3.13.15 (9 skipped), and 3.14.4 (3 skipped).
All applicable tests passed. The skips follow existing version-dependent
runtime features. Added regression coverage checks type rebinding, mutable class
annotations, alias mutation, cached live-frame tools, source invalidation,
directive/profile isolation, database rollback/replacement, fork behavior and
nested worker plans. The wheel built and ran both configured and unconfigured
plain scripts from an isolated Python 3.11 environment.

For live task comparisons against agent baselines, keep acceptance tests, input
snapshots, model settings, tool permissions and budgets fixed; count failures as
well as successes. Follow the [evaluation guide](../docs/performance-evaluation.md)
and collect run wall time, human intervention, token usage and cost separately.
