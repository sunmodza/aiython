# Measuring Aiython changes

Aiython executes one suspended Python statement at a time. A model call is an
implementation step inside that invocation, not a new Python iteration. Measure
the program's resulting values and side effects as well as the model interaction.

## What to measure

Run `aiython --stats script.py` and retain the two prefixed JSON lines from stderr
alongside the program output. `aiython stats: [...]` contains invocation records;
`aiython run stats: {...}` summarizes the script run. Provider progress messages
can also appear on stderr, so parse by prefix. For each invocation, inspect `model_calls`, `tool_counts`,
`invalid_batches`, `tool_failures`, `provider_errors`, `token_usage`,
`model_call_seconds`, `provider_seconds`, `runtime_seconds`,
`context_build_seconds`, and `request_bytes`. The stats contain counts and
timings, not prompts or values. `provider_seconds` includes SDK and network time;
`request_bytes` estimates serialized model messages and tools, not wire bytes.
`retry_backoff_seconds` remains zero because ambiguous requests are not retried.

The additional fields distinguish local setup, model work, and capabilities:

| Field | Scope |
| --- | --- |
| Run `total_seconds` | Wall time inside `run_script`, including configuration, preparation, execution and cleanup. Excludes interpreter startup and imports before the runner. |
| Run `config_seconds` | Resolve the script path and configuration; small when configuration is supplied directly. |
| Run `parse_seconds`, `prepare_seconds` | Frontend parsing and AST preparation/compilation, or restoration of cached metadata. Includes project modules prepared during execution. |
| Run `execution_seconds` | Execute the prepared entry script, including its imports, model requests, capabilities and normal Python work. |
| Run `preparation_cache_hits`, `preparation_cache_misses` | Process-local preparation cache lookups through `compile_source`. |
| Invocation `invocation_seconds` | Wall time within `ToolAgent.run`/`arun`, including context construction and tool/model rounds; excludes outer request construction and final runtime validation. |
| Invocation `capability_seconds` | Sum of complete capability operations, including routing, cache work and provider calls. |
| Invocation `capability_provider_seconds` | Sum of non-local adapter calls, including failed attempts. |
| Invocation `cache_seconds` | Capability cache-key hashing, lookups and writes, including cache hits. |
| Invocation `sdk_load_seconds` | Time loading the reasoning SDK; included in model/provider timing. |

These are **inclusive timings**, not additive slices: `runtime_seconds` includes
waiting for capability tools, and parallel capability durations overlap each
other. Imported-module preparation overlaps `execution_seconds`. Use run wall
time for end-to-end comparisons, not a sum of these fields. Use a parent-process
timer for CLI startup. Custom agents need their own invocation instrumentation.

The first performance question is where wall time goes. For the configured
provider and model, compare model-call time and completion tokens with local runtime time.
OpenRouter describes total latency as time to first token plus generated tokens
divided by generation throughput. OpenAI's latency guide likewise recommends
reducing generated tokens and sequential requests before optimizing small input
prompts. These are general heuristics, not a prediction of Aiython's speedup.

Sources: [OpenRouter latency guide](https://openrouter.ai/docs/guides/best-practices/latency-and-performance),
[OpenAI latency guide](https://developers.openai.com/api/docs/guides/latency-optimization).

## Evaluate behavior before timing

Keep a fixed set of complete programs covering values, mutation, loops, typed
results, recovery, and capability calls. Include held-out programs and cases
derived from actual failures. Check exact observable results and side effects:
for example, the number of iterations, calls to a scoring function, and whether
the model wrote a file. A plausible final answer alone can conceal an incorrect
execution path. Record failures and tool-call traces as regression cases. Run
each model-backed case repeatedly, because one successful pass does not establish
reliability; report both per-case pass rate and the fraction of cases passing
every repetition. This follows the repeated-trial reliability idea in
[τ-bench](https://arxiv.org/abs/2406.12045) and the guidance to build continuous,
task-specific evals in [OpenAI's evaluation guide](https://developers.openai.com/api/docs/guides/evaluation-best-practices).

## Compare one change at a time

Use the same model, provider, inputs, and correctness checks for baseline and
candidate. Interleave runs (for example A-B-B-A) to reduce drift from provider
load. Compare elapsed time per full program and per invocation, including p50 and
p90; also report calls, input/output tokens, cache hits, errors, and retries.
Only promote a change after its behavior passes the suite and latency gains are
larger than run-to-run variation. For the current project, preserve the existing
gate: all baseline-success cases stay correct, no extra calls or errors per case,
at least 5% less total wall time, and no case more than 10% slower.

The earlier five-case A/B runs are exploratory. In one unchanged ticket case,
two baseline runs produced 237 and 1,334 completion tokens, with 2.9 and 12.9
seconds of wall time. Two runs per variant cannot separate a small optimization
from that variability. Prompt compression and altered terminal tools reduced
request bytes in those runs but did not pass the latency and call-count gate.

## Prioritize the next experiment

1. Correctness and execution semantics first: require each suspended statement
   to produce the right value and effects before judging speed.
2. If model time dominates, investigate extra sequential requests and generated
   tokens. Keep tool results small, but verify that doing so does not remove data
   needed for a correct decision.
3. Inspect `cached_tokens` before changing prompt prefixes. Aiython already puts
   stable context before live state and records cache usage when the provider
   reports it; cache support and hit rates vary by model and provider. See
   [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching).
4. Check cold start separately from warm invocations. LiteLLM loads lazily on the
   first AI call; ordinary Python should not pay for that import.

No finite benchmark proves correctness for every future program. It provides a
repeatable gate and a way to expand coverage when a new failure appears.

## Offline runtime benchmark

   [`benchmarks/runtime_overhead.py`](https://github.com/sunmodza/aiython/blob/main/benchmarks/runtime_overhead.py) measures
local execution, repeated preparation, SQLite cache reads, and fresh-process
startup without contacting a model. The AI-invocation fixture uses a fake
provider that completes in one response. Every execution checks its result.

```sh
uv run --locked python benchmarks/runtime_overhead.py --output after.json
uv run --locked python benchmarks/runtime_overhead.py \
  --source-root /path/to/before/src --output before.json
```

Keep the interpreter and dependency environment fixed. Run A-B-B-A without
concurrent tests or benchmarks. Each file contains seven measured repetitions
after a warm-up, source and runner hashes, dependency versions and the raw
samples. Repeated preparation measures a warm process-local cache; startup uses
a fresh process but filesystem caches may be warm. These are different cases.

See the [recorded comparison](https://github.com/sunmodza/aiython/blob/main/benchmarks/README.md) and its raw JSON evidence.
Time reduction is `100 * (before - after) / before`; speedup is `before / after`.
A 50% time reduction is 2x speed, not a 50% throughput increase. Report the case
and timing scope with each claim. These results do not establish model-backed
task speedup or a comparison against other agent products.

The runtime retains all type checkpoints. Primitive contracts avoid repeated
compilation and diagnostic allocation on successful validation, but mutable
containers are still inspected on every required check. Repeatedly appending to
a growing typed list can therefore still take quadratic total validation work.
Skipping a check based only on identity or length would miss changes through
aliases and foreign helpers. Provider latency and unnecessary model rounds also
need separate live evaluation; prompt guidance alone is not evidence of a gain.
