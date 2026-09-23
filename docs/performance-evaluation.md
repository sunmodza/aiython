# Measuring Aithon changes

Aithon executes one suspended Python statement at a time. A model call is an
implementation step inside that invocation, not a new Python iteration. Measure
the program's resulting values and side effects as well as the model interaction.

## What to measure

Run `aithon --stats script.py` and retain the JSON line from stderr alongside the
program output. For each invocation, inspect `model_calls`, `tool_counts`,
`invalid_batches`, `tool_failures`, `provider_errors`, `token_usage`,
`model_call_seconds`, `provider_seconds`, `runtime_seconds`,
`context_build_seconds`, and `request_bytes`. The stats contain counts and
timings, not prompts or values. `provider_seconds` includes SDK and network time;
`request_bytes` estimates serialized model messages and tools, not wire bytes.
`retry_backoff_seconds` remains zero because ambiguous requests are not retried.

The first performance question is where wall time goes. For the configured
provider and model, compare model-call time and completion tokens with local runtime time.
OpenRouter describes total latency as time to first token plus generated tokens
divided by generation throughput. OpenAI's latency guide likewise recommends
reducing generated tokens and sequential requests before optimizing small input
prompts. These are general heuristics, not a prediction of Aithon's speedup.

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
3. Inspect `cached_tokens` before changing prompt prefixes. Aithon already puts
   stable context before live state and records cache usage when the provider
   reports it; cache support and hit rates vary by model and provider. See
   [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching).
4. Check cold start separately from warm invocations. LiteLLM loads lazily on the
   first AI call; ordinary Python should not pay for that import.

No finite benchmark proves correctness for every future program. It provides a
repeatable gate and a way to expand coverage when a new failure appears.
