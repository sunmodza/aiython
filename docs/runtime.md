# How the runtime works

Aiython extends Python where parsing or execution cannot proceed. Valid Python runs normally. The frontend turns unparseable source spans into AI calls, and eligible unhandled exceptions can reach recovery checkpoints. Natural language is one example of such source. Python owns statement order, loops, and side effects; the model handles the current invocation.

Use `aiython --explain PATH` to see the detected blocks and checkpoints before a run. A valid Python expression such as `result = unknown_name` follows Python semantics first and may enter recovery after a `NameError`. Ambiguous source can become a larger AI block, so inspect boundaries if a side effect must happen at a precise point.

Inside loops and methods, recovery checkpoints surround individual statements. If `ticket.age` fails after `seen.append(ticket)`, a `retry` repeats only `ticket.age`, so the append and earlier iterations are not replayed. A failure in a loop header or another enclosing statement cannot use `retry` when it would replay prior effects; the model can repair state and complete or reraise. A statement that already made partial changes before failing is not rolled back.

## Live frame and result types

The model can use tools to read bindings, inspect object handles, evaluate expressions, or execute Python against the suspended frame. A handle refers to the original object, so returning it preserves identity. A type annotation on the target constrains the value before assignment; the same [type contracts](type-safety.md) apply to normal Python and AI results.

These tools have the permissions of the running process. `eval` and `exec` are available when the profile permits `execute_code`; they are not a security boundary. Relevant source and object metadata are included in model context, and model tools can inspect more when permitted. Use trusted source code and a provider whose data handling meets your needs.

## Model tool protocol

A model response may contain a batch of tool calls. A completed AI invocation ends with one `finish` outcome. The outcome has a single `kind`:

```text
finish(outcome={"kind":"literal","value":"The answer is 42"})
finish(outcome={"kind":"expression","code":"x + y"})
finish(outcome={"kind":"reference","id":"answer"})
finish(outcome={"kind":"handle","id":"object-1"})
finish(outcome={"kind":"none"})
finish(outcome={"kind":"error","reason":"No route is configured"})
```

For example, `evaluate(code="x + y", result_id="answer")` and a reference outcome can return a live object in one response. `execute` performs assignments and mutations. A result-required expression cannot finish with `none`; a standalone side-effect statement can. Recovery uses `recover(action="complete", outcome=...)` for a replacement, or `retry` and `reraise` without an outcome.

Aiython validates the entire batch before any tool runs. The terminal call must be last. Malformed batches have no effects; two invalid batches or repeated empty responses stop with a clear error. Once a tool has changed state, Aiython does not roll it back or silently replay it. Completed capability steps are tracked across a repair attempt so the model can reuse them rather than repeat a submission.

## Capabilities and parallel work

Configured capabilities run through a validated plan that can refer to previous step results, live bindings, or object handles. Some local pure steps can run concurrently; Python still determines when the plan is requested. See [capabilities](capabilities.md) for routes and bundled file examples.

Python also owns task and worker creation. A collaboration group can attach optional message tools to AI invocations across threads, coroutines, and processes. Async model and blocking capability calls leave the event loop available to other coroutines. See [collaboration](collaboration.md) for its mailbox and worker behavior.

The default invocation deadline is 120 seconds, with a separate resumable deadline for video generation. A request whose submission status is uncertain is not retried automatically. Use `--stats` to inspect actual calls and time, and [measure performance](performance-evaluation.md) against correctness before comparing provider latency.

## Reusing local work

The entry runner, project importer and subprocess loader use a bounded,
process-local preparation cache keyed by source content, filename, Python cache
tag, runtime class and entry mode. Compiled code can be reused; each runtime gets
fresh AST/directive metadata and keeps its own profiles, agents and live objects.
No prepared code or metadata is loaded from a disk cache. The cache holds at most
32 programs and 8 MiB of serialized metadata; source larger than 256 KiB bypasses
it. A new process starts with an empty cache.

Static source hints are reused within a runtime; source digests and small tool
expressions/snippets use bounded caches. Tool execution still reads the current frame, and container
metadata and type validation inspect current values. Configuration validation
loads lazily when a project config is present; the model SDK loads on demand.

A capability runtime reuses up to four plan workers and serialized SQLite
connections. Provider calls do not hold the database lock. Nested plans inside a
plan worker execute their ready steps sequentially to avoid waiting on the same
saturated worker pool. Connections and workers reopen after a fork or an explicit
close. The script runner closes these resources at exit; embedded callers should
call `runtime.capabilities.close()` when finished with a runtime.
