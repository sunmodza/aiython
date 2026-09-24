# Collaboration across parallel Python work

Python owns task creation, scheduling, loops, and side effects. Aithon offers an
optional group mailbox when tasks need to exchange messages. Ordinary scripts do
not create a broker or add collaboration tools to model requests.

```python
from concurrent.futures import ThreadPoolExecutor
from aithon import group, join

def worker(ticket):
    with join(ticket) as me:
        me.send("main", {"status": "ready"})
        return me.wait(timeout=5)[0]["payload"]

with group() as team:
    ticket = team.invite("worker")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(worker, ticket)
        print(team.wait(timeout=5))
        team.send("worker", {"command": "finish"})
        print(result.result())
```

The same `join(ticket)` call works inside an `asyncio` task, a thread, or a
separate process. Use `await me.wait_async(timeout=...)` in coroutines so the
event loop remains free. A participant ticket belongs to one invited name; pass
it explicitly to a worker that does not inherit the current Python context.
`contextvars` inheritance across threads is not guaranteed, so relying on
implicit propagation would produce different results on different Python builds.
AI model calls inside `async def` use LiteLLM's async gateway. Blocking
capability adapters run in a worker thread for the duration of the plan, so
other coroutines can continue while a capability call is in flight.

For `ProcessPoolExecutor` with `spawn` or `forkserver`, submit the importable
`aithon.worker_entry` when the worker is in a project module:

```python
from aithon import worker_entry

future = pool.submit(worker_entry, ticket, "my_project.workers", "run", payload)
```

The worker function receives the ticket as its first argument and calls
`join(ticket)` inside the task it wants to register. The CLI provides an
import-safe main-module bootstrap while a group is active, so a process can
re-import an Aithon entry script without parsing its natural-language blocks as
plain Python. The worker function must still be importable, as Python's process
executors require. `worker_entry` installs the project loader before importing
the worker, so Aithon blocks in the worker or its imported project modules are
handled the same way. Under `InterpreterPoolExecutor`, it uses an isolated
Python process because the current `pydantic-core` native extension cannot
load in a subinterpreter. The worker still joins the same group and returns
its result through the executor. A subinterpreter reuses its isolated worker
process for later calls to the same project, so process startup happens once
per interpreter and project. Plain Python workers can use `join(ticket)`
directly in a subinterpreter without the wrapper.

An attached AI invocation receives `list_peers`, `send_message`, and
`read_messages` tools. A message is available in the mailbox immediately, but
it does not interrupt an in-flight model request or cause a new model request.
At the next invocation, the agent sees its unread count and can read messages.
If the suspended statement only needs to return its inbox, it can call
`read_messages(result_id="inbox")` and `finish` with a reference to `inbox` in
one model response.

Within an async AI block, model calls and blocking capability plans run off the
event loop. Frame reads, writes, result handles, and output validation stay on
the originating task. A capability plan sees a snapshot of the frame bindings
taken after earlier tools in the same batch have completed; referenced Python
objects retain their identity.

Payloads are JSON values up to 64 KiB. Each recipient has a 100-message queue;
overflow raises `BufferError`. `send` acknowledges mailbox acceptance, not
recipient processing. A caller may reuse a message ID for an explicit retry;
the broker deduplicates recent IDs. `read` removes messages, so applications
that need durable delivery should persist their own work and acknowledge it at
the application level. Group tickets are bearer credentials: do not log or
publish them. Closing a group invalidates its tickets and wakes waiters.

For a separately hosted A2A agent, install `aithon[a2a]` and call
`await participant.send_a2a(url, payload)`. This uses the official A2A SDK,
returns its task/message events, and does not retry an uncertain submission.
For bearer authentication, pass `api_key_env="MY_A2A_KEY"`; Aithon reads the
environment variable and does not store the key in a ticket or message.
Local mailbox traffic does not use A2A or HTTP.
