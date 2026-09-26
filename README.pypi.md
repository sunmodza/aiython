# Aiython

**When Python doesn’t know what to do, Aiython does.**

Aiython runs Python normally and can bring in AI when Python cannot parse the source or continue execution. AI works with the live program state; Python keeps control of execution.

```python
from typing import Literal

tickets = [
    "After uploading a PDF, the ticket page freezes until I refresh the browser.",
    "Could you email last month's invoice and update the billing contact for our team?",
]
queues = {"bug": [], "billing": []}

for ticket in tickets:
    kind: Literal["bug", "billing"] = classify this ticket
    queues[kind].append(ticket)

summary = summarize the routed tickets in one sentence
print(queues, summary)
```

Save this as `tickets.py` and run it with `aiython`; AI classifies each ticket, then summarizes the completed queues once.

## Install and run

Use CPython 3.11 or newer. For a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add aiython
uv run aiython setup
uv run aiython --explain tickets.py
uv run aiython tickets.py
```

For an activated virtual environment, use `python -m pip install aiython` or `uv pip install aiython`, then run `aiython setup` and `aiython tickets.py`. For a standalone CLI, use `uv tool install aiython`; this gives you the `aiython` command in an isolated environment. Install Aiython in the project environment when your script imports other project dependencies.

`--explain` shows where AI is invoked without executing the script or calling a model. `setup` creates or updates project configuration, lets you choose a tool-capable model, and stores an entered key in a private, gitignored project file. Running the final command uses your chosen provider and may incur charges.

The PyPI distribution, CLI, and Python import are all named **`aiython`**.

## What stays in Python

Python owns statement order, loops, assignments, and side effects. AI works at the current execution boundary, with access to live objects and checks on declared result types. Ordinary Python code runs without loading LiteLLM or contacting a provider.

Use `aiython --help` to see the CLI, `aiython config show` to inspect configuration, and `aiython --stats tickets.py` to see model calls and timings. Additional routes for vision, documents, embeddings, reranking, speech, images, and video are configured only when needed.

Aiython is not a sandbox. Frame tools can use `eval` and `exec` with your process permissions, and relevant source or object data may be sent to your configured provider. Use trusted code and review provider data handling.

Licensed under MIT.
