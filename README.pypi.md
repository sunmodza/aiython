# Aiython

**When Python doesn’t know what to do, Aiython does.**

Aiython runs Python normally and can bring in AI when Python cannot parse the source or continue execution. AI works with the live program state; Python keeps control of execution.

```python
from typing import Literal

inbox = ["Upload crashes", "Invoice please"]
queues = {"bug": [], "billing": []}

for message in inbox:
    kind: Literal["bug", "billing"] = classify this message
    queues[kind].append(message)

print(queues)
```

Save this as `inbox.py` and run it with `aiython`; Python cannot parse `classify this message` on its own.

## Install and run

Use CPython 3.11 or newer. For a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add aiython
uv run aiython setup
uv run aiython --explain inbox.py
uv run aiython inbox.py
```

For an activated virtual environment, use `python -m pip install aiython` or `uv pip install aiython`, then run `aiython setup` and `aiython inbox.py`. For a standalone CLI, use `uv tool install aiython`; this gives you the `aiython` command in an isolated environment. Install Aiython in the project environment when your script imports other project dependencies.

`--explain` shows where AI is invoked without executing the script or calling a model. `setup` creates or updates project configuration, lets you choose a tool-capable model, and stores an entered key in a private, gitignored project file. Running the final command uses your chosen provider and may incur charges.

The PyPI distribution, CLI, and Python import are all named **`aiython`**.

## What stays in Python

Python owns statement order, loops, assignments, and side effects. AI works at the current execution boundary, with access to live objects and checks on declared result types. Ordinary Python code runs without loading LiteLLM or contacting a provider.

Use `aiython --help` to see the CLI, `aiython config show` to inspect configuration, and `aiython --stats inbox.py` to see model calls and timings. Additional routes for vision, documents, embeddings, reranking, speech, images, and video are configured only when needed.

Aiython is not a sandbox. Frame tools can use `eval` and `exec` with your process permissions, and relevant source or object data may be sent to your configured provider. Use trusted code and review provider data handling.

Licensed under MIT.
