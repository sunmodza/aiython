# Get started

Use CPython **3.11 or newer**. Aiython's command, PyPI package, and Python import are all named `aiython`.

## 1. Install Aiython

Choose the installation that matches where your script runs:

| Where you run scripts | Install | Run |
| --- | --- | --- |
| Existing [uv](https://docs.astral.sh/uv/) project | `uv add aiython` | `uv run aiython ...` |
| Activated virtual environment | `python -m pip install aiython` or `uv pip install aiython` | `aiython ...` |
| Standalone command | `uv tool install aiython` | `aiython ...` |

Use a project environment if your script imports other project dependencies. The standalone tool has an isolated environment.

## 2. Save a script

Save this as `tickets.py`:

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

Python controls the loop and the queue updates. Aiython resolves `classify this ticket` for each ticket, then resolves the summary after the loop.

## 3. Set up a model

```bash
aiython setup
```

Choose a model that supports tool calling. Setup creates or updates `aiython.toml` and can store an entered key in the gitignored `.aiython/credentials.env` file. If you installed Aiython in a uv project, use `uv run aiython setup` instead. See [configuration](configuration.md) for profiles, custom endpoints, and capability routes.

## 4. Inspect, then run

```bash
aiython --explain tickets.py
aiython tickets.py
```

`--explain` shows the AI boundary without executing the script or contacting a provider. Running the script uses your chosen provider and may incur charges. For a uv project, prefix both commands with `uv run`.

Run `aiython` without a script in a terminal for an interactive console. It keeps variables, type annotations, and future imports between commands, and accepts AI expressions. Pass `--profile NAME` or `--config PATH` to select its model configuration. Piped input runs as a Python script, with or without an explicit `-` argument.

## Next steps

- Try the [small examples](examples.md) to see one behavior at a time.
- Read [how the runtime works](runtime.md) to understand live objects and recovery.
- Add [capability routes](capabilities.md) only when a script needs documents, vision, speech, images, or video.
