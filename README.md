<p align="center">
  <img src="assets/readme/icon.png" width="88" alt="Aiython icon">
</p>

<h1 align="center">Aiython</h1>

<p align="center">
  <strong>When Python doesn’t know what to do, Aiython does.</strong>
</p>

<p align="center">
  Run ordinary Python with inline AI instructions and AI-assisted recovery.
</p>

<p align="center">
  <a href="docs/index.md">Documentation</a> ·
  <a href="examples/recipes/README.md">Examples</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img
    src="assets/readme/runtime-debug.gif"
    alt="Python routes tickets in a loop while Aiython handles inline AI instructions."
  >
</p>

Aiython runs Python normally. When Python cannot parse an inline instruction
or cannot continue through an eligible error, Aiython can hand that part to AI
using the live program state, validate the result, and let Python continue.

```python
kind: Literal["bug", "billing"] = classify this ticket
```

That is not normal Python syntax. With Aiython, it can become part of a running
Python program.

## Quick start

### 1. Install

With `uv`:

```bash
uv tool install aiython
```

Or with `pip`:

```bash
pip install aiython
```

### 2. Set up your AI provider

```bash
aiython setup
```

### 3. Run your Python script with Aiython

Replace:

```bash
python app.py
```

with:

```bash
aiython app.py
```

Modules work the same way:

```bash
python -m package
# becomes
aiython -m package
```

## Example

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

Python owns the loop, variables, and normal execution.

Aiython handles the inline requests using the live program state. The declared
`Literal["bug", "billing"]` also constrains the result before execution continues.

## What Aiython does

- **Runs ordinary Python normally.** Existing Python remains Python.
- **Handles inline AI instructions.** Natural-language intent can appear directly in the program.
- **Works with live program state.** AI can reason about relevant values already available during execution.
- **Validates declared types.** Type annotations can constrain AI-generated results.
- **Can assist with eligible runtime failures.** Python stays in control while Aiython provides a recovery path.

Aiython is a runtime layer, not a replacement programming language and not just
another LLM API wrapper.

## How it works

```text
Python source
     │
     ▼
Normal Python execution
     │
     ├── Python can continue ───────────────► Python
     │
     └── AI-eligible instruction / failure
                     │
                     ▼
               Aiython bridge
                     │
              live program state
                     │
                     ▼
                 AI provider
                     │
              validate result
                     │
                     ▼
             continue execution
```

Python remains responsible for ordinary execution. Aiython only steps in for
supported inline instructions or eligible recovery paths.

## Security

Aiython is **not a sandbox**.

AI tools run with the permissions of your Python process. Relevant source code,
runtime values, or other context may be sent to the configured AI provider.

Review the documentation before using Aiython with sensitive data or
high-privilege environments.

## Documentation

See the [documentation](docs/index.md) for configuration, providers, runtime
behavior, type handling, recovery, and security details.

For runnable examples, see [examples](examples/recipes/README.md).

---

[MIT License](LICENSE)
