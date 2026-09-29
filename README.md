<p align="center">
  <img src="assets/readme/icon.png" width="88" alt="Aiython icon">
</p>

<h1 align="center">Aiython</h1>

<p align="center">
  <strong>Put AI at the exact line where your Python program needs it.</strong>
</p>

<p align="center">
  Keep Python in control of the loop. Use typed, inline AI requests where you need judgment.
</p>

<p align="center">
  <a href="docs/index.md">Documentation</a> ·
  <a href="examples/recipes/README.md">Examples</a> ·
  <a href="https://youtu.be/CMTDqCJqunc">30-second concept video</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img
    src="assets/readme/runtime-debug.gif"
    alt="Python routes tickets in a loop while Aiython handles inline AI instructions."
  >
</p>

Aiython runs ordinary Python. At an inline instruction or eligible runtime
error, it can ask a configured model using live program state, check the
result, and resume execution.

```python
kind: Literal["bug", "billing"] = classify this ticket
```

That is not valid Python syntax on its own. Aiython resolves it at runtime;
Python still controls the loop and uses the checked `Literal` value.

## Quick start

Use Python 3.11 or newer. To inspect the included example **without an API key
or a model call**, clone the repo and run:

```bash
git clone https://github.com/sunmodza/aiython.git
cd aiython
uv run --locked aiython --explain examples/recipes/03_loop.py
```

`--explain` shows the detected AI instructions and recovery checkpoints. It
does not run the script. For an actual model-backed run:

### 1. Install

For a standalone command with `uv`:

```bash
uv tool install aiython
```

Or in your project's Python environment:

```bash
pip install aiython
```

### 2. Set up your AI provider

```bash
aiython setup
```

Choose a tool-calling model. Calls can incur provider charges; review the
[permissions and data exposure](#security) before running unfamiliar scripts.

### 3. Run your Python script with Aiython

```bash
aiython app.py
```

You can inspect your own script first with `aiython --explain app.py`.

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
