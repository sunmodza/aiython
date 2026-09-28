<p align="center"><img src="assets/readme/icon.png" width="88" alt="Aiython icon"></p>

<h1 align="center">Aiython</h1>

<p align="center"><strong>When Python doesn’t know what to do, Aiython does.</strong></p>

<p align="center"><a href="docs/index.md">Documentation</a> · <a href="examples/recipes/README.md">Examples</a></p>

Aiython runs Python normally and can bring in AI when Python cannot parse the source or continue execution. AI works with the live program state; Python keeps control of execution.

## Example

![Python routes tickets in a loop; Aiython handles the inline requests.](assets/readme/runtime-debug.gif)

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

Python runs the loop and updates `queues`. Aiython handles the inline requests using live program state and checks the declared `Literal` result.

## Try it

**Run an existing Python script by replacing the command:**

```text
python app.py  →  aiython app.py
```

```bash
pip install aiython
aiython setup
aiython app.py
```

The same works for modules: `python -m package` → `aiython -m package`. Your script and its arguments stay the same. Aiython checks declared types and can call AI for inline requests or eligible errors.

Aiython is not a sandbox: AI tools run with your process permissions, and relevant code or data may be sent to your provider.

[Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
