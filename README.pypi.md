# Aiython

**When Python doesn’t know what to do, Aiython does.**

**Run an existing Python script by replacing the command:**

```text
python app.py  →  aiython app.py
```

The same works for modules: `python -m package` → `aiython -m package`. Your script and its arguments stay the same. Aiython checks declared types and can call AI for inline requests or eligible errors.

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

Python runs the loop and updates `queues`. Aiython handles the inline requests using live program state and checks the declared `Literal` result.

## Try it

Save the example as `tickets.py`:

```bash
pip install aiython
aiython setup
aiython tickets.py
```

Preview AI calls without running the script: `aiython --explain tickets.py`.

[Documentation](https://sunmodza.github.io/aiython-docs/) · [Examples](https://sunmodza.github.io/aiython-docs/examples/) · [Installation](https://sunmodza.github.io/aiython-docs/getting-started/)

Aiython is not a sandbox: AI tools run with your process permissions, and relevant code or data may be sent to your provider. MIT licensed.
