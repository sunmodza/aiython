"""Python routes each message; AI classifies only the current one."""
from typing import Literal


inbox = ["Upload crashes", "Invoice please"]
queues = {"bug": [], "billing": []}

for message in inbox:
    kind: Literal["bug", "billing"] = classify this message
    queues[kind].append(message)

print(queues)
