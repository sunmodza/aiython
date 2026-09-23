"""Return an existing Python object, preserving its identity."""
from dataclasses import dataclass


@dataclass
class Guide:
    title: str
    description: str


guides = [
    Guide("Reset a password", "Recover access to an existing account"),
    Guide("Export reports", "Download analytics as CSV files"),
    Guide("Invite a teammate", "Add another person to a workspace"),
]
question = "How do I download my analytics?"

selected: Guide = choose the Guide in guides that best answers question and return that same object

assert any(selected is guide for guide in guides)
print(selected.title)
