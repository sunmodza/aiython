"""Python owns the loop; Aithon handles the current item each iteration."""
from typing import Literal


messages = [
    "Please send me the invoice for last month.",
    "The app crashes whenever I upload a photo.",
    "Can your product export CSV files?",
]
labels: list[str] = []

for message in messages:
    label: Literal["billing", "bug", "question"] = classify only the current message
    labels.append(label)

print(list(zip(messages, labels)))
