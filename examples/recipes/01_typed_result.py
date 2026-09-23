"""Return a structured value whose fields are checked at runtime."""
from typing import Literal, TypedDict


class Draft(TypedDict):
    priority: Literal["low", "normal", "urgent"]
    reply: str


message = "I cannot sign in and my presentation starts in 20 minutes."

draft: Draft = read message and choose a priority and draft a reply for the customer

print(draft["priority"])
print(draft["reply"])
