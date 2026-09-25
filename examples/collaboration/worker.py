"""Importable worker for the parallel collaboration example."""
from aiython import join


def process_item(ticket, item):
    with join(ticket) as participant:
        participant.send("main", {"item": item, "result": item * item})
        return item * item
