"""Importable worker for the parallel collaboration example."""
from aithon import join


def process_item(ticket, item):
    with join(ticket) as participant:
        participant.send("main", {"item": item, "result": item * item})
        return item * item
