"""Importable worker that needs no model or configuration dependencies."""

from aithon import join


def send(ticket):
    with join(ticket) as participant:
        participant.send("main", 42)
    return 7
