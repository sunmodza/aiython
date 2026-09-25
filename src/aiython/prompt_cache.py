"""Compact, deterministic JSON for model context and tool results."""
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
