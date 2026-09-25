"""Isolated worker for native extensions unavailable in subinterpreters."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import sys


def _run(request: Path) -> bytes:
    from .collaboration import worker_entry

    with request.open("rb") as file:
        ticket, module, function, args, kwargs = pickle.load(file)
    try:
        result = ("ok", worker_entry(ticket, module, function, *args, **kwargs))
    except BaseException as exc:
        result = ("error", type(exc).__name__, str(exc))
    try:
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        return pickle.dumps(("error", type(exc).__name__, str(exc)),
                            protocol=pickle.HIGHEST_PROTOCOL)


def main():
    # Keep the control pipe private. User code still sees EOF on stdin, while
    # stdout and stderr remain attached to the caller's terminal.
    control = sys.stdin.buffer
    with open(os.devnull) as empty:
        sys.stdin = empty
        for line in control:
            request_name, result_name = json.loads(line)
            result = Path(result_name)
            data = _run(Path(request_name))
            temporary = result.with_suffix(".tmp")
            with temporary.open("xb") as file:
                file.write(data)
            os.replace(temporary, result)


if __name__ == "__main__":
    main()
