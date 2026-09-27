"""Import-safe main module used by multiprocessing spawn and forkserver.

The parent CLI sets AIYTHON_SPAWN_ENTRY only while an Aiython script is running.
Python imports this module as ``__mp_main__`` in a worker, matching normal
``if __name__ == '__main__'`` behavior while compiling Aiython source first.
"""
from __future__ import annotations

if __name__ == "__mp_main__":
    import os
    from pathlib import Path
    import sys

    from aiython.cli import ProjectFinder, read_source
    from aiython.config import resolve
    from aiython.frontend import runtime_binding_name
    from aiython.runtime import Runtime

    source_path = Path(os.environ["AIYTHON_SPAWN_ENTRY"])
    runtime = Runtime(resolve(source_path))
    source = read_source(source_path)
    globals().update({"__file__": str(source_path), "__package__": None,
                      runtime_binding_name(source): runtime})
    sys.path.insert(0, str(source_path.parent))
    sys.meta_path.insert(0, ProjectFinder(runtime))
    exec(runtime.compile_source(source, str(source_path)), globals())
