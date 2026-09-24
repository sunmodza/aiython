"""Import-safe main module used by multiprocessing spawn and forkserver.

The parent CLI sets AITHON_SPAWN_ENTRY only while an Aithon script is running.
Python imports this module as ``__mp_main__`` in a worker, matching normal
``if __name__ == '__main__'`` behavior while compiling Aithon source first.
"""
from __future__ import annotations

if __name__ == "__mp_main__":
    import os
    from pathlib import Path
    import sys

    from aithon.cli import ProjectFinder, read_source
    from aithon.config import resolve
    from aithon.frontend import RUNTIME_NAME, parse
    from aithon.runtime import Runtime

    source_path = Path(os.environ["AITHON_SPAWN_ENTRY"])
    runtime = Runtime(resolve(source_path))
    unit = parse(read_source(source_path), str(source_path))
    globals().update({"__file__": str(source_path), "__package__": None,
                      RUNTIME_NAME: runtime})
    sys.path.insert(0, str(source_path.parent))
    sys.meta_path.insert(0, ProjectFinder(runtime))
    exec(runtime.prepare(unit), globals())
