"""Build and test an exact CPython release with Aiython experimental hooks.

The output is a patched CPython interpreter, not a complete Aiython runtime.
Run with the same Python patch version as the requested source release.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = tomllib.loads((ROOT / "native/cpython.lock.toml").read_text())["versions"]
CPYTHON_TESTS = (
    "test_compile", "test_dis", "test_scope", "test_sys", "test_frame",
    "test_exceptions", "test_generators", "test_coroutines",
)


def run(*command: str, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=VERSIONS)
    parser.add_argument("--source-dir", type=Path,
                        help="clean CPython checkout; defaults to a temporary build directory")
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = (args.source_dir or Path(tempfile.gettempdir()) / f"aiython-cpython-hooks-{args.version}").resolve()
    patch = ROOT / "native/patches" / f"cpython-{args.version}-hooks.patch"
    if not patch.is_file():
        raise SystemExit(f"Missing pinned CPython patch: {patch}")

    run(sys.executable, str(ROOT / "scripts/build_cpython_baseline.py"), args.version,
        "--source-dir", str(source), "--jobs", str(args.jobs))
    run("git", "apply", "--check", str(patch), cwd=source)
    run("git", "apply", str(patch), cwd=source)
    if args.version.startswith("3.12."):
        run("make", "regen-cases", cwd=source)
    elif not args.version.startswith("3.11."):
        run("make", "regen-generated-cases", cwd=source)
    run("make", "-s", f"-j{args.jobs}", cwd=source)
    run(str(source / "python"), "-I", str(ROOT / "native/tests/store_hook.py"))
    run("uv", "run", "--locked", "--python", str(source / "python"),
        "python", "native/tests/typed_bridge.py")
    run(str(source / "python"), "-I", "-m", "test", "-q", *CPYTHON_TESTS, cwd=source)
    print(f"Experimental CPython {args.version} hooks verified at {source / 'python'}")


if __name__ == "__main__":
    main()
