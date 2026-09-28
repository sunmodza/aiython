"""Build an exact CPython baseline before applying Aiython VM changes.

This produces stock CPython; it does not enable Aiython's type or recovery
features. Example: python scripts/build_cpython_baseline.py 3.14.4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
LOCK = tomllib.loads((ROOT / "native/cpython.lock.toml").read_text())["versions"]


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def available_stdlib(python: Path) -> set[str]:
    probe = ("import importlib.util, json, sys\n"
             "def available(name):\n"
             "    try: return importlib.util.find_spec(name) is not None\n"
             "    except Exception: return False\n"
             "print(json.dumps(sorted(name for name in sys.stdlib_module_names "
             "if available(name))))\n")
    output = subprocess.check_output([str(python), "-I", "-c", probe], text=True)
    return set(json.loads(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=LOCK)
    parser.add_argument("--build-root", type=Path,
                        default=Path(tempfile.gettempdir()) / "aiython-cpython-baselines")
    parser.add_argument("--source-dir", type=Path, help="reuse an existing CPython checkout")
    parser.add_argument("--reconfigure", action="store_true",
                        help="rerun configure after changing build dependencies or flags")
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    source = ((args.source_dir or args.build_root / f"cpython-{args.version}").resolve())
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--branch",
            f"v{args.version}", "https://github.com/python/cpython.git", str(source))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != LOCK[args.version]:
        raise SystemExit(f"CPython {args.version}: expected {LOCK[args.version]}, got {commit}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=source):
        raise SystemExit('CPython source has tracked changes; baseline must be unmodified')
    if args.reconfigure or not (source / "Makefile").exists():
        run("./configure", f"--prefix={source / 'install'}", cwd=source)
    run("make", "-s", f"-j{args.jobs}", cwd=source)
    actual = subprocess.check_output([str(source / "python"), "-c",
                                      "import json, sys, sysconfig; "
                                      "print('.'.join(map(str, sys.version_info[:3]))); "
                                      "print(sysconfig.get_config_var('CONFIG_ARGS')); "
                                      "print(json.dumps(sorted(k for k, v in "
                                      "sysconfig.get_config_vars().items() if "
                                      "k.startswith('MODULE_') and k.endswith('_STATE') "
                                      "and v == 'missing')))"],
                                     cwd=source, text=True).strip()
    version, config_args, missing_modules_json = actual.splitlines()
    if version != args.version:
        raise SystemExit(f"Built interpreter reported {version}, expected {args.version}")
    if '--disable-test-modules' in config_args or '--without-ensurepip' in config_args:
        raise SystemExit('Build omitted CPython modules or ensurepip; reconfigure the checkout')
    missing_modules = json.loads(missing_modules_json)
    if missing_modules:
        raise SystemExit(f"Built interpreter has missing optional extensions: {', '.join(missing_modules)}")
    if sys.version_info[:3] != tuple(map(int, args.version.split('.'))):
        raise SystemExit('Run this script with the same CPython patch version as the build')
    missing = available_stdlib(Path(sys.executable)) - available_stdlib(source / "python")
    if missing:
        raise SystemExit(f"Built interpreter lacks reference stdlib modules: {', '.join(sorted(missing))}")
    print(f"Verified CPython {args.version} at {source / 'python'} ({commit})")


if __name__ == "__main__":
    main()
