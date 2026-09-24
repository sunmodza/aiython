from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import sys
import tokenize
import types
from pathlib import Path

from .config import describe, resolve
from .frontend import RUNTIME_NAME, parse
from .models import AithonError, ConfigError
from .runtime import Runtime


def read_source(path):
    try:
        with tokenize.open(path) as file:
            return file.read()
    except OSError as exc:
        raise AithonError(f"Cannot read source file: {path} ({type(exc).__name__})") from None


class ProjectLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, runtime):
        super().__init__(name, path)
        self.runtime = runtime

    def exec_module(self, module):
        unit = parse(read_source(self.path), self.path)
        module.__dict__[RUNTIME_NAME] = self.runtime
        exec(self.runtime.prepare(unit), module.__dict__)


class ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.config.project_root.resolve()
        self.aithon_root = Path(__file__).parent.resolve()

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if not spec or not isinstance(spec.loader, importlib.machinery.SourceFileLoader) or not spec.origin:
            return None
        location = Path(spec.origin).resolve()
        if not location.is_relative_to(self.root) or location.is_relative_to(self.aithon_root):
            return None
        relative = location.relative_to(self.root)
        if any(part in {".venv", "venv", "site-packages", "dist-packages", "__pycache__"} for part in relative.parts):
            return None
        spec.loader = ProjectLoader(fullname, spec.origin, self.runtime)
        return spec


def run_script(path: Path, arguments=(), *, config=None, agent_factory=None, stats=False, trace_plan=False):
    path = path.resolve()
    runtime = Runtime(config or resolve(path), agent_factory=agent_factory, stats=stats, trace_plan=trace_plan)
    unit = parse(read_source(path), str(path))
    code = runtime.prepare(unit, entry=True)
    module = types.ModuleType("__main__")
    module.__dict__.update({"__file__": str(path), "__package__": None,
                            "__spec__": None,
                            "__cached__": None, RUNTIME_NAME: runtime,
                            "__builtins__": __builtins__})
    old_main = sys.modules.get("__main__")
    old_argv, old_path = sys.argv, sys.path[:]
    old_spawn_entry = os.environ.get("AITHON_SPAWN_ENTRY")
    finder = ProjectFinder(runtime)
    try:
        sys.modules["__main__"] = module
        sys.argv = [str(path), *arguments]
        sys.path.insert(0, str(path.parent))
        sys.meta_path.insert(0, finder)
        os.environ["AITHON_SPAWN_ENTRY"] = str(path)
        exec(code, module.__dict__)
        return module.__dict__
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        sys.meta_path.remove(finder)
        if old_spawn_entry is None:
            os.environ.pop("AITHON_SPAWN_ENTRY", None)
        else:
            os.environ["AITHON_SPAWN_ENTRY"] = old_spawn_entry
        if old_main is not None:
            sys.modules["__main__"] = old_main
        else:
            sys.modules.pop("__main__", None)
        runtime.stats.report()


def parser():
    result = argparse.ArgumentParser(
        prog="aithon", description="Python with project-scoped AI execution",
        epilog="First run: aithon setup. Inspect configuration: aithon config show.")
    result.add_argument("--config")
    group = result.add_mutually_exclusive_group()
    group.add_argument("--profile", help="Default profile; source directives may override it")
    group.add_argument("--force-profile", help="Use this profile for every AI invocation")
    result.add_argument("--explain", action="store_true", help="Show blocks/checkpoints without executing code")
    result.add_argument("--trace-plan", action="store_true", help="Trace actual capability routes, cache and timings")
    result.add_argument("--stats", action="store_true", help="Report model calls, tools, request bytes and timings on stderr")
    result.add_argument("script")
    result.add_argument("args", nargs=argparse.REMAINDER)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["setup"]:
            from .setup import setup
            setup(arguments[1:])
            return
        if arguments[:1] == ["jobs"]:
            command = argparse.ArgumentParser(prog="aithon jobs")
            command.add_argument("action", choices=("list", "resume"))
            command.add_argument("operation", nargs="?")
            command.add_argument("--script", default="main.py")
            command.add_argument("--config")
            command.add_argument("--profile")
            args = command.parse_args(arguments[1:])
            config = resolve(Path(args.script), config_path=args.config, profile=args.profile)
            runtime = Runtime(config)
            name = config.cli_profile or config.default_profile
            if name not in config.profiles:
                raise ConfigError("Select a configured profile for jobs")
            profile = config.profiles[name]
            runtime.capabilities.require(profile, "read_asset")
            if args.action == "list":
                print(json.dumps(runtime.capabilities.store.jobs(), indent=2))
            elif not args.operation:
                command.error("resume requires an operation ID")
            else:
                print(runtime.capabilities.resume_job(profile, args.operation).path)
            return
        if arguments[:2] == ["config", "show"]:
            command = argparse.ArgumentParser(prog="aithon config show")
            command.add_argument("--script", default="main.py")
            command.add_argument("--config")
            args = command.parse_args(arguments[2:])
            print(json.dumps(describe(resolve(Path(args.script), config_path=args.config)), indent=2, ensure_ascii=False))
            return
        args = parser().parse_args(arguments)
        path = Path(args.script)
        config = resolve(path, config_path=args.config, profile=args.profile, force_profile=args.force_profile)
        if args.explain:
            path = path.resolve()
            unit = parse(read_source(path), str(path))
            runtime = Runtime(config)
            runtime.prepare(unit, entry=True)
            print(json.dumps({"config": describe(config), "blocks": [
                {"statement": b.statement, "span": vars(b.span), "expression": b.expression, "output_type": b.output_type,
                 "plan": "requires runtime intent resolution", "cache": "unknown", "cost": "unknown",
                 "directive": vars(unit.directives.at(b.span.line))} for b in unit.blocks.values()],
                "checkpoints": [{"span": vars(c.span), "statement": c.statement}
                                for c in runtime.checkpoints.values()]}, ensure_ascii=False, indent=2))
            return
        run_script(path, args.args, config=config, stats=args.stats, trace_plan=args.trace_plan)
    except AithonError as exc:
        # Keep the original runtime cause visible without leaking provider internals.
        if exc.__cause__:
            import traceback
            traceback.print_exception(exc.__cause__)
        print(f"aithon: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
