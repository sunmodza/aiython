from __future__ import annotations

import ast
import inspect
import linecache
import math
import re
import symtable
import sys
import threading
import types
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Any

from .frontend import RUNTIME_NAME, Unit
from .models import (AgentRequest, AithonError, ConfigError, DirectiveContext,
                     RecoveryDecision, RecoveryRequest, ResolvedConfig, SourceSpan)
from .stats import Stats


# Calling type's built-in descriptors bypasses user metaclass __getattribute__
# and descriptors shadowing __name__/__module__/__qualname__.
_TYPE_FIELDS = {name: type.__dict__[name] for name in ("__name__", "__module__", "__qualname__", "__mro__", "__dict__")}


def type_field(cls, name):
    return _TYPE_FIELDS[name].__get__(cls, type(cls))



@dataclass
class Checkpoint:
    unit: Unit
    span: SourceSpan
    statement: str
    target: str | None
    output_type: str | None = None


class FrameNamespace(MutableMapping):
    def __init__(self, frame, global_names):
        self.locals = frame.f_locals
        self.globals = frame.f_globals
        self.global_names = global_names

    def __getitem__(self, name):
        return (self.globals if name in self.global_names else self.locals)[name]

    def __setitem__(self, name, value):
        (self.globals if name in self.global_names else self.locals)[name] = value

    def __delitem__(self, name):
        del (self.globals if name in self.global_names else self.locals)[name]

    def __iter__(self):
        return iter(dict.fromkeys([*self.locals, *(n for n in self.global_names if n in self.globals)]))

    def __len__(self):
        return sum(1 for _ in self)


class RuntimeBridge:
    """One live frame, with handles that preserve Python object identity."""

    def __init__(self, frame: types.FrameType, manager: Runtime, *, traceback=None):
        self.frame = frame
        self.manager = manager
        self.handles: dict[str, Any] = {}
        self._identities: dict[int, str] = {}
        self.traceback = traceback
        collaboration = sys.modules.get("aithon.collaboration")
        self.participant = collaboration.current() if collaboration is not None else None
        code = frame.f_code
        global_names = manager.global_names.get((code.co_filename, code.co_firstlineno, code.co_name), set())
        self._namespace = FrameNamespace(frame, global_names)

    def namespace(self):
        return self._namespace

    def eval(self, code: str) -> Any:
        from .source_guard import protect_source
        with protect_source(self.manager):
            value = eval(code, self.frame.f_globals, self.namespace())
        self.manager.types.check_frame(self.frame)
        return value

    def exec(self, code: str) -> None:
        from .typed_runtime import TypedTransformer
        tree = TypedTransformer(snippet=True).visit(ast.parse(code, '<aithon-exec>'))
        ast.fix_missing_locations(tree)
        self.namespace()[RUNTIME_NAME] = self.manager
        from .source_guard import protect_source
        with protect_source(self.manager):
            exec(compile(tree, '<aithon-exec>', 'exec'), self.frame.f_globals, self.namespace())
        self.manager.types.check_frame(self.frame)

    def get(self, name: str) -> Any:
        return self.eval(name)

    def set(self, name: str, value: Any) -> None:
        if not name.isidentifier() or name.startswith("__aithon_"):
            raise ValueError("Binding must be a non-reserved Python identifier")
        self.manager.types.assignment_in(self.frame, value, name)
        self.namespace()[name] = value

    def handle(self, value: Any, *, include_value: bool = True) -> dict:
        identity = id(value)
        key = self._identities.get(identity)
        if key is None:
            key = f"object-{len(self.handles) + 1}"
            self.handles[key] = value  # Strong reference prevents id reuse.
            self._identities[identity] = key
        cls = type(value)
        result = {"handle": key}
        for public, internal in (("type", "__name__"), ("module", "__module__"), ("qualname", "__qualname__")):
            metadata = type_field(cls, internal)
            result[public] = metadata if type(metadata) is str else None
        if include_value and type(value) in (str, int, float, bool, type(None)) and not (
                type(value) is float and not math.isfinite(value)):
            result["value"] = value if not isinstance(value, str) else value[:4000]
        return result

    def describe_handles(self, objects: dict[str, Any]) -> dict:
        from .object_metadata import structural_hint
        result = {}
        scopes = self.manager.types.scopes(self.frame)
        for name, value in objects.items():
            if name.startswith('__'):
                continue
            metadata = self.handle(value, include_value=False)
            metadata.pop('module', None)
            metadata.pop('qualname', None)
            if isinstance(value, type):
                metadata['type'] = 'class'
            metadata.update(structural_hint(value, self.manager, type_field))
            for scope, bindings in scopes:
                if name in bindings and bindings[name] is value:
                    annotation = scope.declarations.get(name)
                    if type(annotation) is str:
                        metadata['annotation'] = annotation[:300]
                    break
            result[name] = metadata
        return result

    def _source_frame(self, selector):
        if selector == "active":
            return self.frame
        if selector != "origin":
            raise ValueError("frame must be active or origin")
        frame = None
        tb = self.traceback
        while tb:
            if tb.tb_frame.f_code.co_filename in self.manager.units:
                frame = tb.tb_frame
            tb = tb.tb_next
        return frame

    def get_frame_code(self, frame="active") -> dict:
        target = self._source_frame(frame)
        if target is None:
            return {"available": False, "frame": frame}
        return self.manager.frame_source(target)

    def code_context(self, request) -> dict:
        def nearby(source, span):
            if not source.get("available"):
                return source
            lines = source["code"].splitlines(keepends=True)
            base = source["start_line"]
            # Short frames fit cheaply in the prompt. Include preceding setup
            # and definitions so the model need not spend a round fetching them.
            first = base if len(source["code"].encode()) <= 4000 else max(base, span.line - 12)
            last = min(source["end_line"], span.end_line + 12)
            text = "".join(lines[first - base:last - base + 1])
            limited = text[:12000]
            end = first + max(0, len(limited.splitlines()) - 1)
            return {"available": True, "filename": source["filename"],
                    "start_line": first, "end_line": end, "code": limited,
                    "truncated": first > base or last < source["end_line"] or len(limited) < len(text)}
        source = self.get_frame_code()
        # Custom agents/tests may provide a source string without registering a unit.
        if not source.get("available"):
            source = {"available": True, "filename": request.span.filename, "start_line": 1,
                      "end_line": max(1, len(request.frame_code.splitlines())), "code": request.frame_code}
        result = {"active": nearby(source, request.span)}
        if isinstance(request, RecoveryRequest):
            result["origin"] = nearby(self.get_frame_code("origin"), request.origin)
        return result

    def dereference(self, handle: str) -> Any:
        if handle not in self.handles:
            raise ValueError(f"Unknown object handle: {handle}")
        return self.handles[handle]

    def inspect(self, handle: str, depth: int = 1, limit: int = 20) -> dict:
        from itertools import islice
        if type(depth) is not int or not 1 <= depth <= 3 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("inspect requires depth 1..3 and limit 1..100")
        remaining = 200
        expanded = set()

        def describe(value, levels):
            nonlocal remaining
            result = self.handle(value)
            remaining -= 1
            if levels and type(value) in (dict, list, tuple) and result["handle"] not in expanded and remaining > 0:
                result.update(contents(value, levels))
            return result

        def contents(value, levels):
            key = self.handle(value)["handle"]
            expanded.add(key)
            result = {"type": type_field(type(value), "__name__")}
            if type(value) is dict:
                entries = []
                for k, v in islice(value.items(), limit):
                    if remaining < 2:
                        break
                    entries.append({"key": describe(k, 0), "value": describe(v, levels - 1)})
                result.update(entries=entries, truncated=len(entries) < len(value))
            elif type(value) in (list, tuple):
                items = []
                for item in islice(value, limit):
                    if remaining <= 0:
                        break
                    items.append(describe(item, levels - 1))
                result.update(items=items, truncated=len(items) < len(value))
            else:
                # Bypass user descriptors, metaclass hooks and __dir__.
                attributes = {}
                for cls in reversed(type_field(type(value), "__mro__")):
                    attributes.update(type_field(cls, "__dict__"))
                descriptor = attributes.get("__dict__")
                if type(descriptor) is types.GetSetDescriptorType:
                    instance_attributes = descriptor.__get__(value, type(value))
                    if type(instance_attributes) is dict:
                        attributes.update(instance_attributes)
                result["attributes"] = {k: describe(v, levels - 1)
                                        for k, v in islice(attributes.items(), min(limit, remaining))}
                result["truncated"] = len(result["attributes"]) < len(attributes)
            return result

        value = self.dereference(handle)
        if type(value) in (str, int, float, bool, type(None)):
            return self.handle(value)
        return contents(value, depth)

    def frames(self) -> list[dict]:
        result = []
        tb = self.traceback
        while tb:
            frame = tb.tb_frame
            if frame.f_code.co_filename in self.manager.units:
                result.append({"filename": frame.f_code.co_filename, "line": tb.tb_lineno,
                               "name": frame.f_code.co_name, "active": frame is self.frame,
                               "locals": {k: self.handle(v) for k, v in frame.f_locals.items()
                                          if not k.startswith("__aithon_")}})
            tb = tb.tb_next
        return result


class DynamicNames(ast.NodeTransformer):
    def __init__(self, unit: Unit):
        self.table = symtable.symtable(unit.transformed, unit.filename, "exec")
        self.used_tables: set[int] = set()

    def in_scope(self, name: str, line: int, body):
        parent = self.table
        child = next((c for c in parent.get_children()
                      if c.get_name() == name and c.get_lineno() == line and c.get_id() not in self.used_tables), None)
        if child is None:
            return [self.visit(n) for n in body]
        self.used_tables.add(child.get_id())
        self.table = child
        result = [self.visit(n) for n in body]
        self.table = parent
        return result

    def visit_FunctionDef(self, node):
        node.decorator_list = [self.visit(n) for n in node.decorator_list]
        node.args.defaults = [self.visit(n) for n in node.args.defaults]
        node.args.kw_defaults = [self.visit(n) if n else n for n in node.args.kw_defaults]
        node.body = self.in_scope(node.name, node.lineno, node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(n) for n in node.decorator_list]
        node.bases = [self.visit(n) for n in node.bases]
        node.keywords = [self.visit(n) for n in node.keywords]
        node.body = self.in_scope(node.name, node.lineno, node.body)
        return node

    def visit_Lambda(self, node):
        node.args.defaults = [self.visit(n) for n in node.args.defaults]
        node.args.kw_defaults = [self.visit(n) if n else n for n in node.args.kw_defaults]
        node.body = self.in_scope("lambda", node.lineno, [node.body])[0]
        return node

    def visit_ListComp(self, node):
        return self.comprehension(node, "listcomp")

    def visit_SetComp(self, node):
        return self.comprehension(node, "setcomp")

    def visit_DictComp(self, node):
        return self.comprehension(node, "dictcomp")

    def visit_GeneratorExp(self, node):
        return self.comprehension(node, "genexpr")

    def comprehension(self, node, name):
        first = node.generators[0]
        first.iter = self.visit(first.iter)
        outer_iter = first.iter
        first.iter = ast.Constant(None)
        # The first iterable is evaluated in the enclosing scope.
        parent = self.table
        child = next((c for c in parent.get_children() if c.get_name() == name and c.get_lineno() == node.lineno), None)
        if child:
            self.table = child
        node = self.generic_visit(node)
        self.table = parent
        first.iter = outer_iter
        return node

    def visit_AnnAssign(self, node):
        node.target = self.visit(node.target)
        if node.value: node.value = self.visit(node.value)
        return node

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load) or node.id.startswith("__aithon_") or node.id == "super":
            # Keep CPython's compiler recognition of zero-argument super(),
            # which creates the implicit __class__ closure cell.
            return node
        if self.table.get_type() == symtable.SymbolTableType.MODULE:
            return node
        try:
            symbol = self.table.lookup(node.id)
        except KeyError:
            return node
        if symbol.is_global() and not symbol.is_declared_global():
            call = ast.Call(ast.Attribute(ast.Name(RUNTIME_NAME, ast.Load()), "lookup", ast.Load()),
                            [ast.Constant(node.id)], [])
            return ast.copy_location(call, node)
        return node


class AsyncCalls(ast.NodeTransformer):
    """Await suspended AI calls in coroutines without changing Python scheduling."""

    def __init__(self):
        self.in_async = False

    def _body(self, node, active):
        previous = self.in_async
        self.in_async = active
        node.body = [self.visit(item) for item in node.body]
        self.in_async = previous
        return node

    def visit_AsyncFunctionDef(self, node):
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [self.visit(item) if item else None for item in node.args.kw_defaults]
        return self._body(node, True)

    def visit_FunctionDef(self, node):
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [self.visit(item) if item else None for item in node.args.kw_defaults]
        return self._body(node, False)

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        return self._body(node, False)

    def visit_Lambda(self, node):
        previous = self.in_async
        self.in_async = False
        node.body = self.visit(node.body)
        self.in_async = previous
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (self.in_async and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == RUNTIME_NAME and node.func.attr == "execute"):
            node.func.attr = "aexecute"
            return ast.copy_location(ast.Await(value=node), node)
        return node


class Runtime:
    error_type = BaseException

    def __init__(self, config: ResolvedConfig, *, agent_factory=None, stats=False, trace_plan=False):
        self._lock = threading.RLock()
        self.config = config
        from .typed_runtime import TypeRuntime
        self.types = TypeRuntime()
        from .capabilities import CapabilityRuntime
        self.capabilities = CapabilityRuntime(config, trace=trace_plan)
        self.agent_factory = agent_factory
        self.units: dict[str, Unit] = {}
        self.blocks = {}
        self.checkpoints: dict[str, Checkpoint] = {}
        self.agents = {}
        self.global_names = {}
        self.stats = Stats(stats)
        self.frame_sources = {}
        self.source_nodes = {}

    def register(self, unit):
        self.units[unit.filename] = unit
        functions, classes = {}, {}
        for node in ast.walk(unit.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for line in {node.lineno, *(d.lineno for d in node.decorator_list)}:
                    functions.setdefault((node.name, line), []).append(node)
            elif isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append(node)
        self.source_nodes[unit.filename] = (functions, classes)
        lines = unit.source.splitlines(keepends=True)
        self.frame_sources[(unit.filename, 1, "<module>")] = {
            "available": True, "filename": unit.filename, "start_line": 1,
            "end_line": max(1, len(lines)), "code": unit.source}
        for node in ast.walk(unit.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            first = min([node.lineno, *(n.lineno for n in getattr(node, "decorator_list", []))])
            name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
            self.frame_sources[(unit.filename, first, name)] = {
                "available": True, "filename": unit.filename, "start_line": first,
                "end_line": node.end_lineno, "code": "".join(lines[first - 1:node.end_lineno])}

    def frame_source(self, frame):
        code = frame.f_code
        key = (code.co_filename, code.co_firstlineno, code.co_name)
        return self.frame_sources.get(key, self.frame_sources.get(
            (code.co_filename, 1, "<module>"), {"available": False, "filename": code.co_filename}))

    def prepare(self, unit: Unit, *, entry: bool = False):
        with self._lock:
            return self._prepare(unit, entry=entry)

    def _prepare(self, unit: Unit, *, entry: bool = False):
        self.register(unit)
        self.blocks.update({key: (unit, block) for key, block in unit.blocks.items()})
        linecache.cache[unit.filename] = (len(unit.source), None, unit.source.splitlines(True), unit.filename)
        tree = unit.tree
        from .typed_runtime import ExpectedTypes, TypedTransformer
        ExpectedTypes(unit.blocks).visit(tree)
        tables = [symtable.symtable(unit.transformed, unit.filename, "exec")]
        definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        while tables:
            table = tables.pop()
            tables.extend(table.get_children())
            name, line = table.get_name(), table.get_lineno()
            for definition in definitions:
                if definition.name == name and definition.lineno == line and definition.decorator_list:
                    line = min(d.lineno for d in definition.decorator_list)
                    break
            self.global_names[(unit.filename, line, name)] = {
                s.get_name() for s in table.get_symbols() if s.is_declared_global()}
        if unit.blocks:
            tree = DynamicNames(unit).visit(tree)
        tree = TypedTransformer().visit(tree)
        if unit.blocks and any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(tree)):
            tree = AsyncCalls().visit(tree)
        ast.fix_missing_locations(tree)
        if entry:
            body = []
            for index, node in enumerate(tree.body):
                from .typed_runtime import SCOPE
                internal_scope = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == SCOPE for t in node.targets)
                internal_check = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                  and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'checkpoint')
                if internal_scope or internal_check:
                    body.append(node)
                    continue
                if (isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
                    index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                    body.append(node)
                    continue
                key = f"{unit.filename}:checkpoint:{index}"
                target = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target = node.targets[0].id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                    target = node.target.id
                span = SourceSpan(unit.filename, node.lineno, node.col_offset,
                                  node.end_lineno or node.lineno, node.end_col_offset or 0)
                statement = ast.get_source_segment(unit.source, node) or "".join(
                    unit.source.splitlines(True)[span.line - 1:span.end_line]).rstrip()
                self.checkpoints[key] = Checkpoint(unit, span, statement, target, ast.unparse(node.annotation) if isinstance(node, ast.AnnAssign) else None)
                # A loop permits explicit retry without replaying earlier checkpoints.
                template = ast.parse(
                    "while True:\n"
                    "    try:\n"
                    "        pass\n"
                    f"    except {RUNTIME_NAME}.error_type as __aithon_error__:\n"
                    f"        if {RUNTIME_NAME}.recover({key!r}, __aithon_error__):\n"
                    "            continue\n"
                    "        break\n"
                    "    else:\n"
                    "        break\n"
                ).body[0]
                for generated in ast.walk(template):
                    if hasattr(generated, "lineno"):
                        generated.lineno = node.lineno
                        generated.end_lineno = node.end_lineno
                        generated.col_offset = node.col_offset
                        generated.end_col_offset = node.end_col_offset
                template.body[0].body = [node]
                body.append(template)
            tree.body = body
        ast.fix_missing_locations(tree)
        return compile(tree, unit.filename, "exec", dont_inherit=True)

    def lookup(self, name: str):
        frame = inspect.currentframe().f_back
        try:
            if name in frame.f_locals:
                return frame.f_locals[name]
            if name in frame.f_globals:
                return frame.f_globals[name]
            if name in frame.f_builtins:
                return frame.f_builtins[name]
            raise NameError(f"name {name!r} is not defined")
        finally:
            del frame

    def selection(self, context: DirectiveContext):
        name = self.config.force_profile or context.profile or self.config.cli_profile or self.config.default_profile
        if name is None or name not in self.config.profiles:
            raise ConfigError(f"AI execution requires a configured profile (selected: {name!r}); run 'aithon setup' in the project")
        profile = self.config.profiles[name]
        with self._lock:
            if name not in self.agents:
                if self.agent_factory:
                    self.agents[name] = self.agent_factory(profile)
                else:
                    from .agent import ToolAgent
                    from .providers import load_provider
                    self.agents[name] = ToolAgent(load_provider(self.config, profile))
        prompts = ((profile.prompt,) if profile.prompt else ()) + context.prompts
        return profile, prompts, self.agents[name]

    def request(self, statement, span, unit, frame, context):
        profile, prompts, agent = self.selection(context)
        objects = {k: v for k, v in frame.f_locals.items() if not k.startswith("__aithon_")}
        names = set(re.findall(r"[^\W\d]\w*", statement, flags=re.UNICODE))
        related = {k: v for k, v in objects.items() if k in names}
        for name in names - related.keys():
            if name in frame.f_globals and not name.startswith("__aithon_"):
                related[name] = frame.f_globals[name]
        frame_code = self.frame_source(frame).get("code", unit.source)
        return AgentRequest(statement, frame_code, related,
                            {k: v for k, v in objects.items() if k not in related},
                            span, profile, prompts, capability=context.capability, provider=context.provider), agent

    def execute(self, key: str):
        unit, block = self.blocks[key]
        frame = inspect.currentframe().f_back
        try:
            request, agent = self.request(block.statement, block.span, unit, frame,
                                          unit.directives.at(block.span.line))
            request.output_type = block.output_type
            request.requires_result = block.expression
            bridge = RuntimeBridge(frame, self)
            value = agent.execute(request, bridge)
            from .type_constraints import validate_output
            validate_output(value, request.output_type, self.types.namespace(frame))
            return value
        finally:
            del frame

    async def aexecute(self, key: str):
        unit, block = self.blocks[key]
        frame = inspect.currentframe().f_back
        try:
            request, agent = self.request(block.statement, block.span, unit, frame,
                                          unit.directives.at(block.span.line))
            request.output_type = block.output_type
            request.requires_result = block.expression
            bridge = RuntimeBridge(frame, self)
            if callable(getattr(agent, "aexecute", None)):
                value = await agent.aexecute(request, bridge)
            else:
                value = agent.execute(request, bridge)
            from .type_constraints import validate_output
            validate_output(value, request.output_type, self.types.namespace(frame))
            return value
        finally:
            del frame

    @staticmethod
    def recoverable(error: BaseException) -> bool:
        if not isinstance(error, Exception) or isinstance(error, AithonError):
            return False
        if isinstance(error, BaseExceptionGroup):
            return all(Runtime.recoverable(e) for e in error.exceptions)
        # concurrent.futures uses Exception for cancellation; asyncio uses BaseException.
        import concurrent.futures
        return not isinstance(error, concurrent.futures.CancelledError)

    def recover(self, key: str, error: BaseException) -> bool:
        if not self.recoverable(error):
            from .type_constraints import TypeViolation, UnsupportedType
            if isinstance(error, (TypeViolation, UnsupportedType)) and not getattr(error, '_aithon_location', False):
                origin = None
                trace = error.__traceback__
                while trace:
                    if trace.tb_frame.f_code.co_filename in self.units:
                        origin = (trace.tb_frame.f_code.co_filename, trace.tb_lineno)
                    trace = trace.tb_next
                if origin:
                    error.args = (f'{origin[0]}:{origin[1]}: {error}',)
                    error._aithon_location = True
            raise error
        checkpoint = self.checkpoints[key]
        frame = inspect.currentframe().f_back
        # State belongs to this entry frame, never to a process-global runtime.
        counts = frame.f_locals.setdefault("__aithon_recovery_counts__", {})
        attempt = counts.get(key, 0) + 1
        counts[key] = attempt
        if attempt > 2:
            counts.pop(key, None)
            raise error
        origin_unit, origin_line = checkpoint.unit, checkpoint.span.line
        tb = error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename in self.units:
                origin_unit = self.units[tb.tb_frame.f_code.co_filename]
                origin_line = tb.tb_lineno
            tb = tb.tb_next
        try:
            request, agent = self.request(checkpoint.statement, checkpoint.span, checkpoint.unit,
                                          frame, origin_unit.directives.at(origin_line))
            request.output_type = checkpoint.output_type
            recovery = RecoveryRequest(**vars(request), exception=error, traceback=error.__traceback__,
                                       origin=SourceSpan(origin_unit.filename, origin_line, 0, origin_line, 0),
                                       attempt=attempt, replacement_target=checkpoint.target)
            decision = agent.recover(recovery, RuntimeBridge(frame, self, traceback=error.__traceback__))
            if not isinstance(decision, RecoveryDecision) or decision.action not in ("complete", "retry", "reraise"):
                raise AithonError("Agent returned an invalid recovery decision")
            if decision.action == "retry":
                return True
            counts.pop(key, None)
            if decision.action == "reraise":
                if decision.explanation:
                    error.add_note(decision.explanation)
                raise error
            if decision.has_value:
                if checkpoint.target is None:
                    raise AithonError("A replacement value requires a single-name assignment checkpoint")
                from .type_constraints import validate_output
                validate_output(decision.value, request.output_type, self.types.namespace(frame))
                frame.f_locals[checkpoint.target] = decision.value
            return False
        except AithonError as exc:
            raise exc from error
        finally:
            del frame
