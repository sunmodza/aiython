from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
from functools import lru_cache
import inspect
import linecache
import math
import pickle
import re
import symtable
import sys
import threading
import types
from time import perf_counter
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Any

from .frontend import RUNTIME_NAME, Unit, runtime_binding_name
from .models import (AgentRequest, AiythonError, ConfigError, DirectiveContext,
                     RecoveryDecision, RecoveryRequest, ResolvedConfig, SourceSpan)
from .stats import Stats


# Process-local cache: never deserialize code or metadata from project files.
# Code objects are immutable; metadata is copied on restoration so runtimes do
# not share ASTs, directives, profiles, live values or recovery state.
_PREPARED = OrderedDict()
_PREPARED_LOCK = threading.RLock()
_PREPARED_LIMIT = 32
_PREPARED_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=512)
def statement_names(statement):
    return frozenset(re.findall(r"[^\W\d]\w*", statement, flags=re.UNICODE))


@lru_cache(maxsize=128)
def nearby_source(filename, base, end_line, code, line, last_line):
    lines = code.splitlines(keepends=True)
    first = base if len(code.encode()) <= 4000 else max(base, line - 12)
    last = min(end_line, last_line + 12)
    text = ''.join(lines[first - base:last - base + 1])
    limited = text[:12000]
    end = first + max(0, len(limited.splitlines()) - 1)
    return {'available': True, 'filename': filename, 'start_line': first,
            'end_line': end, 'code': limited,
            'truncated': first > base or last < end_line or len(limited) < len(text)}


@lru_cache(maxsize=64)
def focused_source(filename, base, end_line, code, line, last_line, related_names, model):
    """Select relevant lines from the bounded source window with grep-ast."""
    size = len(code.encode())
    render_nearby = nearby_source if size <= 128 * 1024 else nearby_source.__wrapped__
    nearby = render_nearby(filename, base, end_line, code, line, last_line)
    if size <= 4000 or size > 128 * 1024 or not filename.endswith('.py'):
        return nearby
    lines = nearby['code'].splitlines()
    first, last = line - nearby['start_line'], last_line - nearby['start_line']
    if first < 0 or last < first or last >= len(lines):
        return nearby
    try:
        from grep_ast import TreeContext
        view = TreeContext(filename, nearby['code'], color=False, line_number=False,
                           parent_context=True, child_context=False, last_line=False,
                           margin=0, mark_lois=False, header_max=1, loi_pad=2,
                           show_top_of_file_parent_scope=True)
        selected = set(range(first, last + 1))
        preceding = range(max(0, first - 12), first)
        for name in related_names[:8]:
            pattern = re.compile(r'\b' + re.escape(name) + r'\b')
            matches = [i for i in preceding if pattern.search(lines[i])]
            if matches:
                selected.add(matches[-1])
        view.add_lines_of_interest(selected)
        view.add_context()
    except Exception:
        # Unsupported or malformed source still has the original bounded view.
        return nearby
    visible = sorted(view.show_lines & set(range(len(lines))))
    parts = []
    previous = -1
    for index in visible:
        if index > previous + 1:
            indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
            parts.append(indent + '...\n')
        parts.append(lines[index] + '\n')
        previous = index
    compact = ''.join(parts)
    if len(compact) > 12000 or len(compact.encode()) >= len(nearby['code'].encode()):
        return nearby
    try:
        from .providers import sdk
        count = sdk().token_counter
        # The selection marker and JSON wrapper cost a few tokens too.
        if count(model=model, text=compact) + 8 >= count(model=model, text=nearby['code']):
            return nearby
    except Exception:
        return nearby
    return {'available': True, 'filename': filename,
            'start_line': nearby['start_line'] + visible[0],
            'end_line': nearby['start_line'] + visible[-1], 'code': compact,
            'truncated': nearby['truncated'] or len(visible) < len(lines),
            'selection': 'syntax'}


@lru_cache(maxsize=128)
def expression_code(source):
    # eval(str) strips leading spaces/tabs; compile(str, ..., 'eval') does not.
    return compile(source.lstrip(' \t'), '<string>', 'eval')


@lru_cache(maxsize=128)
def snippet_code(source, runtime_name=RUNTIME_NAME):
    from .typed_runtime import TypedTransformer
    tree = TypedTransformer(snippet=True, runtime_name=runtime_name).visit(ast.parse(source, '<aiython-exec>'))
    ast.fix_missing_locations(tree)
    return compile(tree, '<aiython-exec>', 'exec')


def internal_binding(name, runtime_name):
    return (name == runtime_name or name == runtime_name + 'recovery_counts'
            or name.startswith(runtime_name + 'recovery_attempt_'))


# Calling type's built-in descriptors bypasses user metaclass __getattribute__
# and descriptors shadowing __name__/__module__/__qualname__.
_TYPE_FIELDS = {name: type.__dict__[name] for name in ("__name__", "__module__", "__qualname__", "__mro__", "__dict__")}
_MODULE_SYMBOL_TABLE = getattr(getattr(symtable, "SymbolTableType", None), "MODULE", "module")
if sys.implementation.name == "cpython" and sys.version_info < (3, 13):
    import ctypes
    _LOCALS_TO_FAST = ctypes.pythonapi.PyFrame_LocalsToFast
    _LOCALS_TO_FAST.argtypes = (ctypes.py_object, ctypes.c_int)
    _LOCALS_TO_FAST.restype = None
else:
    _LOCALS_TO_FAST = None


def type_field(cls, name):
    return _TYPE_FIELDS[name].__get__(cls, type(cls))



@dataclass
class Checkpoint:
    unit: Unit
    span: SourceSpan
    statement: str
    target: str | None
    output_type: str | None = None
    retry_allowed: bool = True


class FrameNamespace(MutableMapping):
    def __init__(self, frame, global_names):
        self.frame = frame
        self._locals = frame.f_locals if _LOCALS_TO_FAST is None else None
        self.globals = frame.f_globals
        self.global_names = global_names

    @property
    def locals(self):
        return self.frame.f_locals if self._locals is None else self._locals

    def __getitem__(self, name):
        return (self.globals if name in self.global_names else self.locals)[name]

    def __setitem__(self, name, value):
        (self.globals if name in self.global_names else self.locals)[name] = value
        if name not in self.global_names and _LOCALS_TO_FAST is not None:
            _LOCALS_TO_FAST(self.frame, 0)

    def __delitem__(self, name):
        del (self.globals if name in self.global_names else self.locals)[name]
        if name not in self.global_names and _LOCALS_TO_FAST is not None:
            _LOCALS_TO_FAST(self.frame, 1)

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
        collaboration = sys.modules.get("aiython.collaboration")
        self.participant = collaboration.current() if collaboration is not None else None
        code = frame.f_code
        global_names = manager.global_names.get((code.co_filename, code.co_firstlineno, code.co_name), set())
        self._namespace = FrameNamespace(frame, global_names)

    def namespace(self):
        return self._namespace

    def eval(self, code: str) -> Any:
        from .source_guard import protect_source
        with protect_source(self.manager):
            if type(code) is str and len(code) <= 64 * 1024:
                code = expression_code(code)
            value = eval(code, self.frame.f_globals, self.namespace())
        self.manager.types.check_frame(self.frame)
        return value

    def exec(self, code: str) -> None:
        prepare = snippet_code if len(code) <= 64 * 1024 else snippet_code.__wrapped__
        occupied = {name for namespace in (self.frame.f_globals, self.frame.f_locals)
                    for name, value in namespace.items() if value is not self.manager}
        runtime_name = runtime_binding_name(code, occupied)
        compiled = prepare(code, runtime_name)
        self.frame.f_globals[runtime_name] = self.manager
        from .source_guard import protect_source
        with protect_source(self.manager):
            exec(compiled, self.frame.f_globals, self.namespace())
        self.manager.types.check_frame(self.frame)

    def get(self, name: str) -> Any:
        return self.eval(name)

    def set(self, name: str, value: Any) -> None:
        unit = self.manager.units.get(self.frame.f_code.co_filename)
        runtime_name = unit.runtime_name if unit else RUNTIME_NAME
        if not name.isidentifier() or internal_binding(name, runtime_name):
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
            render = focused_source if len(source['code']) <= 256 * 1024 else focused_source.__wrapped__
            return dict(render(source['filename'], source['start_line'], source['end_line'],
                               source['code'], span.line, span.end_line,
                               tuple(sorted(request.related_objects)), request.profile.model))
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
                unit = self.manager.units[frame.f_code.co_filename]
                runtime_name = unit.runtime_name if unit is not None else RUNTIME_NAME
                result.append({"filename": frame.f_code.co_filename, "line": tb.tb_lineno,
                               "name": frame.f_code.co_name, "active": frame is self.frame,
                               "locals": {k: self.handle(v) for k, v in frame.f_locals.items()
                                          if not internal_binding(k, runtime_name)}})
            tb = tb.tb_next
        return result


class DynamicNames(ast.NodeTransformer):
    def __init__(self, unit: Unit):
        self.table = symtable.symtable(unit.transformed, unit.filename, "exec")
        self.runtime_name = unit.runtime_name
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
        if not isinstance(node.ctx, ast.Load) or node.id == self.runtime_name or node.id == "super":
            # Keep CPython's compiler recognition of zero-argument super(),
            # which creates the implicit __class__ closure cell.
            return node
        if self.table.get_type() == _MODULE_SYMBOL_TABLE:
            return node
        try:
            symbol = self.table.lookup(node.id)
        except KeyError:
            return node
        if symbol.is_global() and not symbol.is_declared_global():
            call = ast.Call(ast.Attribute(ast.Name(self.runtime_name, ast.Load()), "lookup", ast.Load()),
                            [ast.Constant(node.id)], [])
            return ast.copy_location(call, node)
        return node


class AsyncCalls(ast.NodeTransformer):
    """Await suspended AI calls in coroutines without changing Python scheduling."""

    def __init__(self, runtime_name=RUNTIME_NAME):
        self.in_async = False
        self.runtime_name = runtime_name

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
                and node.func.value.id == self.runtime_name and node.func.attr == "execute"):
            node.func.attr = "aexecute"
            return ast.copy_location(ast.Await(value=node), node)
        return node


def install_checkpoint(runtime, unit, node, key, *, retry_allowed=True, scoped_retries=False):
    """Catch one statement without replaying statements before it."""
    runtime_name = unit.runtime_name
    target = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        target = node.targets[0].id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        target = node.target.id
    span = SourceSpan(unit.filename, node.lineno, node.col_offset,
                      node.end_lineno or node.lineno, node.end_col_offset or 0)
    statement = ast.get_source_segment(unit.source, node) or "".join(
        unit.source.splitlines(True)[span.line - 1:span.end_line]).rstrip()
    runtime.checkpoints[key] = Checkpoint(unit, span, statement, target,
        ast.unparse(node.annotation) if isinstance(node, ast.AnnAssign) else None,
        retry_allowed)
    if scoped_retries:
        # Store retry counts in the type scope, away from user bindings.
        template = ast.parse(
            "while True:\n"
            "    try:\n"
            "        pass\n"
            f"    except {runtime_name}.error_type:\n"
            f"        if {runtime_name}.recover({key!r}, {runtime_name}.current_exception()):\n"
            "            continue\n"
            "        break\n"
            "    else:\n"
            f"        {runtime_name}.clear_recovery_count({key!r})\n"
            "        break\n"
        ).body[0]
        attempt = template.body[0]
    else:
        counter = runtime_name + 'recovery_attempt_' + hashlib.sha256(key.encode()).hexdigest()[:16]
        template = ast.parse(
            "if True:\n"
            f"    {counter} = 0\n"
            "    while True:\n"
            "        try:\n"
            "            pass\n"
            f"        except {runtime_name}.error_type:\n"
            f"            {counter} += 1\n"
            f"            if {runtime_name}.recover({key!r}, {runtime_name}.current_exception(), {counter}):\n"
            "                continue\n"
            "            break\n"
            "        else:\n"
            "            break\n"
        ).body[0]
        attempt = template.body[1].body[0]
    for generated in ast.walk(template):
        if hasattr(generated, "lineno"):
            generated.lineno = node.lineno
            generated.end_lineno = node.end_lineno
            generated.col_offset = node.col_offset
            generated.end_col_offset = node.end_col_offset
    attempt.body = [node]
    return template


def iter_traceback(trace):
    while trace is not None:
        yield trace
        trace = trace.tb_next


class NestedCheckpoints(ast.NodeTransformer):
    """Repair individual nested statements while preserving Python's own handlers."""
    SIMPLE = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Delete,
              ast.Return, ast.Assert, ast.Import, ast.ImportFrom)

    def __init__(self, runtime, unit):
        self.runtime, self.unit, self.serial = runtime, unit, 0
        self.scoped_retries = False

    def generated(self, node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        func = node.value.func
        while isinstance(func, ast.Attribute):
            func = func.value
        return isinstance(func, ast.Name) and func.id == self.unit.runtime_name

    def body(self, statements, *, nested=True, preserve_docstring=False, scoped_retries=False):
        result = []
        for index, node in enumerate(statements):
            node = self.visit(node)
            if (nested and isinstance(node, self.SIMPLE) and hasattr(node, 'lineno')
                    and not (preserve_docstring and index == 0 and isinstance(node, ast.Expr)
                             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
                    and not self.generated(node)):
                self.serial += 1
                key = f'{self.unit.filename}:nested-checkpoint:{self.serial}'
                node = install_checkpoint(self.runtime, self.unit, node, key, scoped_retries=scoped_retries)
            result.append(node)
        return result

    def visit_Module(self, node):
        node.body = self.body(node.body, nested=False)
        return node

    def visit_FunctionDef(self, node):
        previous = self.scoped_retries
        self.scoped_retries = True
        node.body = self.body(node.body, preserve_docstring=True,
                              scoped_retries=True)
        self.scoped_retries = previous
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        previous = self.scoped_retries
        self.scoped_retries = True
        node.body = self.body(node.body, preserve_docstring=True, scoped_retries=True)
        self.scoped_retries = previous
        return node

    def visit_For(self, node):
        node.body = self.body(node.body, scoped_retries=self.scoped_retries)
        node.orelse = self.body(node.orelse, scoped_retries=self.scoped_retries)
        return node

    visit_AsyncFor = visit_For
    visit_While = visit_For
    visit_If = visit_For

    def visit_Match(self, node):
        for case in node.cases:
            case.body = self.body(case.body, scoped_retries=self.scoped_retries)
        return node

    def visit_Try(self, node):
        # User try/except/finally and with managers must see exceptions first.
        # TypedTransformer's outer function guard is an implementation detail.
        if getattr(node, '_aiython_module_guard', False):
            node.body = self.body(node.body, nested=False)
        elif getattr(node, '_aiython_type_guard', False) or getattr(node, '_aiython_class_guard', False):
            node.body = self.body(node.body, scoped_retries=self.scoped_retries)
        return node

    visit_TryStar = visit_Try


class Runtime:
    # Recovery handles ordinary failures; scope cleanup also sees control signals.
    error_type = Exception
    scope_error_type = BaseException

    @staticmethod
    def current_exception():
        return sys.exception()

    def __init__(self, config: ResolvedConfig, *, agent_factory=None, stats=False, trace_plan=False):
        self._lock = threading.RLock()
        self.config = config
        from .typed_runtime import TypeRuntime
        self.types = TypeRuntime(self)
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
        self.source_hints = {}
        self._source_revisions = OrderedDict()

    def source_revision(self, source):
        if len(source) > 256 * 1024:
            return hashlib.sha256(source.encode()).hexdigest()
        with self._lock:
            if source not in self._source_revisions:
                self._source_revisions[source] = hashlib.sha256(source.encode()).hexdigest()
                if len(self._source_revisions) > 128:
                    self._source_revisions.popitem(last=False)
            return self._source_revisions[source]

    def compile_source(self, source, filename, *, entry=False):
        """Reuse preparation across executions, always restoring fresh metadata."""
        from .frontend import parse
        key = (type(self), sys.implementation.cache_tag, filename, self.source_revision(source), entry)
        started = perf_counter()
        with _PREPARED_LOCK:
            cached = _PREPARED.get(key)
            if cached is not None:
                _PREPARED.move_to_end(key)
        if cached is not None:
            code, packet = cached
            unit, blocks, checkpoints, frames, nodes, names, hints = pickle.loads(packet)
            with self._lock:
                self.units[filename] = unit
                self.blocks.update(blocks)
                self.checkpoints.update(checkpoints)
                self.frame_sources.update(frames)
                self.source_nodes[filename] = nodes
                self.source_hints.update(hints)
                self.global_names.update(names)
                linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            if self.stats.enabled:
                self.stats.preparation_cache_hits += 1
                self.stats.prepare_seconds += perf_counter() - started
            return code
        unit = parse(source, filename)
        if self.stats.enabled:
            self.stats.parse_seconds += perf_counter() - started
            self.stats.preparation_cache_misses += 1
        code = self.prepare(unit, entry=entry)
        # Large generated programs run normally without displacing the cache.
        if len(source) <= 256 * 1024:
            with self._lock:
                packet = pickle.dumps((unit,
                    {k: v for k, v in self.blocks.items() if v[0].filename == filename},
                    {k: v for k, v in self.checkpoints.items() if v.unit.filename == filename},
                    {k: v for k, v in self.frame_sources.items() if k[0] == filename},
                    self.source_nodes[filename],
                    {k: v for k, v in self.global_names.items() if k[0] == filename},
                    {node: self.source_hints[node] for node in ast.walk(unit.tree)
                     if node in self.source_hints}))
            with _PREPARED_LOCK:
                _PREPARED[key] = (code, packet)
                while len(_PREPARED) > _PREPARED_LIMIT or sum(len(v[1]) for v in _PREPARED.values()) > _PREPARED_BYTES:
                    _PREPARED.popitem(last=False)
        return code

    def register(self, unit):
        self.units[unit.filename] = unit
        from .object_metadata import source_hint
        functions, classes = {}, {}
        for node in ast.walk(unit.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.source_hints[node] = source_hint(node)
                for line in {node.lineno, *(d.lineno for d in node.decorator_list)}:
                    functions.setdefault((node.name, line), []).append(node)
            elif isinstance(node, ast.ClassDef):
                self.source_hints[node] = source_hint(node)
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
        started = perf_counter()
        try:
            with self._lock:
                return self._prepare(unit, entry=entry)
        finally:
            if self.stats.enabled:
                self.stats.prepare_seconds += perf_counter() - started

    def _prepare(self, unit: Unit, *, entry: bool = False):
        self.register(unit)
        self.blocks.update({key: (unit, block) for key, block in unit.blocks.items()})
        linecache.cache[unit.filename] = (len(unit.source), None, unit.source.splitlines(True), unit.filename)
        tree = unit.tree
        from .typed_runtime import ExpectedTypes, TypedTransformer
        ExpectedTypes(unit.blocks, unit.runtime_name).visit(tree)
        tables = [symtable.symtable(unit.transformed, unit.filename, "exec")]
        definitions = {(n.name, n.lineno): min([n.lineno, *(d.lineno for d in n.decorator_list)])
                       for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        while tables:
            table = tables.pop()
            tables.extend(table.get_children())
            name, line = table.get_name(), table.get_lineno()
            line = definitions.get((name, line), line)
            self.global_names[(unit.filename, line, name)] = {
                s.get_name() for s in table.get_symbols() if s.is_declared_global()}
        if unit.blocks:
            tree = DynamicNames(unit).visit(tree)
        tree = TypedTransformer(runtime_name=unit.runtime_name).visit(tree)
        if unit.blocks and any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(tree)):
            tree = AsyncCalls(unit.runtime_name).visit(tree)
        tree = NestedCheckpoints(self, unit).visit(tree)
        ast.fix_missing_locations(tree)
        if entry:
            guard = next((node for node in tree.body if getattr(node, '_aiython_module_guard', False)), None)
            if guard is not None:
                body = []
                for index, node in enumerate(guard.body):
                    internal_check = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                      and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'checkpoint')
                    if internal_check:
                        body.append(node)
                        continue
                    key = f"{unit.filename}:checkpoint:{index}"
                    retry_allowed = not isinstance(node, (ast.For, ast.AsyncFor, ast.While,
                        ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.Match,
                        ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    body.append(install_checkpoint(self, unit, node, key, retry_allowed=retry_allowed,
                                                   scoped_retries=True))
                guard.body = body
        ast.fix_missing_locations(tree)
        return compile(tree, unit.filename, "exec", dont_inherit=True)

    def lookup(self, name: str):
        frame = inspect.currentframe().f_back
        try:
            if name in frame.f_locals:
                return frame.f_locals[name]
            if frame.f_code.co_name in ("<listcomp>", "<setcomp>", "<dictcomp>", "<genexpr>"):
                parent = frame.f_back
                while parent is not None and parent.f_code.co_filename == frame.f_code.co_filename:
                    if name in parent.f_locals:
                        return parent.f_locals[name]
                    parent = parent.f_back
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
            raise ConfigError(f"AI execution requires a configured profile (selected: {name!r}); run 'aiython setup' in the project")
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
        objects = {k: v for k, v in frame.f_locals.items()
                   if not internal_binding(k, unit.runtime_name)}
        names = statement_names(statement)
        related = {k: v for k, v in objects.items() if k in names}
        for name in names - related.keys():
            if name in frame.f_globals and not internal_binding(name, unit.runtime_name):
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
            self.types.validate_output(value, request.output_type, frame)
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
            self.types.validate_output(value, request.output_type, frame)
            return value
        finally:
            del frame

    @staticmethod
    def recoverable(error: BaseException) -> bool:
        if not isinstance(error, Exception) or isinstance(error, AiythonError):
            return False
        if isinstance(error, BaseExceptionGroup):
            return all(Runtime.recoverable(e) for e in error.exceptions)
        # concurrent.futures uses Exception for cancellation; asyncio uses BaseException.
        import concurrent.futures
        return not isinstance(error, concurrent.futures.CancelledError)

    def recover(self, key: str, error: BaseException, attempt: int | None = None) -> bool:
        if not self.recoverable(error):
            from .type_constraints import TypeViolation, UnsupportedType
            if isinstance(error, (TypeViolation, UnsupportedType)) and not getattr(error, '_aiython_location', False):
                origin = None
                trace = error.__traceback__
                while trace:
                    if trace.tb_frame.f_code.co_filename in self.units:
                        origin = (trace.tb_frame.f_code.co_filename, trace.tb_lineno)
                    trace = trace.tb_next
                if origin:
                    error.args = (f'{origin[0]}:{origin[1]}: {error}',)
                    error._aiython_location = True
            raise error
        checkpoint = self.checkpoints[key]
        frame = inspect.currentframe().f_back
        # Function/module boundaries pass their own counter. Class boundaries
        # keep theirs outside the metaclass's namespace.
        counts = None
        if attempt is None:
            from .typed_runtime import SCOPE, Scope, frame_scope
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            counts = (scope.recovery_counts if isinstance(scope, Scope) else
                      frame.f_locals.setdefault(checkpoint.unit.runtime_name + 'recovery_counts', {}))
            attempt = counts.get(key, 0) + 1
            counts[key] = attempt
        if attempt > 2:
            if counts is not None: counts.pop(key, None)
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
            retry_allowed = checkpoint.retry_allowed and not any(
                tb_frame.tb_frame is not frame and tb_frame.tb_frame.f_code.co_filename in self.units
                for tb_frame in iter_traceback(error.__traceback__))
            recovery = RecoveryRequest(**vars(request), exception=error, traceback=error.__traceback__,
                                       origin=SourceSpan(origin_unit.filename, origin_line, 0, origin_line, 0),
                                       attempt=attempt, replacement_target=checkpoint.target,
                                       retry_allowed=retry_allowed)
            decision = agent.recover(recovery, RuntimeBridge(frame, self, traceback=error.__traceback__))
            if not isinstance(decision, RecoveryDecision) or decision.action not in ("complete", "retry", "reraise"):
                raise AiythonError("Agent returned an invalid recovery decision")
            if decision.action == "retry":
                if not retry_allowed:
                    raise AiythonError("Retry would replay an enclosing statement and its side effects")
                return True
            if counts is not None: counts.pop(key, None)
            if decision.action == "reraise":
                if decision.explanation:
                    error.add_note(decision.explanation)
                raise error
            if decision.has_value:
                if checkpoint.target is None:
                    raise AiythonError("A replacement value requires a single-name assignment checkpoint")
                self.types.validate_output(decision.value, request.output_type, frame)
                frame.f_locals[checkpoint.target] = decision.value
            return False
        except AiythonError as exc:
            raise exc from error
        finally:
            del frame

    def clear_recovery_count(self, key):
        from .typed_runtime import SCOPE, frame_scope
        frame = inspect.currentframe().f_back
        try:
            (frame_scope(frame) or frame.f_locals[SCOPE]).recovery_counts.pop(key, None)
        finally:
            del frame
