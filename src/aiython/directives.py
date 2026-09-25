from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass

from .models import DirectiveContext, DirectiveError


def tolerant_tokens(source: str) -> list[tokenize.TokenInfo]:
    result = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            result.append(token)
    except IndentationError:
        # Lexing comments and expression boundaries does not require indentation.
        # Retry without leading whitespace, then restore original token columns.
        lines = source.splitlines(keepends=True)
        indents = [len(line) - len(line.lstrip(" \t")) for line in lines]
        result = []
        try:
            for token in tokenize.generate_tokens(io.StringIO("".join(line.lstrip(" \t") for line in lines)).readline):
                start_line, start_col = token.start
                end_line, end_col = token.end
                start_col += indents[start_line - 1] if start_line <= len(indents) else 0
                end_col += indents[end_line - 1] if end_line <= len(indents) else 0
                result.append(token._replace(start=(start_line, start_col), end=(end_line, end_col)))
        except (tokenize.TokenError, SyntaxError):
            pass
    except (tokenize.TokenError, SyntaxError):
        pass
    return result


@dataclass
class Annotation:
    line: int
    indent: int
    context: DirectiveContext
    end_line: int | None = None
    region: bool = False


class Directives:
    def __init__(self, source: str, filename: str, *, tokens=None):
        self.source = source
        self.filename = filename
        self.annotations: list[Annotation] = []
        self.boundaries: set[int] = set()
        self.bindings: list[tuple[int, int, int, DirectiveContext]] = []
        stack: list[Annotation] = []
        pending: Annotation | None = None
        tokens = tolerant_tokens(source) if tokens is None else tokens
        lines = source.splitlines()
        # Only actual COMMENT tokens count, never text inside a string.
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            line, column = token.start
            if lines[line - 1][:column].strip():
                continue
            text = token.string
            if not text.startswith("# aiython:"):
                continue
            self.boundaries.add(line)
            payload = text[len("# aiython:"):].strip()
            mode = "point"
            if payload == "end":
                if not stack or stack[-1].indent != column:
                    self.fail(line, "end must match begin at the same indentation")
                opened = stack.pop()
                opened.end_line = line - 1
                # A region cannot cross a dedent even if it later re-indents.
                if any(t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                      tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER)
                       and opened.line < t.start[0] < line and t.start[1] < column
                       for t in tokens):
                    self.fail(line, "region crosses an execution suite")
                continue
            if payload.startswith("begin ") or payload == "begin":
                mode, payload = "region", payload[5:].strip()
            values: dict[str, str] = {}
            while payload:
                match = re.match(r"(profile|prompt|capability|provider)\s*=\s*", payload)
                if not match:
                    self.fail(line, "expected profile, prompt, capability or provider with a quoted value")
                key = match[1]
                if key in values:
                    self.fail(line, f"duplicate {key}")
                try:
                    value, consumed = json.JSONDecoder().raw_decode(payload[match.end():])
                except ValueError:
                    self.fail(line, "directive values must be double-quoted strings")
                if not isinstance(value, str):
                    self.fail(line, "directive values must be strings")
                if key in ("profile", "provider", "capability") and not value.strip():
                    self.fail(line, f"{key} must not be empty")
                values[key] = value
                payload = payload[match.end() + consumed:]
                if payload and not payload[0].isspace():
                    self.fail(line, "separate directive fields with whitespace")
                payload = payload.strip()
            if mode == "point" and not values:
                self.fail(line, "empty directive")
            context = DirectiveContext(values.get("profile"),
                                       (values["prompt"],) if "prompt" in values else (),
                                       values.get("capability", "").replace("-", "_") or None, values.get("provider"))
            if mode == "region":
                annotation = Annotation(line, column, context, region=True)
                self.annotations.append(annotation)
                stack.append(annotation)
                pending = None
            else:
                between = lines[pending.line:line - 1] if pending else []
                adjacent = pending and pending.indent == column and all(
                    not text.strip() or text.lstrip().startswith("#") for text in between)
                if adjacent:
                    for key in ("profile", "provider", "capability"):
                        if getattr(pending.context, key) and getattr(context, key):
                            self.fail(line, f"duplicate {key} in statement annotation")
                    pending.context = pending.context.extend(context)
                else:
                    pending = Annotation(line, column, context)
                    self.annotations.append(pending)
        if stack:
            self.fail(stack[-1].line, "begin has no matching end")

    def fail(self, line: int, message: str):
        raise DirectiveError(f"{self.filename}:{line}: {message}")

    def bind(self, tree: ast.AST) -> None:
        if not self.annotations:
            self.bindings = []
            return
        statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
        self.bindings = []
        lines = self.source.splitlines()
        for annotation in self.annotations:
            if annotation.region:
                self.bindings.append((annotation.line, annotation.end_line or annotation.line,
                                      annotation.line, annotation.context))
                continue
            def start_line(node):
                return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
            candidates = [n for n in statements
                          if start_line(n) > annotation.line and n.col_offset == annotation.indent]
            candidates.sort(key=lambda n: (start_line(n), n.col_offset))
            if not candidates:
                self.fail(annotation.line, "annotation has no following statement")
            node = candidates[0]
            for text in lines[annotation.line:start_line(node) - 1]:
                if text.strip() and not text.lstrip().startswith("#"):
                    self.fail(annotation.line, "annotation must precede a statement in its suite")
                if text.lstrip().startswith(("# aiython: begin", "# aiython: end")):
                    self.fail(annotation.line, "statement annotation cannot cross a region boundary")
            self.bindings.append((start_line(node), node.end_lineno or node.lineno,
                                  annotation.line, annotation.context))
        self.bindings.sort(key=lambda b: b[2])

    def at(self, line: int) -> DirectiveContext:
        result = DirectiveContext()
        for start, end, declared, context in self.bindings:
            if start <= line <= end:
                result = result.extend(context)
        return result
