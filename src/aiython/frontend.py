from __future__ import annotations

import ast
import keyword
import re
import sys
import tokenize
from dataclasses import dataclass

from .directives import Directives, tolerant_tokens
from .models import SourceSpan

RUNTIME_NAME = "__aiython_runtime__"


@dataclass
class Block:
    id: str
    start: int
    end: int
    statement: str
    span: SourceSpan
    expression: bool = True
    output_type: str | None = None


@dataclass
class Unit:
    source: str
    filename: str
    tree: ast.Module
    directives: Directives
    blocks: dict[str, Block]
    transformed: str


class Frontend:
    """Compile-guided, token-boundary recovery with enclosing-statement fallback.

    CPython is the grammar oracle. Candidate edits never execute source. Every
    accepted edit removes an error or advances to a later source position.
    """

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.lines = source.splitlines(keepends=True)
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.tokens = tolerant_tokens(source)
        self.directives = Directives(source, filename, tokens=self.tokens)
        self.serial = 0

    def offset(self, line: int, column: int) -> int:
        return min(len(self.source), self.offsets[min(max(line - 1, 0), len(self.offsets) - 1)] + column)

    def position(self, offset: int) -> tuple[int, int]:
        import bisect
        line = min(bisect.bisect_right(self.offsets, offset), max(len(self.lines), 1))
        return line, offset - self.offsets[line - 1]

    def render(self, blocks: list[Block]) -> tuple[str, list[int]]:
        pieces, mapping = [], []
        cursor = 0
        for block in sorted(blocks, key=lambda b: b.start):
            pieces.append(self.source[cursor:block.start])
            mapping.extend(range(cursor, block.start))
            call = f"{RUNTIME_NAME}.execute({block.id!r})"
            count = self.source[block.start:block.end].count("\n")
            replacement = "(" + call + "\n" * count + ")" if count else call
            pieces.append(replacement)
            mapping.extend([block.start] * len(replacement))
            cursor = block.end
        pieces.append(self.source[cursor:])
        mapping.extend(range(cursor, len(self.source)))
        mapping.append(len(self.source))
        return "".join(pieces), mapping

    def check(self, text: str) -> SyntaxError | None:
        try:
            # A real filename lets CPython consult the *original* file while
            # formatting error offsets for our longer generated call text.
            # Use an in-memory filename until transformation is complete.
            compile(text, "<aiython-parser>", "exec", dont_inherit=True)
        except SyntaxError as exc:
            return exc
        return None

    @staticmethod
    def error_offset(text: str, error: SyntaxError) -> int:
        lines = text.splitlines(keepends=True)
        return sum(map(len, lines[:max(0, (error.lineno or 1) - 1)])) + max(0, (error.offset or 1) - 1)

    def block(self, start: int, end: int, expression: bool = True) -> Block:
        self.serial += 1
        line, column = self.position(start)
        end_line, end_column = self.position(end)
        return Block(f"{self.filename}:{self.serial}", start, end, self.source[start:end],
                     SourceSpan(self.filename, line, column, end_line, end_column), expression)

    def candidates(self, position: int) -> list[tuple[int, int, bool]]:
        line, _ = self.position(position)
        tokens = self.tokens
        ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
                   tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
        useful = [t for t in tokens if t.type not in ignored and t.string.strip()]
        # The logical statement containing the error may span several lines.
        lower, upper = 0, len(self.source)
        for t in tokens:
            if t.type == tokenize.NEWLINE:
                at = self.offset(*t.end)
                if at <= position:
                    lower = at
                else:
                    upper = at
                    break
        local = [t for t in useful if lower <= self.offset(*t.start) < upper]
        candidates = []
        # In invalid natural-language headers, a trailing ':' introduces an
        # immediately following bullet list. Valid Python files never enter
        # recovery, and a blank line ends this explicit continuation.
        header = self.lines[line - 1].rstrip()
        indent = len(header) - len(header.lstrip(" \t"))
        last = line
        if header.endswith(":"):
            while last < len(self.lines):
                following = self.lines[last]
                following_indent = len(following) - len(following.lstrip(" \t"))
                if following_indent != indent or not following.lstrip().startswith("- "):
                    break
                last += 1
        preferred = []
        # Keep the assignment boundary even when a large NL dict/schema exceeds
        # the bounded expression search. This is grammar checked, not regex-based.
        for token in local:
            if token.type != tokenize.OP or token.string != "=":
                continue
            first = self.offset(*local[0].start)
            eq_end = self.offset(*token.end)
            try:
                prefix = ast.parse(self.source[first:eq_end].lstrip() + " None")
            except SyntaxError:
                continue
            if len(prefix.body) != 1 or not isinstance(prefix.body[0], (ast.Assign, ast.AnnAssign)):
                continue
            rhs = next((t for t in local if self.offset(*t.start) >= eq_end), None)
            rhs_tokens = [t for t in local if self.offset(*t.start) >= eq_end]
            depth = 0
            for index, item in enumerate(rhs_tokens):
                if item.type == tokenize.OP:
                    if item.string in ('(', '[', '{'): depth += 1
                    elif item.string in (')', ']', '}'): depth -= 1
                    elif item.string == ';' and depth == 0:
                        rhs_tokens = rhs_tokens[:index]
                        break
            natural_prefix = (len(rhs_tokens) > 1 and all(t.type == tokenize.NAME and not keyword.iskeyword(t.string) for t in rhs_tokens[:2]))
            if rhs is not None and (natural_prefix or len(local) > 100 or last > line or local[-1].end[0] > local[0].start[0]):
                end = (self.offset(last, len(self.lines[last - 1].rstrip("\r\n")))
                       if last > line else self.offset(*rhs_tokens[-1].end))
                preferred.append((self.offset(*rhs.start), end, True))
            break
        if local and local[0].string in ('return', 'yield') and len(local) > 2:
            if all(t.type == tokenize.NAME and not keyword.iskeyword(t.string) for t in local[1:3]):
                preferred.append((self.offset(*local[1].start), self.offset(*local[-1].end), True))
        if last > line and not preferred:
            preferred.append((self.offset(line, indent), self.offset(last, len(self.lines[last - 1].rstrip("\r\n"))), False))
        # Bounded search avoids quadratic work on huge statements. Fallback is
        # always available, including for grammar slots that cannot hold calls.
        if len(local) <= 100:
            starts = sorted({self.offset(*t.start) for t in local})
            ends = sorted({self.offset(*t.end) for t in local})
            for start in starts:
                for end in ends:
                    if start < end and start <= position < end:
                        delimiters = []
                        balanced = True
                        for token in local:
                            if not (start <= self.offset(*token.start) < end) or token.type != tokenize.OP:
                                continue
                            if token.string in ("(", "[", "{"):
                                delimiters.append(token.string)
                            elif token.string in (")", "]", "}"):
                                if not delimiters or delimiters.pop() != {")": "(", "]": "[", "}": "{"}[token.string]:
                                    balanced = False
                                    break
                        if not balanced or delimiters:
                            continue
                        candidates.append((start, end, True))
        text = self.lines[line - 1]
        start = self.offset(line, len(text) - len(text.lstrip(" \t")))
        end = self.offset(line, len(text.rstrip("\r\n")))
        if start < end:
            candidates.append((start, end, False))
        candidates.sort(key=lambda c: (c[1] - c[0], c[0]))
        # Grow to enclosing suites, including decorators attached to definitions.
        for first in range(line - 1, -1, -1):
            text = self.lines[first]
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            indent = len(text) - len(text.lstrip(" \t"))
            last = max(line, first + 1)
            while last < len(self.lines):
                next_line = self.lines[last]
                next_indent = len(next_line) - len(next_line.lstrip(" \t"))
                stripped = next_line.lstrip()
                if stripped.strip() and not stripped.startswith("#") and next_indent <= indent:
                    if next_indent == indent and stripped.startswith(("else:", "elif ", "except", "finally:")):
                        last += 1
                        continue
                    break
                last += 1
            start = self.offset(first + 1, indent)
            end = self.offset(last, len(self.lines[last - 1].rstrip("\r\n")))
            if start <= position <= end:
                candidates.append((start, end, False))
        candidates.append((0, len(self.source.rstrip("\r\n")), False))
        return list(dict.fromkeys(preferred + candidates))

    def legacy_fstring_candidates(self):
        # Before Python 3.12, tokenize exposes an entire f-string as one
        # STRING token. Recover simple invalid replacement fields from that
        # token, then let CPython validate the transformed expression.
        for token in tolerant_tokens(self.source):
            if token.type != tokenize.STRING or not re.match(r"(?i)^[rubf]*f[rubf]*['\"]", token.string):
                continue
            for match in re.finditer(r"(?<!\{)\{([^{}]+)\}(?!\})", token.string):
                expression = re.split(r"[!:]", match.group(1), maxsplit=1)[0]
                try:
                    ast.parse(expression, mode="eval")
                except SyntaxError:
                    start = self.offset(*token.start) + match.start(1)
                    yield start, start + len(match.group(1))

    def build(self) -> Unit:
        blocks: list[Block] = []
        while True:
            rendered, mapping = self.render(blocks)
            error = self.check(rendered)
            if error is None:
                break
            if sys.version_info < (3, 12) and error.msg.startswith("f-string:"):
                replacement = next(((start, end) for start, end in self.legacy_fstring_candidates()
                                    if not any(start < block.end and block.start < end for block in blocks)), None)
                if replacement is not None:
                    blocks.append(self.block(*replacement))
                    continue
            index = min(self.error_offset(rendered, error), len(mapping) - 1)
            position = mapping[index]
            chosen = None
            for start, end, expression in self.candidates(position):
                if end <= start:
                    continue
                # Never cut an already selected edit in half.
                if any(start < b.end and b.start < end and not (start <= b.start and b.end <= end)
                       for b in blocks):
                    continue
                candidate = self.block(start, end, expression)
                remaining = [b for b in blocks if not (start <= b.start and b.end <= end)]
                proposed = remaining + [candidate]
                text, source_map = self.render(proposed)
                # A syntactically valid edit inside string text is not a runtime
                # call (notably when deleting an f-string brace). Reject it.
                real_calls = sum(t.type == tokenize.NAME and t.string == RUNTIME_NAME
                                 for t in tolerant_tokens(text))
                if sys.version_info < (3, 12):
                    legacy_fields = set(self.legacy_fstring_candidates())
                    real_calls += sum((block.start, block.end) in legacy_fields for block in proposed)
                if real_calls < len(proposed):
                    continue
                next_error = self.check(text)
                if next_error is None:
                    chosen = proposed
                    break
                next_index = min(self.error_offset(text, next_error), len(source_map) - 1)
                next_position = source_map[next_index]
                gap = self.source[end:next_position]
                separated = (self.position(next_position)[0] > self.position(position)[0]
                             or re.search(r"[,;:]|\b(?:if|else|for)\b", gap))
                if next_position >= end and next_position > position and separated:
                    chosen = proposed
                    break
            if chosen is None:
                # The whole source can always be represented by a call.
                raise error
            blocks = chosen
        if not blocks:
            tree = ast.parse(self.source, self.filename)
            self.directives.bind(tree)
            return Unit(self.source, self.filename, tree, self.directives, {}, self.source)
        # Mark standalone calls, then combine consecutive invalid statements.
        tree = ast.parse(rendered, self.filename)
        standalone = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and self.is_call(node.value):
                standalone[node.value.args[0].value] = node
        for block in blocks:
            block.expression = block.id not in standalone
        blocks.sort(key=lambda b: b.start)
        merged: list[Block] = []
        for block in blocks:
            previous = merged[-1] if merged else None
            if previous and not previous.expression and not block.expression:
                gap = self.source[previous.end:block.start]
                has_directive = any(previous.span.end_line < n <= block.span.line
                                    for n in self.directives.boundaries)
                trivia = all(not line.strip() or line.lstrip().startswith("#") for line in gap.splitlines())
                if trivia and not has_directive and previous.span.column == block.span.column:
                    merged[-1] = self.block(previous.start, block.end, False)
                    continue
            merged.append(block)
        rendered, mapping = self.render(merged)
        tree = ast.parse(rendered, self.filename)
        generated_lines = rendered.splitlines(keepends=True)
        generated_offsets = [0]
        for text in generated_lines:
            generated_offsets.append(generated_offsets[-1] + len(text))
        for node in ast.walk(tree):
            for line_attr, col_attr in (("lineno", "col_offset"), ("end_lineno", "end_col_offset")):
                line = getattr(node, line_attr, None)
                column = getattr(node, col_attr, None)
                if line is None or column is None:
                    continue
                # AST columns count UTF-8 bytes; tokenize columns count characters.
                prefix = generated_lines[line - 1].encode("utf-8")[:column].decode("utf-8", errors="ignore")
                offset = generated_offsets[line - 1] + len(prefix)
                original_line, original_col = self.position(mapping[min(offset, len(mapping) - 1)])
                setattr(node, line_attr, original_line)
                setattr(node, col_attr, len(self.lines[original_line - 1][:original_col].encode("utf-8")))
        by_id = {b.id: b for b in merged}
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and self.is_call(node.value):
                by_id[node.value.args[0].value].output_type = ast.unparse(node.annotation)
        for node in ast.walk(tree):
            if self.is_call(node):
                block = by_id[node.args[0].value]
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        child.lineno, child.end_lineno = block.span.line, block.span.end_line
                        child.col_offset = len(self.lines[block.span.line - 1][:block.span.column].encode("utf-8"))
                        child.end_col_offset = len(self.lines[block.span.end_line - 1][:block.span.end_column].encode("utf-8"))
        self.directives.bind(tree)
        return Unit(self.source, self.filename, tree, self.directives,
                    {b.id: b for b in merged}, rendered)

    @staticmethod
    def is_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == RUNTIME_NAME
                and node.func.attr == "execute" and bool(node.args)
                and isinstance(node.args[0], ast.Constant))


def parse(source: str, filename: str = "<aiython>") -> Unit:
    return Frontend(source, filename).build()
