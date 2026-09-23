"""Bounded, read-only project source tools. Never import searched modules."""
from pathlib import Path
import os

SOURCE_SUFFIXES = {'.py', '.pyi', '.md', '.rst', '.txt', '.js', '.ts', '.tsx', '.jsx', '.sql'}
EXCLUDED = {'node_modules', 'venv', 'env', 'build', 'dist', 'vendor', '__pycache__'}
MAX_FILE_BYTES = 512_000
MAX_SEARCH_BYTES = 8_000_000
MAX_ENTRIES = 10_000


class Codebase:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, name):
        relative = Path(name)
        if relative.is_absolute():
            try:
                relative = relative.relative_to(self.root)
            except ValueError:
                raise ValueError('Source path is outside project') from None
        if not relative.parts or '..' in relative.parts:
            raise ValueError('Use a source path inside the project without ..')
        for part in relative.parts:
            if part.startswith('.') or part in EXCLUDED or part.endswith('.egg-info'):
                raise ValueError('Path is excluded from codebase tools')
        candidate = self.root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError('Symlinks are excluded from codebase tools')
        if not candidate.resolve().is_relative_to(self.root):
            raise ValueError('Path is outside project')
        if candidate.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError('Only source/documentation files are available; configuration and credentials are excluded')
        return candidate

    def files(self):
        count = 0
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            count += len(dirs) + len(files)
            if count > MAX_ENTRIES:
                raise OverflowError('Project scan limit reached; use a more specific source path')
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in EXCLUDED
                              and not d.endswith('.egg-info') and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                relative = (Path(directory) / name).relative_to(self.root).as_posix()
                try:
                    path = self.path(relative)
                except ValueError:
                    continue
                if path.is_file():
                    yield relative

    def text(self, name):
        path = self.path(name)
        if not path.is_file():
            raise ValueError('Expected a regular source file')
        # Bound allocation even if the file grows after a size check.
        with path.open('rb') as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError('Source exceeds 512 KB read limit')
        if b'\x00' in data:
            raise ValueError('Binary source is not supported')
        return data.decode('utf-8-sig')

    def list_files(self, query='', offset=0, limit=100):
        items = []
        matched = 0
        try:
            for name in self.files():
                if query not in name:
                    continue
                matched += 1
                if matched <= offset:
                    continue
                if len(items) == limit:
                    return {'files': items, 'truncated': True, 'next_offset': offset + limit}
                items.append(name)
        except OverflowError as exc:
            return {'files': items, 'truncated': True, 'reason': str(exc)}
        return {'files': items, 'truncated': False}

    def search(self, query, path=None, limit=30):
        if not query or '\n' in query:
            raise ValueError('query must be a nonempty single-line literal')
        matches, skipped, size = [], 0, 0
        names = [path] if path is not None else self.files()
        try:
            for name in names:
                try:
                    source = self.text(name)
                except (ValueError, OSError, UnicodeError):
                    if path is not None:
                        raise
                    skipped += 1
                    continue
                size += len(source.encode('utf-8'))
                if size > MAX_SEARCH_BYTES:
                    return {'matches': matches, 'truncated': True, 'skipped_files': skipped,
                            'reason': 'Search byte limit reached; narrow to a source path'}
                for line, text in enumerate(source.splitlines(), 1):
                    if query in text:
                        if len(matches) == limit:
                            return {'matches': matches, 'truncated': True, 'skipped_files': skipped}
                        matches.append({'path': name, 'line': line, 'text': text[:1000],
                                        'line_truncated': len(text) > 1000})
        except OverflowError as exc:
            return {'matches': matches, 'truncated': True, 'skipped_files': skipped, 'reason': str(exc)}
        return {'matches': matches, 'truncated': False, 'skipped_files': skipped}

    def read(self, path, start_line=1, end_line=None):
        end_line = start_line + 99 if end_line is None else end_line
        if end_line < start_line or end_line - start_line >= 200:
            raise ValueError('Read between 1 and 200 lines per call')
        lines = self.text(path).splitlines()
        selected, size = [], 0
        for number in range(start_line, min(end_line, len(lines)) + 1):
            text = lines[number - 1]
            full_length = len(text)
            text = text[:8000]
            if size + len(text) > 16000:
                break
            selected.append({'line': number, 'text': text, 'line_truncated': full_length > len(text)})
            size += len(text)
        last = selected[-1]['line'] if selected else start_line - 1
        return {'path': path, 'lines': selected, 'total_lines': len(lines),
                'truncated': last < min(end_line, len(lines)),
                'next_line': last + 1 if last < len(lines) else None}
