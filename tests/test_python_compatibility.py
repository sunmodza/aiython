"""Compare ordinary Python execution with Aiython in separate processes."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiython.cli import run_script


class PythonCompatibilityTests(unittest.TestCase):
    def test_entry_annotations_follow_python_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('')
            for version, present in (((3, 13), True), ((3, 14), False)):
                with self.subTest(version=version), patch('aiython.cli.sys.version_info', version):
                    namespace = run_script(path)
                    self.assertEqual('__annotations__' in namespace, present)

    def test_symlink_script_metadata_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source').mkdir()
            (root / 'links').mkdir()
            target = root / 'source' / 'target.py'
            target.write_text('import inspect, sys\n'
                              'print(sys.argv[0], __file__, __loader__.path, '
                              'inspect.currentframe().f_code.co_filename, sys.path[0], sep="\\n")\n')
            (root / 'links' / 'alias.py').symlink_to(target)
            python = subprocess.run([sys.executable, 'links/alias.py'], cwd=root,
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-m', 'aiython', 'links/alias.py'],
                                     cwd=root, capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_relative_script_argument_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('import sys\nprint(sys.argv[0])\nprint(__file__)\n')
            python = subprocess.run([sys.executable, 'main.py'], cwd=directory,
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-m', 'aiython', 'main.py'],
                                     cwd=directory, capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_language_constructs_match_cpython(self):
        cases = {
            'empty script': '',
            'entry builtins module': '''import builtins
print(type(__builtins__).__name__, __builtins__ is builtins)
''',
            'entry module metadata': '''print(type(__loader__).__name__, __loader__.name, __loader__.path == __file__)
print('__annotations__' in globals(), '__annotate__' in globals())
''',
            'module annotation metadata': '''value: int = 2
print('__annotations__' in globals(), '__annotate__' in globals())
''',
            'metaclass namespace': '''class Meta(type):
    def __new__(meta, name, bases, namespace):
        print(sorted(namespace))
        return super().__new__(meta, name, bases, namespace)
class Example(metaclass=Meta):
    value: int = 3
print(Example.value)
''',
            'custom class namespace': '''class Namespace(dict):
    def pop(self, *args): raise TypeError('removal forbidden')
    def __delitem__(self, key): raise TypeError('removal forbidden')
class Meta(type):
    @classmethod
    def __prepare__(meta, name, bases): return Namespace()
    def __new__(meta, name, bases, namespace):
        print(sorted(namespace))
        return super().__new__(meta, name, bases, namespace)
class Example(metaclass=Meta):
    for index in range(2):
        value: int = index
print(Example.value)
''',
            'class locals': '''class Example:
    snapshot = sorted(locals())
print(Example.snapshot)
''',
            'function locals': '''def run(value):
    result = value + 1
    print(sorted(locals()))
    return result
print(run(2))
''',
            'function frame locals': '''import inspect
def run(value):
    result = value + 1
    print(sorted(inspect.currentframe().f_locals))
    return result
print(run(2))
''',
            'attribute assignment locals and order': '''events = []
class Box:
    def __setattr__(self, name, value):
        events.append(('set', name, value))
        super().__setattr__(name, value)
box = Box()
def produce():
    events.append('value')
    return 3
def pick():
    events.append('target')
    return box
def run():
    pick().value = produce()
    print(run.__code__.co_varnames, sorted(locals()))
run()
print(events)
''',
            'async function locals': '''import asyncio
async def run(value):
    result = value + 1
    await asyncio.sleep(0)
    print(sorted(locals()))
    return result
print(asyncio.run(run(2)))
''',
            'recursive function scopes': '''def factorial(value: int) -> int:
    if value == 0:
        return 1
    return value * factorial(value - 1)
print(factorial(6))
''',
            'concurrent async function scopes': '''import asyncio
async def worker(value: int) -> tuple[int, list[str]]:
    await asyncio.sleep(0)
    return value, sorted(locals())
async def run():
    print(await asyncio.gather(worker(1), worker(2)))
asyncio.run(run())
''',
            'failed class body leaves no scope': '''try:
    class Broken:
        value: int = 1
        raise ValueError('stop')
except ValueError:
    pass
class Working:
    value: int = 2
print(Working.value)
''',
            'nested class and zero argument super': '''class Base:
    def value(self): return 1
class Outer:
    class Child(Base):
        def value(self): return super().value() + 1
print(Outer.Child().value())
''',
            'decorated slotted dataclass': '''from dataclasses import dataclass
@dataclass(slots=True)
class Point:
    x: int
print(Point(2).x, hasattr(Point(2), '__dict__'))
''',
            'enum': '''from enum import Enum
class Color(Enum):
    RED = 1
    BLUE = 2
print([item.name for item in Color])
''',
            'pattern matching': '''value = ('ok', 3)
match value:
    case ('ok', number): print(number)
    case _: print('missing')
''',
            'exception groups': '''try:
    raise ExceptionGroup('both', [ValueError('a'), TypeError('b')])
except* ValueError as error:
    print(type(error.exceptions[0]).__name__)
except* TypeError as error:
    print(type(error.exceptions[0]).__name__)
''',
            'generator delegation': '''def inner():
    received = yield 1
    return received
def outer():
    result = yield from inner()
    print(result)
item = outer()
print(next(item))
try: item.send(7)
except StopIteration: pass
''',
            'async generator': '''import asyncio
async def numbers():
    yield 1
    yield 2
async def run():
    print([number async for number in numbers()])
asyncio.run(run())
''',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            for name, source in cases.items():
                with self.subTest(name=name):
                    path.write_text(source)
                    python = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', str(path)],
                                             capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))
