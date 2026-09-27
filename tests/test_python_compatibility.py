"""Compare ordinary Python execution with Aiython in separate processes."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PythonCompatibilityTests(unittest.TestCase):
    def test_language_constructs_match_cpython(self):
        cases = {
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
