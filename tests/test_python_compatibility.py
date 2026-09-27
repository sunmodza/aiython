"""Compare ordinary Python execution with Aiython in separate processes."""

import json
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from aiython.cli import main, module_source, run_script


class PythonCompatibilityTests(unittest.TestCase):
    def test_module_startup_without_existing_main_restores_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'standalone.py').write_text('value = 1\n')
            for version, annotations_present in (((3, 13), True), ((3, 14), False)):
                with self.subTest(version=version), patch.dict(sys.modules, {'__main__': None}), \
                        patch('pathlib.Path.cwd', return_value=root), patch.object(sys, 'path', sys.path[:]), \
                        patch('aiython.cli.sys.version_info', version):
                    spec, source, code, initial_main = module_source('standalone')
                    self.assertNotIn('__main__', sys.modules)
                    self.assertEqual(spec.name, 'standalone')
                    self.assertEqual(source, 'value = 1\n')
                    self.assertEqual(code.co_name, '<module>')
                    self.assertEqual(initial_main.__name__, '__main__')
                    self.assertEqual('__annotations__' in vars(initial_main), annotations_present)

    def test_safe_path_module_mode_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'safemodule.py').write_text('import sys\nprint(sys.flags.safe_path, sys.path[:2])\n')
            environment = {**os.environ, 'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
            python = subprocess.run([sys.executable, '-P', '-m', 'safemodule'], cwd=root, env=environment,
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-P', '-m', 'aiython', '-m', 'safemodule'], cwd=root,
                                     env=environment, capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_parent_package_startup_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'startup'
            package.mkdir()
            (package / '__init__.py').write_text('''import sys
initial_main = sys.modules['__main__']
if hasattr(initial_main, '__annotations__'):
    initial_main.__annotations__['from_init'] = int
print(sys.argv, sys.orig_argv, initial_main.__spec__, '__file__' in vars(initial_main))
''')
            (package / 'module.py').write_text('''import sys
from . import initial_main
print(initial_main is sys.modules['__main__'], 'from_init' in globals().get('__annotations__', {}), sys.path[0])
''')
            python = subprocess.run([sys.executable, '-B', '-m', 'startup.module', 'arg'], cwd=root,
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-B', '-m', 'aiython', '-m', 'startup.module', 'arg'],
                                     cwd=root, capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_console_entry_point_finds_current_directory_modules(self):
        console = Path(sys.executable).with_name('aiython')
        self.assertTrue(console.is_file())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'localmodule.py').write_text('import sys\nprint(sys.argv, sys.orig_argv, sys.path[0])\n')
            python = subprocess.run([sys.executable, '-m', 'localmodule', 'arg'], cwd=root,
                                    capture_output=True, text=True)
            aiython = subprocess.run([str(console), '-m', 'localmodule', 'arg'], cwd=root,
                                     capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_embedded_module_run_restores_host_import_path(self):
        original_path = sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'embedded_module.py').write_text('value = 1\n')
            with patch('pathlib.Path.cwd', return_value=root):
                main(['-m', 'embedded_module'])
        self.assertEqual(sys.path, original_path)

    def test_module_and_package_execution_match_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'example'
            package.mkdir()
            (package / '__init__.py').write_text('value = 7\n')
            source = '''import atexit, sys
from . import value
print(value, __name__, __package__, __spec__.name, __file__, __cached__)
print(sys.argv, sys.orig_argv, sys.path[0])
atexit.register(lambda: print('EXIT', sys.modules['__main__'].__file__,
                              sys.modules['__main__'].__cached__))
'''
            (package / 'module.py').write_text(source)
            (package / '__main__.py').write_text(source)
            for name in ('example.module', 'example'):
                with self.subTest(name=name):
                    python = subprocess.run([sys.executable, '-m', name, 'one', '-x'], cwd=root,
                                            capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', '-m', name, 'one', '-x'], cwd=root,
                                             capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_zip_and_sourceless_module_execution_match_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'modules.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('zipmodule.py',
                                 'import sys\nprint(__name__, __spec__.name, __file__, sys.argv)\n')
            source = root / 'sourceless.py'
            source.write_text('import sys\nprint(__file__, __cached__, sys.argv, sys.orig_argv)\n')
            py_compile.compile(str(source), cfile=str(root / 'sourceless.pyc'), doraise=True)
            source.unlink()
            environment = {**os.environ, 'PYTHONPATH': str(archive) + os.pathsep + os.environ.get('PYTHONPATH', '')}
            for name in ('zipmodule', 'sourceless'):
                with self.subTest(name=name):
                    python = subprocess.run([sys.executable, '-m', name, 'arg'], cwd=root, env=environment,
                                            capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', '-m', name, 'arg'], cwd=root,
                                             env=environment, capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_module_cli_reports_invalid_requests(self):
        for arguments, expected in ((['-m'], '-m requires a module name'),
                                    (['-m', 'module_that_does_not_exist'], 'No module named'),
                                    (['--stats'], 'a script path or -m module is required')):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, '-m', 'aiython', *arguments],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sourceless.py'
            source.write_text('value = 1\n')
            py_compile.compile(str(source), cfile=str(root / 'sourceless.pyc'), doraise=True)
            source.unlink()
            result = subprocess.run([sys.executable, '-m', 'aiython', '--explain', '-m', 'sourceless'],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Cannot explain a module without Python source', result.stderr)

    def test_module_config_comes_from_invoking_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            project.mkdir()
            config = project / 'aiython.toml'
            config.write_text('version=3\nmodel="openai/test"\n')
            external = root / 'external'
            external.mkdir()
            (external / 'externalmod.py').write_text('print("ok")\n')
            environment = {**os.environ, 'PYTHONPATH': str(external) + os.pathsep + os.environ.get('PYTHONPATH', '')}
            result = subprocess.run([sys.executable, '-m', 'aiython', '--explain', '-m', 'externalmod'],
                                    cwd=project, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['config']['config'], str(config))
            result = subprocess.run([sys.executable, '-m', 'aiython', '-m', 'externalmod'],
                                    cwd=project, env=environment, capture_output=True, text=True)
            self.assertEqual((result.returncode, result.stdout), (0, 'ok\n'), result.stderr)

    def test_safe_path_modes_match_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            (Path(directory) / 'sibling.py').write_text('value = 2\n')
            path.write_text('''import sys
print(sys.flags.safe_path, sys.path[:2], sys.orig_argv)
try:
    import sibling
except ModuleNotFoundError:
    print('sibling unavailable')
else:
    print(sibling.value)
''')
            for flag in ('-I', '-P'):
                with self.subTest(flag=flag):
                    python = subprocess.run([sys.executable, flag, str(path)],
                                            capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, flag, '-m', 'aiython', str(path)],
                                             capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_original_arguments_preserve_interpreter_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('import sys\nprint(sys.orig_argv)\n')
            python = subprocess.run([sys.executable, '-B', str(path), 'one'],
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-B', '-m', 'aiython', str(path), 'one'],
                                     capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_embedded_run_restores_host_process_state(self):
        original_argv = sys.argv
        original_orig_argv = sys.orig_argv
        original_path = sys.path[:]
        original_main = sys.modules.get('__main__')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('value = 2\n')
            run_script(path)
        self.assertIs(sys.argv, original_argv)
        self.assertIs(sys.orig_argv, original_orig_argv)
        self.assertEqual(sys.path, original_path)
        self.assertIs(sys.modules.get('__main__'), original_main)

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
            'original argument vector': '''import sys
print(sys.argv)
print(sys.orig_argv)
''',
            'atexit script state': '''import atexit, sys
def report():
    main = sys.modules.get('__main__')
    print(sys.argv[0], sys.path[0], getattr(main, '__file__', None),
          '__cached__' in globals(), sep=' | ')
    print(sys.orig_argv)
atexit.register(report)
''',
            'worker script state': '''import sys, threading, time
def worker():
    time.sleep(0.05)
    main = sys.modules.get('__main__')
    print(sys.argv[0], sys.path[0], getattr(main, '__file__', None), sep=' | ')
threading.Thread(target=worker).start()
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
            'class with only a docstring': '''class Example:
    "documentation"
print(Example.__doc__, sorted(Example.__dict__))
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
            'generator locals': '''import inspect
def generate(value):
    yield sorted(locals())
    yield sorted(inspect.currentframe().f_locals)
    yield generate.__code__.co_varnames
item = generate(3)
print(next(item), next(item), next(item))
''',
            'generator collected after suspension': '''import gc, weakref
def generate():
    yield 1
item = generate()
reference = weakref.ref(item)
next(item)
del item
gc.collect()
print(reference() is None)
''',
            'generator resumed on another thread': '''from concurrent.futures import ThreadPoolExecutor
def generate(value: int):
    yield value
    yield value + 1
item = generate(2)
print(next(item))
with ThreadPoolExecutor(max_workers=1) as pool:
    print(pool.submit(next, item).result())
''',
            'async generator': '''import asyncio
async def numbers():
    yield 1
    yield 2
async def run():
    print([number async for number in numbers()])
asyncio.run(run())
''',
            'async generator locals': '''import asyncio, inspect
async def generate(value):
    yield sorted(locals())
    yield sorted(inspect.currentframe().f_locals)
async def run():
    print([item async for item in generate(3)])
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
