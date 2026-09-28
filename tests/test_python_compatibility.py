"""Compare ordinary Python execution with Aiython in separate processes."""

import contextlib
import importlib
import io
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiython.cli import ModuleStartFinder, main, module_details, module_source, run_script
from aiython.config import resolve
from aiython.models import AiythonError, ProfileConfig, ResolvedConfig
from aiython.runtime import Runtime
from aiython.type_constraints import TypeViolation


class PythonCompatibilityTests(unittest.TestCase):
    def test_plain_entry_preserves_annotated_project_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'mixed_native_helper_for_aiython'
            (root / 'main.py').write_text(f'import {name}\n{name}.change()\n')
            (root / f'{name}.py').write_text('value: int = 1\n'
                                             'def change():\n'
                                             '    global value\n'
                                             '    value = "invalid"\n')
            runtime = Runtime(ResolvedConfig(None, root))
            previous = sys.modules.pop(name, None)
            try:
                with self.assertRaises(TypeViolation):
                    run_script(root / 'main.py', config=runtime.config, runtime=runtime)
                self.assertFalse(any(key.startswith(str(root / 'main.py') + ':')
                                     for key in runtime.checkpoints))
            finally:
                sys.modules.pop(name, None)
                if previous is not None:
                    sys.modules[name] = previous

    def test_plain_entry_rechecks_mutated_imported_typed_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'mixed_typed_instance_for_aiython'
            (root / f'{name}.py').write_text('class Box:\n'
                                             '    values: list[int]\n'
                                             '    def __init__(self):\n'
                                             '        self.values = [1]\n')
            previous = sys.modules.pop(name, None)
            try:
                for import_source in (f'import {name}\nmodule = {name}\n',
                                      f'module = __import__("{name}")\n',
                                      f'importer = getattr(__builtins__, "__import__")\n'
                                      f'module = importer("{name}")\n'):
                    with self.subTest(import_source=import_source):
                        sys.modules.pop(name, None)
                        (root / 'main.py').write_text(import_source +
                                                      'box = module.Box()\n'
                                                      'box.values.append("invalid")\n')
                        with self.assertRaises(TypeViolation):
                            run_script(root / 'main.py', config=ResolvedConfig(None, root))
            finally:
                sys.modules.pop(name, None)
                if previous is not None:
                    sys.modules[name] = previous

    def test_unconfigured_program_raises_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'main.py'
            script.write_text('1 / 0\n')
            config = ResolvedConfig(None, root)
            runtime = Runtime(config)
            with self.assertRaises(ZeroDivisionError):
                run_script(script, config=config, runtime=runtime)
            self.assertFalse(runtime.checkpoints)
            result = subprocess.run([sys.executable, '-m', 'aiython', str(script)],
                                    cwd=root, capture_output=True, text=True, timeout=10)
            native = subprocess.run([sys.executable, str(script)], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual((result.returncode, result.stdout, result.stderr),
                             (native.returncode, native.stdout, native.stderr))
            command = subprocess.run([sys.executable, '-m', 'aiython', '-c', '1/0'],
                                     cwd=root, capture_output=True, text=True, timeout=10)
            native_command = subprocess.run([sys.executable, '-c', '1/0'],
                                            cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual((command.returncode, command.stdout, command.stderr),
                             (native_command.returncode, native_command.stdout,
                              native_command.stderr))

    def test_module_resolution_errors_are_explicit(self):
        with self.assertRaisesRegex(AiythonError, 'Relative module names not supported'):
            module_details('.relative')
        with self.assertRaisesRegex(AiythonError, 'Error while finding module specification'):
            module_details('missing_parent_for_aiython.child')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'empty_package_for_aiython'
            package.mkdir()
            (package / '__init__.py').write_text('')
            with patch.object(sys, 'path', [str(root), *sys.path]):
                with self.assertRaisesRegex(AiythonError, 'is a package and cannot be directly executed'):
                    module_details('empty_package_for_aiython')
                nested_main = package / '__main__'
                nested_main.mkdir()
                (nested_main / '__init__.py').write_text('')
                importlib.invalidate_caches()
                with self.assertRaisesRegex(AiythonError, 'Cannot use package as __main__ module'):
                    module_details('empty_package_for_aiython.__main__')
        no_loader = SimpleNamespace(submodule_search_locations=None, loader=None)
        no_code = SimpleNamespace(submodule_search_locations=None,
                                  loader=SimpleNamespace(get_source=lambda _: None, get_code=lambda _: None))
        for spec, expected in ((no_loader, 'namespace package and cannot be executed'),
                               (no_code, 'No code object available')):
            with self.subTest(expected=expected), patch('aiython.cli.importlib.util.find_spec', return_value=spec):
                with self.assertRaisesRegex(AiythonError, expected):
                    module_details('custom_loader')

    def test_module_start_finder_leaves_sourceless_loaders_unchanged(self):
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        with patch('aiython.cli.ProjectFinder.find_spec', return_value=None):
            self.assertIsNone(ModuleStartFinder(runtime).find_spec('missing'))
        loader = SimpleNamespace(get_source=lambda _: None)
        spec = SimpleNamespace(loader=loader, origin='sourceless.pyc')
        with patch('aiython.cli.ProjectFinder.find_spec', return_value=spec):
            self.assertIsNone(ModuleStartFinder(runtime).find_spec('sourceless'))
        runtime.capabilities.close()

    def test_module_resolution_warns_when_parent_imports_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'early_import_package'
            package.mkdir()
            (package / '__init__.py').write_text('from . import task\n')
            (package / 'task.py').write_text('value = 1\n')
            with patch.object(sys, 'path', [str(root), *sys.path]), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', RuntimeWarning)
                spec, source, code = module_details('early_import_package.task')
            self.assertEqual(spec.name, 'early_import_package.task')
            self.assertEqual(source, 'value = 1\n')
            self.assertIsNone(code)
            self.assertEqual(len(caught), 1)
            self.assertIn('found in sys.modules after import of package', str(caught[0].message))

    def test_command_and_stdin_execution_match_cpython(self):
        source = '''import atexit, inspect, sys
def report(stage):
    main = sys.modules['__main__']
    print(stage, sys.argv, sys.orig_argv, sys.path[:2])
    print(__name__, __package__, __spec__, getattr(__loader__, '__name__', type(__loader__).__name__))
    print(vars(main).get('__file__', 'ABSENT'), vars(main).get('__cached__', 'ABSENT'))
    print(inspect.currentframe().f_code.co_filename)
report('RUN')
atexit.register(lambda: report('EXIT'))
'''
        with tempfile.TemporaryDirectory() as directory:
            for name, arguments, input_source in (('command', ['-c', source, 'one', '-x'], None),
                                                  ('stdin', ['-', 'one', '-x'], source),
                                                  ('implicit stdin', [], source)):
                with self.subTest(name=name):
                    python = subprocess.run([sys.executable, *arguments], input=input_source,
                                            cwd=directory, capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', *arguments], input=input_source,
                                             cwd=directory, capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_runtime_name_collision_in_entry_and_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'helper.py').write_text(
                '__aiython_runtime__ = 11\n'
                'def read(): return __aiython_runtime__\n'
                'def internal_names(): return sorted(name for name in globals() '
                'if name.startswith("__aiython_runtime"))\n')
            (root / 'main.py').write_text(
                '__aiython_runtime__ = 7\n'
                '__aiython_runtime_1__ = 8\n'
                'import helper\n'
                'print(__aiython_runtime__, __aiython_runtime_1__, helper.read())\n'
                'print(sorted(name for name in globals() if name.startswith("__aiython_runtime")), '
                'helper.internal_names())\n')
            package = root / 'runtime_package'
            package.mkdir()
            (package / '__init__.py').write_text('__aiython_runtime__ = 5\n')
            (package / '__main__.py').write_text(
                'from . import __aiython_runtime__ as parent_value\n'
                '__aiython_runtime__ = 7\n'
                'print(parent_value, __aiython_runtime__)\n'
                'print(sorted(name for name in globals() if name.startswith("__aiython_runtime")))\n')
            for python_args, aiython_args in ((['main.py'], ['main.py']),
                                              (['-m', 'runtime_package'], ['-m', 'runtime_package'])):
                with self.subTest(python_args=python_args):
                    python = subprocess.run([sys.executable, *python_args], cwd=root,
                                            capture_output=True, text=True, timeout=10)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', *aiython_args], cwd=root,
                                             capture_output=True, text=True, timeout=10)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_source_import_and_module_bytecode_cache_match_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('import', 'module'):
                for disabled in (False, True):
                    with self.subTest(mode=mode, disabled=disabled):
                        outputs = []
                        for executable in ('python', 'aiython'):
                            root = Path(directory) / f'{mode}-{disabled}-{executable}'
                            root.mkdir()
                            (root / 'helper.py').write_text('value = 3\n')
                            (root / 'main.py').write_text(
                                'from pathlib import Path\nimport helper\n'
                                'print(Path(helper.__cached__).is_file())\n')
                            (root / 'target.py').write_text(
                                'from pathlib import Path\nprint(Path(__cached__).is_file())\n')
                            command = (['main.py'] if mode == 'import' else ['-m', 'target'])
                            if executable == 'aiython':
                                command.insert(0, '-m')
                                command.insert(1, 'aiython')
                            env = os.environ.copy()
                            if disabled:
                                env['PYTHONDONTWRITEBYTECODE'] = '1'
                            else:
                                env.pop('PYTHONDONTWRITEBYTECODE', None)
                            result = subprocess.run([sys.executable, *command], cwd=root,
                                                    env=env, capture_output=True, text=True)
                            outputs.append((result.returncode, result.stdout, result.stderr))
                        self.assertEqual(outputs[1], outputs[0])

    def test_ai_source_and_bridge_preserve_user_runtime_name(self):
        from aiython.frontend import parse

        class Agent:
            def __init__(self):
                self.observed = []

            def execute(self, request, bridge):
                bridge.exec('global __aiython_runtime__\n__aiython_runtime__ += 1\n'
                            'def generated(): return __aiython_runtime__')
                self.observed.append(bridge.eval('generated()'))
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'main.py'
            source = ('import asyncio\n'
                      'from aiython.collaboration import Group\n'
                      '__aiython_runtime__ = 7\n'
                      '__aiython_runtime_1__ = 8\n'
                      'answer: int = choose seven\n'
                      'async def choose() -> int:\n'
                      '    return choose seven\n'
                      'async_answer = asyncio.run(choose())\n'
                      'result = (__aiython_runtime__, __aiython_runtime_1__, answer, async_answer, Group().project_root)\n')
            path.write_text(source)
            self.assertEqual(parse(source, str(path)).runtime_name, '__aiython_runtime_2__')
            config = ResolvedConfig(None, root, 'default',
                                    {'default': ProfileConfig('default', 'fake', 'test')})
            agent = Agent()
            namespace = run_script(path, config=config, agent_factory=lambda _: agent)
            self.assertEqual(namespace['result'], (9, 8, 7, 7, str(root)))
            self.assertEqual(agent.observed, [8, 9])
            self.assertEqual([name for name in namespace if name.startswith('__aiython_runtime')],
                             ['__aiython_runtime__', '__aiython_runtime_1__'])

    def test_ai_request_and_bridge_expose_user_prefixed_names(self):
        case = self

        class Agent:
            def execute(self, request, bridge):
                case.assertEqual(
                    {key: request.related_objects[key]
                     for key in ('__aiython_runtime__', '__aiython_user',
                                 '__aiython_recovery_counts__', '__aiython_recovery_attempt_user')},
                    {'__aiython_runtime__': 7, '__aiython_user': 5,
                     '__aiython_recovery_counts__': 3, '__aiython_recovery_attempt_user': 4})
                bridge.set('__aiython_runtime__', 8)
                bridge.set('__aiython_user', 6)
                bridge.set('__aiython_recovery_counts__', 9)
                bridge.set('__aiython_recovery_attempt_user', 10)
                bridge.set('__aiython_runtime_1__', 11)
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'main.py'
            path.write_text('__aiython_runtime__ = 7\n'
                            '__aiython_user = 5\n'
                            '__aiython_recovery_counts__ = 3\n'
                            '__aiython_recovery_attempt_user = 4\n'
                            'answer = choose seven using __aiython_runtime__ and __aiython_user '
                            'and __aiython_recovery_counts__ and __aiython_recovery_attempt_user\n'
                            'result = (__aiython_runtime__, __aiython_user, '
                            '__aiython_recovery_counts__, __aiython_recovery_attempt_user, '
                            '__aiython_runtime_1__, answer)\n')
            agent = Agent()
            config = ResolvedConfig(None, root, 'default',
                                    {'default': ProfileConfig('default', 'fake', 'test')})
            namespace = run_script(path, config=config, agent_factory=lambda _: agent)
            self.assertEqual(namespace['result'], (8, 6, 9, 10, 11, 7))

    def test_directory_and_zipapp_execution_match_cpython(self):
        source = '''import atexit, inspect, sys
from helper import value
def report(stage):
    main = sys.modules['__main__']
    print(stage, value, sys.argv, sys.orig_argv, sys.path[:2])
    print(__name__, __package__, __spec__.name, __spec__.origin)
    print(vars(main).get('__file__', 'ABSENT'), vars(main).get('__cached__', 'ABSENT'))
    print(type(__loader__).__name__, inspect.currentframe().f_code.co_filename)
report('RUN')
atexit.register(lambda: report('EXIT'))
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / 'app'
            app.mkdir()
            (app / '__main__.py').write_text(source)
            (app / 'helper.py').write_text('value = 7\n')
            archive = root / 'app.pyz'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('__main__.py', source)
                package.writestr('helper.py', 'value = 7\n')
            for name in ('app', 'app.pyz'):
                with self.subTest(name=name):
                    python = subprocess.run([sys.executable, name, 'one', '-x'], cwd=root,
                                            capture_output=True, text=True)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', name, 'one', '-x'], cwd=root,
                                             capture_output=True, text=True)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_sourceless_directory_execution_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / 'app'
            app.mkdir()
            source = app / '__main__.py'
            source.write_text('import sys\nprint(__file__, __cached__, sys.argv, sys.path[0])\n')
            py_compile.compile(str(source), cfile=str(app / '__main__.pyc'), doraise=True)
            source.unlink()
            python = subprocess.run([sys.executable, 'app'], cwd=root, capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-m', 'aiython', 'app'], cwd=root,
                                     capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))
            explanation = subprocess.run([sys.executable, '-m', 'aiython', '--explain', 'app'],
                                         cwd=root, capture_output=True, text=True)
            self.assertNotEqual(explanation.returncode, 0)
            self.assertIn('Cannot explain an entry without Python source', explanation.stderr)

    def test_direct_bytecode_execution_matches_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('''import atexit, inspect, sys
def report():
    main = sys.modules['__main__']
    print(sys.argv, sys.orig_argv, sys.path[:2])
    print(vars(main).get('__file__', 'ABSENT'), vars(main).get('__cached__', 'ABSENT'))
    print(type(__loader__).__name__, __package__, __spec__, inspect.currentframe().f_code.co_filename)
    print(sorted(name for name in globals() if name.startswith('__aiython_runtime')))
report()
atexit.register(report)
''')
            bytecode = root / 'direct.pyc'
            py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
            source.unlink()
            python = subprocess.run([sys.executable, 'direct.pyc', 'arg'], cwd=root,
                                    capture_output=True, text=True)
            aiython = subprocess.run([sys.executable, '-m', 'aiython', 'direct.pyc', 'arg'], cwd=root,
                                     capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_new_entry_modes_report_invalid_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / 'empty'
            empty.mkdir()
            for arguments, expected in ((['-c'], '-c requires a command string'),
                                        (['empty'], 'Cannot find __main__')):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([sys.executable, '-m', 'aiython', *arguments], cwd=root,
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
            for arguments, input_source in ((['--explain', '-c', 'value = 1'], None),
                                            (['--explain', '-'], 'value = 1\n')):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([sys.executable, '-m', 'aiython', *arguments], cwd=root,
                                            input=input_source, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)['blocks'], [])

    def test_safe_path_modes_for_new_entries_match_cpython(self):
        source = 'import sys\nprint(sys.flags.safe_path, sys.path[:2])\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / 'app'
            app.mkdir()
            (app / '__main__.py').write_text(source)
            with zipfile.ZipFile(root / 'app.pyz', 'w') as package:
                package.writestr('__main__.py', source)
            for flag in ('-I', '-P'):
                for name, arguments, input_source in (('command', ['-c', source], None),
                                                      ('stdin', ['-'], source),
                                                      ('implicit stdin', [], source),
                                                      ('directory', ['app'], None),
                                                      ('zipapp', ['app.pyz'], None)):
                    with self.subTest(flag=flag, name=name):
                        python = subprocess.run([sys.executable, flag, *arguments], input=input_source,
                                                cwd=root, capture_output=True, text=True)
                        aiython = subprocess.run([sys.executable, flag, '-m', 'aiython', *arguments],
                                                 input=input_source, cwd=root, capture_output=True, text=True)
                        self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                         (python.returncode, python.stdout, python.stderr))

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
                    self.assertIsNone(code)
                    self.assertEqual(initial_main.__name__, '__main__')
                    self.assertEqual('__annotations__' in vars(initial_main), annotations_present)

    def test_module_with_ai_source_uses_aiython_compiler(self):
        class Agent:
            def execute(self, request, runtime):
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'ai_source_package'
            package.mkdir()
            (package / '__init__.py').write_text('value = 1\n')
            (package / 'task.py').write_text('answer = compute the answer\n')
            with patch('pathlib.Path.cwd', return_value=root), patch.object(sys, 'path', sys.path[:]):
                spec, source, code, initial_main = module_source('ai_source_package.task')
                self.assertIsNone(code)
                self.assertIn('compute the answer', source)
                config = ResolvedConfig(None, root, 'default',
                                        {'default': ProfileConfig('default', 'fake', 'test')})
                namespace = run_script(Path(spec.origin), config=config, agent_factory=lambda _: Agent(),
                                       source=source, module_spec=spec, module_invocation='ai_source_package.task',
                                       initial_main=initial_main, entry_kind='module')
                self.assertEqual(namespace['answer'], 7)
            result = subprocess.run([sys.executable, '-m', 'aiython', '--explain', '-m', 'ai_source_package.task'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['blocks']), 1)

    def test_module_parent_package_with_ai_source_uses_same_runtime(self):
        class Agent:
            def execute(self, request, runtime):
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'ai_parent_package'
            package.mkdir()
            (package / '__init__.py').write_text('prefix = decide a number\n')
            (package / 'task.py').write_text('from . import prefix\nanswer = decide another number\n')
            config = ResolvedConfig(None, root, 'default',
                                    {'default': ProfileConfig('default', 'fake', 'test')})
            runtime = Runtime(config, agent_factory=lambda _: Agent())
            with patch('pathlib.Path.cwd', return_value=root), patch.object(sys, 'path', sys.path[:]):
                spec, source, code, initial_main = module_source('ai_parent_package.task', runtime=runtime)
                self.assertIsNone(code)
                self.assertEqual(type(spec.loader).__name__, 'SourceFileLoader')
                namespace = run_script(Path(spec.origin), config=config, runtime=runtime,
                                       source=source, module_spec=spec, module_invocation='ai_parent_package.task',
                                       initial_main=initial_main, entry_kind='module')
                self.assertEqual((namespace['prefix'], namespace['answer']), (7, 7))
            explanation = subprocess.run([sys.executable, '-m', 'aiython', '--explain', '-m',
                                          'ai_parent_package.task'], cwd=root, capture_output=True, text=True)
            self.assertEqual(explanation.returncode, 0, explanation.stderr)
            self.assertEqual([block['statement'] for block in json.loads(explanation.stdout)['blocks']],
                             ['decide another number'])

    def test_explain_nested_ai_packages_does_not_execute_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'outer_package'
            child = parent / 'inner_package'
            child.mkdir(parents=True)
            (parent / '__init__.py').write_text(
                'raise RuntimeError("outer package executed")\nvalue = decide a value\n')
            (child / '__init__.py').write_text(
                'raise RuntimeError("inner package executed")\nvalue = decide another value\n')
            (child / 'task.py').write_text('answer = determine the answer\n')
            result = subprocess.run([sys.executable, '-m', 'aiython', '--explain', '-m',
                                     'outer_package.inner_package.task'], cwd=root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([block['statement'] for block in json.loads(result.stdout)['blocks']],
                             ['determine the answer'])
            runtime = Runtime(ResolvedConfig(None, root))
            try:
                with patch('pathlib.Path.cwd', return_value=root), patch.object(sys, 'path', sys.path[:]):
                    spec, source, code, _ = module_source(
                        'outer_package.inner_package.task', runtime=runtime, explain=True)
                self.assertEqual(spec.name, 'outer_package.inner_package.task')
                self.assertIn('determine the answer', source)
                self.assertIsNone(code)
                self.assertNotIn('outer_package', sys.modules)
                self.assertNotIn('outer_package.inner_package', sys.modules)
            finally:
                runtime.capabilities.close()

    def test_cli_retries_parent_ai_source_with_one_runtime(self):
        class Agent:
            def execute(self, request, runtime):
                return 7

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'aiython.toml').write_text('version=3\nmodel="openai/test"\n')
            package = root / 'retry_parent_package'
            package.mkdir()
            (package / '__init__.py').write_text('prefix = decide a number\n')
            (package / 'task.py').write_text('from . import prefix\nprint(prefix)\n')
            runtime = Runtime(resolve(root / '__main__.py'), agent_factory=lambda _: Agent())
            output = io.StringIO()
            with patch('pathlib.Path.cwd', return_value=root), patch.object(sys, 'path', sys.path[:]), \
                    patch('aiython.cli.Runtime', return_value=runtime), contextlib.redirect_stdout(output):
                main(['-m', 'retry_parent_package.task'])
            self.assertEqual(output.getvalue(), '7\n')

    def test_cli_does_not_repeat_package_that_raises_syntax_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'raised_syntax_package'
            package.mkdir()
            marker = root / 'count.txt'
            (package / '__init__.py').write_text(
                'from pathlib import Path\n'
                f'path = Path({str(marker)!r})\n'
                'path.write_text(path.read_text() + "x" if path.exists() else "x")\n'
                'raise SyntaxError("raised by package", (__file__, 1, 1, "x"))\n')
            (package / 'task.py').write_text('print("unreachable")\n')
            result = subprocess.run([sys.executable, '-m', 'aiython', '-m', 'raised_syntax_package.task'],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('raised by package', result.stderr)
            self.assertEqual(marker.read_text(), 'x')

    def test_missing_module_error_precedes_invalid_project_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'aiython.toml').write_text('version=3\nmodel="openai/test"\n'
                                               'env_file=".aiython/missing.env"\n')
            result = subprocess.run([sys.executable, '-m', 'aiython', '-m', 'module_that_does_not_exist'],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No module named', result.stderr)
            self.assertNotIn('env_file', result.stderr)

    def test_failed_parent_ai_retry_closes_runtime(self):
        config = ResolvedConfig(None, Path.cwd())
        runtime = unittest.mock.Mock()
        first_error = SyntaxError('invalid', ('parent_source.py', 1, 1, 'invalid'))
        with patch('aiython.cli.module_source', side_effect=[first_error, AiythonError('retry failed')]), \
                patch('aiython.cli.resolve', return_value=config), \
                patch('aiython.cli.Runtime', return_value=runtime), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['-m', 'failing_parent.task'])
        runtime.capabilities.close.assert_called_once_with()

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
                                    (['--explain', '-m', 'module_that_does_not_exist'], 'No module named')):
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
            self.assertIn('Cannot explain an entry without Python source', result.stderr)

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

    def test_trace_and_profile_callbacks_match_cpython(self):
        with tempfile.TemporaryDirectory() as directory:
            for callback in ('trace', 'profile'):
                with self.subTest(callback=callback):
                    source = f'''import sys
events = []
def callback(frame, event, arg):
    if frame.f_code.co_name == 'target' and event == 'call':
        events.append('target')
    return callback
sys.set{callback}(callback)
def target(): return 1
target()
sys.set{callback}(None)
print(events)
'''
                    python = subprocess.run([sys.executable, '-c', source], cwd=directory,
                                            capture_output=True, text=True, timeout=10)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', '-c', source],
                                             cwd=directory, capture_output=True, text=True, timeout=10)
                    self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                                     (python.returncode, python.stdout, python.stderr))

    def test_interactive_entries_preserve_main_module_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'main.py'
            script.write_text('value = 3\n')
            module = root / 'samplemod.py'
            module.write_text('value = 3\n')
            app = root / 'app'
            app.mkdir()
            (app / '__main__.py').write_text('value = 3\n')
            entries = ((str(script),), ('-m', 'samplemod'), (str(app),),
                       ('-c', 'value = 3'))
            prompt = ('import sys\n'
                      'print(value, __name__, "__file__" in globals(), '
                      'sys.orig_argv[1:])\n')
            for entry in entries:
                with self.subTest(entry=entry):
                    python = subprocess.run([sys.executable, '-i', *entry], input=prompt,
                                            cwd=root, capture_output=True, text=True, timeout=10)
                    aiython = subprocess.run([sys.executable, '-m', 'aiython', '-i', *entry],
                                             input=prompt, cwd=root, capture_output=True,
                                             text=True, timeout=10)
                    self.assertEqual((aiython.returncode, aiython.stdout),
                                     (python.returncode, python.stdout))

    def test_language_constructs_match_cpython(self):
        cases = {
            'empty script': '',
            'no runtime binding in globals': '''print(sorted(name for name in globals() if name.startswith('__aiython_')))
''',
            'nested statements leave globals unchanged': '''if True:
    value = 1
for item in range(2):
    value += item
print(value, sorted(name for name in globals() if name.startswith('__aiython_')))
''',
            'entry builtins module': '''import builtins
print(type(__builtins__).__name__, __builtins__ is builtins)
''',
            'runtime binding collision': '''__aiython_runtime__ = 7
__aiython_runtime_1__ = 8
print(__aiython_runtime__, __aiython_runtime_1__, f'{__aiython_runtime__}')
globals()['__aiython_runtime__'] += 1
print(__aiython_runtime__)
def local():
    __aiython_runtime__ = 9
    return __aiython_runtime__
class Box:
    __aiython_runtime__ = 10
    def read(self): return self.__aiython_runtime__
print(local(), Box().read())
''',
            'other internal-looking user names': '''__aiython_type_scope__ = 3
__aiython_recovery_counts__ = 4
__aiython_recovery_attempt_user = 5
def values():
    __aiython_type_scope__ = 6
    return __aiython_type_scope__
class Box:
    __aiython_type_scope__ = 7
print(__aiython_type_scope__, __aiython_recovery_counts__,
      __aiython_recovery_attempt_user, values(), Box.__aiython_type_scope__)
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
            'non-name annotation evaluation order': '''events = []
class Target:
    def __setitem__(self, key, value):
        events.append('set')
target = Target()
def key():
    events.append('key')
    return 0
def annotation():
    events.append('annotation')
    return int
target[key()]: annotation() = 2
def fail():
    try:
        missing[also_missing]: 1/0 = 0
    except NameError as error:
        print(type(error).__name__)
    try:
        missing.attr: 1/0 = 0
    except NameError as error:
        print(type(error).__name__)
fail()
print(events)
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
            'method called while metaclass builds mro': '''captured = None
class Meta(type):
    def mro(cls):
        cls.__dict__['capture']()
        return super().mro()
class Example(metaclass=Meta):
    def capture():
        global captured
        captured = __class__
print(captured is Example)
''',
            'private attribute writes in nested classes': '''class Box:
    class Descriptor:
        def __init__(self, getter): self.__getter = getter
        def __get__(self, owner, kind=None): return self.__getter(owner)
    def __init__(self): self.__value: int = 3
    def read(self): return self.__value
    value = Descriptor(read)
box = Box()
print(box.value, box.read(), sorted(vars(box)))
''',
            'custom instance dict descriptors': '''import sys
calls = []
class Box:
    value: int
    def __init__(self): self.value = 3
    @property
    def __dict__(self):
        calls.append('dict property')
        return 'not a dict'
box = Box()
print(box.value, calls)
class Module(type(sys)):
    __dict__ = property(lambda self: 'not a dict')
module = Module('example')
try: dir(module)
except TypeError: print('TypeError')
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
            'custom attribute access during assignment': '''events = []
class Item:
    def __iadd__(self, other):
        events.append(('iadd', other))
        return self
class Box:
    def __init__(self): self.value = Item()
    def __getattribute__(self, name):
        events.append(('get', name))
        return object.__getattribute__(self, name)
    def __setattr__(self, name, value):
        events.append(('set', name))
        object.__setattr__(self, name, value)
box = Box()
box.value += 1
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
            'dataclass InitVar and post init': '''from dataclasses import dataclass, InitVar
@dataclass
class Point:
    x: int
    offset: InitVar[int]
    def __post_init__(self, offset):
        self.x += offset
point = Point(1, 2)
print(point.x, hasattr(point, 'offset'))
''',
            'dataclass keyword only marker': '''from dataclasses import dataclass, KW_ONLY
@dataclass
class Point:
    x: int
    _: KW_ONLY
    label: str = 'x'
print(Point(1, label='ready').x, Point(1).label)
''',
            'dataclass default factory': '''from dataclasses import dataclass, field
@dataclass
class Point:
    values: list[int] = field(default_factory=list)
point = Point()
point.values.append(1)
print(point.values)
''',
            'dataclass union with NoneType': '''from dataclasses import dataclass
from typing import Union
@dataclass
class Item:
    value: Union[int, type(None)] = None
print(Item.__doc__, Item().value)
''',
            'dataclass class local markers': '''from dataclasses import dataclass, InitVar
from typing import ClassVar
@dataclass
class Item:
    ClassMarker = ClassVar
    InitMarker = InitVar
    marker: ClassMarker = 1
    input: InitMarker
    value: int = 0
    def __post_init__(self, input):
        self.value = input
item = Item(2)
print(item.marker, item.value, 'input' in vars(item))
''',
            'dataclass string annotation whitespace': '''from dataclasses import dataclass, InitVar
@dataclass
class Item:
    value: ' int ' = 1
    extra: ' InitVar ' = 2
    def __post_init__(self, extra):
        self.value += extra
item = Item()
print(item.value, hasattr(item, 'extra'))
''',
            'dataclass without generated init': '''from dataclasses import dataclass
@dataclass(slots=True, init=False)
class Point:
    x: int
point = Point()
print(hasattr(point, 'x'))
point.x = 3
print(point.x)
''',
            'slotted dataclass custom pickle state': '''from dataclasses import dataclass, field
import pickle
@dataclass(frozen=True, slots=True)
class Point:
    x: int
    cached: bool = field(default=False, compare=False)
    restored: bool = field(default=False, compare=False)
    def __getstate__(self):
        return [self.x]
    def __setstate__(self, state):
        object.__setattr__(self, 'x', state[0])
        object.__setattr__(self, 'restored', True)
point = pickle.loads(pickle.dumps(Point(2)))
print(point.x, point.restored, hasattr(point, 'cached'))
''',
            'dataclass descriptor with local annotation': '''from dataclasses import dataclass
def check():
    class Descriptor:
        def __get__(self, instance, owner=None):
            return 100 if instance is None else instance._value
        def __set__(self, instance, value):
            instance._value = value
    @dataclass
    class Item:
        value: ' Descriptor ' = Descriptor()
    first, second = Item(), Item(5)
    second.value = 7
    print(first.value, second.value)
    class Setter:
        calls = []
        def __set__(self, instance, value):
            self.calls.append(value)
    @dataclass
    class SetOnly:
        value: Setter = Setter()
    SetOnly(3)
    print(SetOnly.value.calls)
check()
''',
            'dataclass local type alias before decoration': '''from dataclasses import dataclass
def check():
    alias = list
    try:
        @dataclass
        class Item:
            values: alias = []
    except ValueError as error:
        print(type(error).__name__, 'mutable default' in str(error))
check()
''',
            'class annotation ignores dynamic caller alias': '''from dataclasses import dataclass
class Expected:
    pass
def make():
    @dataclass
    class Item:
        value: Expected = Expected()
    return Item()
def caller():
    Expected = str
    return make()
print(type(caller().value).__name__)
''',
            'dataclass escaped local type': '''from dataclasses import dataclass
def make():
    class Local:
        pass
    @dataclass
    class Box:
        value: Local
    return Box(Local())
box = make()
box.value = type(box.value)()
print(type(box.value).__name__)
''',
            'dataclass intermediate non-dataclass base': '''from dataclasses import dataclass
@dataclass
class Base:
    x: int
class Middle(Base):
    y: int
@dataclass
class Child(Middle):
    z: int
child = Child(1, 2)
print(child.x, child.z, hasattr(child, 'y'))
''',
            'dataclass generated init calls custom setter': '''from dataclasses import dataclass
calls = []
@dataclass
class Pair:
    left: int
    right: int
    def __setattr__(self, name, value):
        calls.append((name, value))
        object.__setattr__(self, name, value)
pair = Pair(1, 2)
print(calls, pair.left, pair.right)
''',
            'dataclass subclass of GenericAlias': '''from dataclasses import dataclass
import types
@dataclass
class Alias(types.GenericAlias):
    origin: type
    args: type
alias = Alias(list, int)
print(alias.__origin__.__name__, alias.__args__[0].__name__)
''',
            'pydantic field descriptors': '''from pydantic import BaseModel, Field
class Point(BaseModel):
    x: int = Field(default=1, ge=0)
    values: list[int] = Field(default_factory=list)
point = Point()
point.values.append(2)
print(point.x, point.values)
''',
            'pydantic Annotated field': '''from typing import Annotated
from pydantic import BaseModel, Field
class Point(BaseModel):
    x: Annotated[int, Field(ge=0)] = 2
print(Point().x)
''',
            'Annotated metadata evaluation': '''from typing import Annotated
calls = []
def marker():
    calls.append('called')
    return object()
value: Annotated[int, marker()] = 2
print(value, calls)
''',
            'pydantic private field': '''from pydantic import BaseModel, PrivateAttr
class Point(BaseModel):
    _cache: list[int] = PrivateAttr(default_factory=list)
point = Point()
point._cache.append(2)
print(point._cache)
''',
            'attrs field descriptors': '''from attrs import define, field
@define
class Point:
    x: int = field(default=1)
print(Point().x)
''',
            'attrs Factory default': '''from attrs import define, Factory
@define
class Point:
    values: list[int] = Factory(list)
point = Point()
point.values.append(2)
print(point.values)
''',
            'annotated Python descriptors': '''from functools import cached_property
class Field:
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, obj, owner=None):
        return 2
class Item:
    first: int = Field()
    second: int = property(lambda self: 3)
    third: int = cached_property(lambda self: 4)
item = Item()
print(item.first, item.second, item.third)
''',
            'enum': '''from enum import Enum
class Color(Enum):
    RED = 1
    BLUE = 2
print([item.name for item in Color])
''',
            'annotated enum members and auto': '''from enum import Enum, IntEnum, StrEnum, Flag, auto
class Plain(Enum):
    RED: int = 1
class Number(IntEnum):
    RED: int = auto()
class Text(StrEnum):
    RED: str = auto()
class Bits(Flag):
    RED: int = auto()
print(Plain.RED.value, Number.RED.value, Text.RED.value, Bits.RED.value)
''',
            'self annotated enum member': '''from __future__ import annotations
from enum import Enum, auto
class Color(Enum):
    RED: Color = auto()
color = Color.RED
print(color.value)
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
            'generator function name changes': '''def generate():
    yield 1
item = generate()
item.__name__ = 'item_name'
item.__qualname__ = 'item_qualname'
generate.__name__ = 'generate_name'
generate.__qualname__ = 'generate_qualname'
new_item = generate()
print(item.__name__, item.__qualname__, new_item.__name__, new_item.__qualname__)
''',
            'generator close releases arguments': '''class DetectDelete:
    def __init__(self):
        DetectDelete.deleted = False
    def __del__(self):
        DetectDelete.deleted = True
def generate(arg):
    yield
item = generate(DetectDelete())
item.close()
print(DetectDelete.deleted, item.gi_frame is None)
item = generate(DetectDelete())
next(item)
item.close()
print(DetectDelete.deleted, item.gi_frame is None)
item = generate(DetectDelete())
item.gi_frame.clear()
print(DetectDelete.deleted, item.gi_frame is None)
''',
            'generator delegation identity': '''from collections.abc import Generator
def child():
    received = yield 1
    return received
def plain():
    yield from child()
def typed() -> Generator[int, None, None]:
    def nested():
        yield from child()
    nested_gen = nested()
    print(next(nested_gen), nested_gen.gi_yieldfrom.gi_code.co_name)
    nested_gen.close()
    through_lambda = lambda: (yield from child())
    lambda_gen = through_lambda()
    print(next(lambda_gen), lambda_gen.gi_yieldfrom.gi_code.co_name)
    lambda_gen.close()
    class Local:
        def method(self):
            yield from child()
    method_gen = Local().method()
    print(next(method_gen), method_gen.gi_yieldfrom.gi_code.co_name)
    method_gen.close()
    yield from child()
plain_gen = plain()
print(next(plain_gen), plain_gen.gi_yieldfrom.gi_code.co_name)
plain_gen.close()
typed_gen = typed()
print(next(typed_gen))
typed_gen.close()
''',
            'generator shutdown delegation': '''from collections.abc import Generator
def generate():
    yield from [1, 2]
item = generate()
next(item)
def typed() -> Generator[int, None, None]:
    yield from [1, 2]
typed_item = typed()
next(typed_item)
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
            'typing variadic parameters': '''import typing
from typing import ParamSpec, TypeVarTuple, Unpack
P = ParamSpec('P')
Ts = TypeVarTuple('Ts')
def collect(*args: Unpack[Ts]) -> tuple[Unpack[Ts]]: return args
def collect_qualified(*args: typing.Unpack[Ts]) -> tuple[*Ts]: return args
def forward(*args: P.args, **kwargs: P.kwargs): return args, kwargs
print(collect(1, 'x'), collect_qualified(2, 'y'), forward(1, x=2))
''',
            'callable annotations': '''from typing import Callable
from collections.abc import Callable as AbstractCallable
def apply(fn: Callable[[int], str], value: int) -> str:
    return fn(value)
def invoke(fn: AbstractCallable):
    return fn()
print(apply(str, 3), invoke(lambda: 'ok'))
''',
            'type narrowing return': '''from __future__ import annotations
from typing import TypeGuard
def is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int)
def deferred(value) -> TypeGuard[Undefined]:
    return bool(value)
print(is_int(1), is_int('x'), deferred('yes'))
''',
            'type alias annotation': '''from typing import TypeAlias
Numbers: TypeAlias = list[int]
Forward: TypeAlias = 'dict[str, int]'
numbers: Numbers = [1, 2]
mapping: Forward = {'one': 1}
print(numbers, mapping)
''',
            'generic collections': '''from collections import Counter, OrderedDict, defaultdict, deque
from typing import Deque, DefaultDict
numbers: Deque[int] = deque([1, 2])
mapping: DefaultDict[str, int] = defaultdict(int, {'x': 3})
ordered: OrderedDict[str, int] = OrderedDict([('a', 4)])
counts: Counter[str] = Counter({'a': 2})
print(list(numbers), dict(mapping), list(ordered.items()), dict(counts))
''',
            'generic regex': '''import re
from typing import Pattern, Match
pattern: Pattern[str] = re.compile('a+')
match: Match[str] | None = pattern.search('aa')
print(pattern.pattern, match.group() if match else None)
''',
            'typed streams': '''import io
from typing import AnyStr, IO, TextIO, BinaryIO
text: IO[str] = io.StringIO('alpha')
binary: BinaryIO = io.BytesIO(b'beta')
def read_text(stream: TextIO) -> str: return stream.read()
def read_any(stream: IO[AnyStr]) -> AnyStr: return stream.read()
print(read_text(text), read_any(io.StringIO('gamma')), read_any(binary))
''',
            'abstract mapping implementations': '''from collections import OrderedDict, defaultdict
from typing import Mapping, MutableMapping
def total(values: Mapping[str, int]) -> int: return sum(values.values())
def increment(values: MutableMapping[str, int]) -> int:
    values['x'] += 1
    return values['x']
ordered = OrderedDict(x=2)
defaulted = defaultdict(int, x=3)
print(total(ordered), increment(defaulted))
''',
            'abstract collections and bare aliases': '''from typing import AbstractSet, Collection, List, MutableSequence, Sequence, Tuple
def count(values: Collection[int]) -> int: return len(values)
def extend(values: MutableSequence[int]) -> list[int]:
    values.append(2)
    return list(values)
def size(values: AbstractSet[str]) -> int: return len(values)
bare: List = [1, 'x']
bare_tuple: Tuple = (1, 'x')
view: Sequence[int] = memoryview(b'A')
print(count({1: 'one'}), extend(bytearray(b'A')), size(frozenset({'x'})), bare, bare_tuple, list(view))
''',
            'hashable and sized annotations': '''from typing import Hashable, Sized
def describe(key: Hashable, values: Sized) -> tuple[int, int]:
    return hash(key), len(values)
print(describe(3, [1, 2]))
''',
            'chain map annotation': '''from collections import ChainMap
from typing import ChainMap as TypedChainMap
layers: TypedChainMap[str, int] = ChainMap({'x': 1}, {'y': 2})
print(layers['x'], layers['y'], len(layers.maps))
''',
            'mapping view annotations': '''from collections import OrderedDict
from typing import ItemsView, KeysView, MappingView, ValuesView
mapping = OrderedDict(x=1, y=2)
keys: KeysView[str] = mapping.keys()
values: ValuesView[int] = mapping.values()
items: ItemsView[str, int] = mapping.items()
view: MappingView[tuple[str, int]] = items
print(list(keys), list(values), list(items), list(view))
''',
            'container and reversible annotations': '''from collections import OrderedDict, deque
from typing import Container, Reversible
keys: Container[str] = {'x': 1, 'y': 2}
ordered: Reversible[str] = OrderedDict(x=1, y=2)
numbers: Reversible[int] = deque([1, 2, 3])
print('x' in keys, list(reversed(ordered)), list(reversed(numbers)))
''',
            'bare abstract annotations': '''from contextlib import nullcontext
from typing import ByteString, ContextManager, Generator, Iterable, Type
items: Iterable = range(2)
generator: Generator = (i for i in items)
context: ContextManager = nullcontext(3)
binary: ByteString = bytearray(b'xy')
kind: Type = int
with context as value:
    print(list(generator), value, list(binary), kind('4'))
''',
            'union class annotations': '''from typing import Annotated, Sequence, TypeVar
T = TypeVar('T', int, str)
def same(kind: type[T], value: T) -> T:
    return value
one: type[int | str] = int
none_type: type[None] = type(None)
annotated: type[Annotated[int, 'number']] = int
concrete: type[list[int]] = list
abstract: type[Sequence[int]] = list
print(one('2'), none_type(), annotated('3'), same(str, 'four'),
      concrete([1, 2]), abstract([3]))
''',
            'unpacked typed dict keyword arguments': '''from typing import NotRequired, TypedDict, Unpack
class Options(TypedDict):
    count: int
    label: NotRequired[str]
def describe(**kwargs: Unpack[Options]) -> tuple[int, str | None]:
    return kwargs['count'], kwargs.get('label')
print(describe(count=2), describe(count=3, label='ready'))
''',
            'self annotated class fields': '''from typing import Self
class Node:
    next: Self | None
    def __init__(self): self.next = None
class Child(Node): pass
head = Child()
head.next = Child()
print(type(head.next).__name__, head.next.next)
''',
            'self with renamed method receivers': '''from typing import Self
class Base:
    def clone(this) -> Self: return type(this)()
    @classmethod
    def create(klass) -> Self: return klass()
    @property
    def same(this) -> Self: return this
class Child(Base): pass
child = Child()
print(type(child.clone()).__name__, type(Child.create()).__name__, type(child.same).__name__)
''',
            'self in assigned methods': '''from typing import Self
class Base: pass
class Child(Base): pass
def clone(this) -> Self: return type(this)()
def create(klass) -> Self: return klass()
Base.clone = clone
setattr(Base, 'create', classmethod(create))
print(type(Child().clone()).__name__, type(Child.create()).__name__)
''',
        }
        if sys.version_info >= (3, 12):
            cases['generic function and mangled method parameters'] = '''def identity[T](value: T) -> T: return value
def decorate(function):
    def wrapper(*args, **kwargs): return function(*args, **kwargs)
    return wrapper
@decorate
def wrapped[T](value: T) -> T: return value
class Box[__T]:
    def pair[__U](self, left: __T, right: __U):
        return (__T, __U, left, right)
box = Box()
first, second, left, right = box.pair(1, 'x')
print(identity(3), wrapped(4), first is Box.__type_params__[0],
      second is Box.pair.__type_params__[0], left, right)
'''
            cases['variadic generic class with parameter list'] = '''class Shape[X, *Y, **Z]: pass
shape = Shape[int, str, bytes, [float, object]]()
print(type(shape).__name__, shape.__orig_class__.__args__[-1])
'''
            cases['generic variadic parameters'] = '''def collect[*Ts](*args: *Ts) -> tuple[*Ts]: return args
def mixed[*Ts](first: int, *args: *Ts) -> tuple[int, *Ts, str]:
    return (first, *args, 'done')
def forward[**P](*args: P.args, **kwargs: P.kwargs): return args, kwargs
print(collect(1, 'x'), mixed(1, 2, 3), forward(1, x=2))
'''
            cases['variadic type aliases'] = '''type TupleAlias[*Ts] = tuple[*Ts]
type Mixed[T, *Ts, U] = tuple[T, *Ts, U]
first: TupleAlias[int, str] = (1, 'x')
second: Mixed[int, str, bool] = (1, 'x', True)
print(first, second)
'''
            cases['variadic generic class'] = '''class Box[*Ts]:
    value: tuple[*Ts]
    def __init__(self, value: tuple[*Ts]): self.value = value
box: Box[int, str] = Box((1, 'x'))
class Child[*Us](Box[*Us]): pass
child: Child[int, str] = Child((2, 'y'))
print(box.value, child.value)
'''
            cases['inherited generic fields'] = '''class Base[T]:
    value: T
    def __init__(self, value): self.value = value
class Middle[U](Base[list[U]]): pass
class Leaf(Middle[int]): pass
leaf: Leaf = Leaf([1, 2])
print(leaf.value)
'''
        if sys.version_info >= (3, 13):
            cases['type is return'] = '''from typing import TypeIs
def is_int(value: object) -> TypeIs[int]:
    return isinstance(value, int)
print(is_int(1), is_int('x'))
'''
            cases['defaulted type aliases'] = '''type Pair[T, U = str] = tuple[T, U]
type Variadic[T, *Ts, U = str] = tuple[T, *Ts, U]
first: Pair[int] = (1, 'x')
second: Variadic[int] = (1, 'x')
third: Variadic[int, bool, str] = (1, True, 'x')
print(first, second, third)
'''
            cases['defaulted generic class'] = '''class Pair[T, U = str]:
    left: T
    right: U
    def __init__(self, left: T, right: U):
        self.left, self.right = left, right
pair: Pair[int] = Pair(1, 'x')
print(pair.left, pair.right)
'''
        if sys.version_info >= (3, 14):
            cases['decorated class source line'] = '''from dataclasses import dataclass
def make():
    @dataclass(slots=True)
    class Item:
        value: int
    return Item
print(make().__firstlineno__ - make.__code__.co_firstlineno)
'''
            cases['template string interpolation'] = '''events = []
def pick():
    events.append('called')
    return 3
template = t"value {pick()!r:>8}"
print(template.strings, [(item.value, item.expression, item.conversion, item.format_spec)
                         for item in template.interpolations], events)
'''
            cases['deferred annotations and type defaults'] = '''def future(value: Missing) -> Unknown: return value
class Box[T = int]:
    item: T
type Alias = list[Missing]
print(Box.__type_params__[0].__default__, Alias.__name__,
      '__annotate__' in Box.__dict__, '__annotate__' in dir(future))
for target in (future, Box):
    try: print(target.__annotations__)
    except NameError as error: print(type(error).__name__)
'''
            cases['unparenthesized exception tuple'] = '''try:
    raise ValueError('bad')
except ValueError, TypeError:
    print('caught')
'''
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
