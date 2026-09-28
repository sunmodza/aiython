"""Interactive cells preserve Python state and Aiython boundaries."""

import builtins
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from aiython.models import ProfileConfig, ResolvedConfig
from aiython.repl import AiythonConsole
from aiython.runtime import Runtime
from aiython.typed_runtime import Scope


class InteractiveConsoleTests(unittest.TestCase):
    def test_syntax_and_setup_errors_are_reported_without_advancing_cell(self):
        with patch.object(self.console, 'compile', side_effect=SyntaxError('bad syntax')):
            with patch.object(self.console, 'showsyntaxerror') as report:
                self.assertFalse(self.console.runsource('value = 1'))
                report.assert_called_once_with('<stdin:1>')
        with patch.object(self.console, 'compile', side_effect=SyntaxError('bad AI')):
            with patch('aiython.repl.parse', side_effect=SyntaxError('bad AI')):
                with patch.object(self.console, 'showsyntaxerror') as report:
                    self.assertFalse(self.console.runsource('choose'))
                    report.assert_called_once_with('<stdin:1>')
        with patch.object(self.console, 'compile', side_effect=SyntaxError('bad AI')):
            with patch('aiython.repl.parse', side_effect=RuntimeError('parser failed')):
                with patch.object(self.console, 'showtraceback') as report:
                    self.assertFalse(self.console.runsource('choose'))
                    report.assert_called_once_with()
        with patch.object(self.console, 'compile', side_effect=SyntaxError('bad AI')):
            with patch('aiython.repl.parse', side_effect=SystemExit(3)):
                with self.assertRaises(SystemExit):
                    self.console.runsource('choose')
        for error, reporter in ((SyntaxError('bad transform'), 'showsyntaxerror'),
                                (RuntimeError('compiler failed'), 'showtraceback')):
            with self.subTest(error=error):
                with patch.object(self.runtime.bridge, 'prepare_unit', side_effect=error):
                    with patch.object(self.console, reporter) as report:
                        self.assertFalse(self.console.runsource('1 + 2'))
                        if reporter == 'showsyntaxerror':
                            report.assert_called_once_with('<stdin:1>')
                        else:
                            report.assert_called_once_with()
        with patch.object(self.runtime.bridge, 'prepare_unit', side_effect=SystemExit(4)):
            with self.assertRaises(SystemExit):
                self.console.runsource('1 + 2')
        self.assertEqual(self.console.cell_number, 0)

    def test_native_compiled_cell_runs_without_runtime_binding(self):
        code = compile('answer = 3', '<stdin:1>', 'exec')
        with patch.object(self.runtime.bridge, 'prepare_unit', return_value=(code, True)):
            with patch('aiython.repl.bind_runtime') as bind:
                self.assertFalse(self.console.runsource('answer = 3'))
                bind.assert_not_called()
        self.assertEqual(self.namespace['answer'], 3)

    def setUp(self):
        self.had_underscore = '_' in vars(builtins)
        self.original_underscore = vars(builtins).get('_')
        class Agent:
            def execute(self, request, bridge):
                return 7

        root = Path.cwd()
        config = ResolvedConfig(None, root, 'default',
                                {'default': ProfileConfig('default', 'fake', 'model')})
        self.runtime = Runtime(config, agent_factory=lambda _: Agent())
        self.namespace = {'__name__': '__main__', '__builtins__': builtins}
        self.runtime.types.interactive_globals = self.namespace
        self.runtime.types.interactive_scope = Scope()
        self.console = AiythonConsole(self.runtime, self.namespace)

    def tearDown(self):
        self.runtime.capabilities.close()
        if self.had_underscore:
            builtins._ = self.original_underscore
        else:
            vars(builtins).pop('_', None)

    def test_display_keeps_native_cell_tree(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(self.console.runsource('1 + 2'))
        self.assertEqual(output.getvalue(), '3\n')
        filename = '<stdin:1>'
        source_tree = self.runtime.units[filename].tree
        self.assertEqual(compile(source_tree, filename, 'exec', dont_inherit=True),
                         compile('1 + 2', filename, 'exec', dont_inherit=True))

    def test_compound_input_waits_and_displays_last_expression(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(self.console.runsource('for number in range(2):'))
            self.assertTrue(self.console.runsource('for number in range(2):\n    print(number)'))
            self.assertFalse(self.console.runsource('for number in range(2):\n    print(number)\n'))
            self.assertFalse(self.console.runsource('number'))
            self.assertFalse(self.console.runsource('_ + 1'))
        self.assertEqual(output.getvalue(), '0\n1\n1\n2\n')

    def test_ai_syntax_and_type_contract_persist_across_cells(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertFalse(self.console.runsource('answer: int = choose seven'))
            self.assertFalse(self.console.runsource('answer'))
            self.assertFalse(self.console.runsource('answer = "wrong"'))
            self.assertFalse(self.console.runsource('answer'))
        self.assertEqual(output.getvalue(), '7\n7\n')
        self.assertIn('TypeViolation', errors.getvalue())

    def test_ai_syntax_in_compound_input_waits_for_blank_line(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(self.console.runsource('for number in range(2):\n'
                                                   '    answer: int = choose seven'))
            self.assertFalse(self.console.runsource('for number in range(2):\n'
                                                    '    answer: int = choose seven\n'))
            self.assertFalse(self.console.runsource('answer'))
        self.assertEqual(output.getvalue(), '7\n')

    def test_future_annotations_apply_to_later_cells(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertFalse(self.console.runsource('from __future__ import annotations'))
            self.assertFalse(self.console.runsource('def later(value: Missing):\n'
                                                    '    return value\n'))
        self.assertEqual(self.namespace['later'].__annotations__['value'], 'Missing')
        self.assertEqual(errors.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
