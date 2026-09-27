"""Interactive cells preserve Python state and Aiython boundaries."""

import builtins
import contextlib
import io
from pathlib import Path
import unittest

from aiython.models import ProfileConfig, ResolvedConfig
from aiython.repl import AiythonConsole
from aiython.runtime import Runtime
from aiython.typed_runtime import Scope


class InteractiveConsoleTests(unittest.TestCase):
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
