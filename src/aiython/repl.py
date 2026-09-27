"""Interactive Python cells compiled through the Aiython runtime."""

import __future__
import ast
import code
import sys

from .frontend import parse
from .runtime import bind_runtime
from .typed_runtime import helper

FUTURE_FLAGS = sum(getattr(__future__, name).compiler_flag
                   for name in __future__.all_feature_names)


class AiythonConsole(code.InteractiveConsole):
    def __init__(self, runtime, namespace):
        super().__init__(locals=namespace, filename='<stdin>')
        self.runtime = runtime
        self.cell_number = 0

    def raw_input(self, prompt=''):
        sys.stderr.write(prompt)
        sys.stderr.flush()
        return input()

    def runsource(self, source, filename='<stdin>', symbol='single'):
        cell_filename = f'<stdin:{self.cell_number + 1}>'
        unit = None
        try:
            complete = self.compile(source, cell_filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            try:
                unit = parse(source, cell_filename)
                if not unit.blocks:
                    raise
                complete = self.compile(unit.transformed, cell_filename, symbol)
            except (OverflowError, SyntaxError, ValueError):
                self.showsyntaxerror(cell_filename)
                return False
            except SystemExit:
                raise
            except BaseException:
                self.showtraceback()
                return False
        if complete is None:
            return True
        try:
            unit = unit or parse(source, cell_filename)
            if unit.tree.body and isinstance(unit.tree.body[-1], ast.Expr):
                expression = unit.tree.body[-1]
                expression.value = helper('display', expression.value,
                                          runtime_name=unit.runtime_name)
            flags = self.compile.compiler.flags & FUTURE_FLAGS
            compiled = bind_runtime(self.runtime.prepare(unit, entry=True, flags=flags),
                                    unit.runtime_name, self.runtime)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(cell_filename)
            return False
        except SystemExit:
            raise
        except BaseException:
            self.showtraceback()
            return False
        self.cell_number += 1
        self.runcode(compiled)
        return False
