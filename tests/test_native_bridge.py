"""Bridge contracts that also run without a patched CPython interpreter."""

import sys
import unittest

from aiython.native_bridge import NativeTypeBridge
from aiython.type_constraints import TypeViolation


class NativeBridgeTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(sys, 'monitoring'), 'requires Python 3.12+')
    def test_yield_monitor_rejects_value_and_runs_generator_finally(self):
        source = '''from typing import Generator
cleanup = []
def values() -> Generator[int, None, None]:
    try:
        yield from (1, "invalid")
    finally:
        cleanup.append("closed")
'''
        filename = '<bridge-yield-monitor>'
        bridge = NativeTypeBridge()
        code = bridge.compile_source(source, filename)
        self.assertEqual(code, compile(source, filename, 'exec', dont_inherit=True))
        namespace = {}
        before = [sys.monitoring.get_tool(number) for number in range(6)]
        with bridge._yield_monitor():
            exec(code, namespace)
            with self.assertRaisesRegex(TypeViolation, 'yield'):
                list(namespace['values']())
        self.assertEqual(namespace['cleanup'], ['closed'])
        self.assertEqual([sys.monitoring.get_tool(number) for number in range(6)], before)
