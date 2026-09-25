import os
import tempfile
import unittest
from pathlib import Path

from aiython.cli import run_script
from aiython.config import resolve


@unittest.skipUnless(os.environ.get("AIYTHON_INTEGRATION_CONFIG"), "live provider test is opt-in")
class LiveProviderTests(unittest.TestCase):
    def test_live_basic_demo_mutates_existing_and_creates_new_bindings(self):
        script = Path(__file__).resolve().parents[1] / 'examples' / 'basic_demo.py'
        config = resolve(script, config_path=os.environ['AIYTHON_INTEGRATION_CONFIG'])
        result = run_script(script, config=config)
        self.assertEqual(tuple(result[key] for key in ('x', 'y', 'z')), (13, 20, 33))

    def test_live_syntax_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "main.py"
            script.write_text("x = 10\ny = 20\nresult = add x and y please\n")
            config = resolve(script, config_path=os.environ["AIYTHON_INTEGRATION_CONFIG"])
            result = run_script(script, config=config)
            self.assertEqual(result["result"], 30)
