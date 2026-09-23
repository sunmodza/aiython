import os
import tempfile
import unittest
from pathlib import Path

from aithon.cli import run_script
from aithon.config import resolve


@unittest.skipUnless(os.environ.get("AITHON_INTEGRATION_CONFIG"), "live provider test is opt-in")
class LiveProviderTests(unittest.TestCase):
    def test_live_syntax_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "main.py"
            script.write_text("x = 10\ny = 20\nresult = add x and y please\n")
            config = resolve(script, config_path=os.environ["AITHON_INTEGRATION_CONFIG"])
            result = run_script(script, config=config)
            self.assertEqual(result["result"], 30)
