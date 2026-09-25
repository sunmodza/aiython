"""Programs outside the bundled examples, checked by observable effects."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from aiython.agent import ToolAgent
from aiython.cli import run_script
from aiython.config import resolve
from aiython.models import ProfileConfig, ResolvedConfig


class HeldOutPrograms(unittest.TestCase):
    def test_ordinary_fibonacci_never_imports_provider_sdks(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'plain.py'
            script.write_text('''import sys
left, right = 0, 1
for _ in range(32):
    left, right = right, left + right
assert left == 2178309
assert "litellm" not in sys.modules
assert "openrouter" not in sys.modules
print(left)
''')
            result = subprocess.run([sys.executable, '-m', 'aiython', str(script)],
                                    capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout), (0, '2178309\n'), result.stderr)

    def test_python_owns_iteration_and_side_effect_order(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'program.py'
            script.write_text('''observed = []
def record(value):
    observed.append(value)
product = 1
for n in range(1, 7):
    factor: int = choose the current factor from n
    product *= factor
    record(factor)
''')
            provider = Mock()
            provider.complete.side_effect = [
                {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': f'return-{n}', 'type': 'function', 'function': {'name': 'finish',
                    'arguments': json.dumps({'outcome': {'kind': 'expression', 'code': 'n'}})}}]}
                for n in range(1, 7)]
            profile = ProfileConfig('default', 'fake', 'model')
            result = run_script(script, config=ResolvedConfig(None, script.parent, 'default', {'default': profile}),
                                agent_factory=lambda _: ToolAgent(provider))
        self.assertEqual(result['product'], 720)
        self.assertEqual(result['observed'], [1, 2, 3, 4, 5, 6])
        self.assertEqual(provider.complete.call_count, 6)

    def test_v3_config_to_litellm_to_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'aiython.toml').write_text('version=3\nmodel="openai/test"\n')
            script = root / 'program.py'
            script.write_text('left = 8\nright = 13\nanswer: int = add the two numbers\n')
            sdk = Mock(completion=Mock(return_value={'choices': [{'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'done', 'type': 'function',
                'function': {'name': 'finish', 'arguments': json.dumps({
                    'outcome': {'kind': 'expression', 'code': 'left + right'}})}}]}}]}))
            with patch('aiython.providers.sdk', return_value=sdk):
                result = run_script(script, config=resolve(script))
        self.assertEqual(result['answer'], 21)
        self.assertEqual(sdk.completion.call_count, 1)
