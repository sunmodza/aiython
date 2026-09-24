import contextlib
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import Mock

from aithon.agent import ToolAgent
from aithon.capabilities import CapabilityRuntime
from aithon.cli import run_script
from aithon.config import resolve
from aithon.models import ConfigError, ProviderError, ResolvedConfig


class ConfigHintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'aithon.toml'
        self.path.write_text('version=3\nmodel="openai/test"\n')
        self.script = self.root / 'main.py'
        self.script.write_text('speech = speak the answer aloud\n')

    def test_missing_route_adds_valid_commented_v3_example_once(self):
        original = self.path.read_text()
        config = resolve(self.script)
        runtime = CapabilityRuntime(config)
        profile = config.profiles['default']
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            for _ in range(2):
                with self.assertRaisesRegex(ConfigError, 'aithon setup --capability speech_to_text'):
                    runtime.routes(profile, 'speech_to_text')
        updated = self.path.read_text()
        self.assertTrue(updated.startswith(original))
        self.assertEqual(updated.count('# >>> Aithon missing route:'), 1)
        self.assertIn('# [capabilities]', updated)
        self.assertIn('# speech_to_text = "openai/YOUR_TRANSCRIPTION_MODEL"', updated)
        self.assertEqual(tomllib.loads(updated)['model'], 'openai/test')
        self.assertEqual(output.getvalue().count('added commented'), 1)
        self.assertIn('aithon setup --capability speech_to_text', output.getvalue())

    def test_profile_hint_uses_profile_table(self):
        self.path.write_text(self.path.read_text() + '[profiles.fast]\nmodel="openai/fast"\n')
        config = resolve(self.script)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'aithon setup --capability image_generation --profile fast'):
                CapabilityRuntime(config).routes(config.profiles['fast'], 'image_generation')
        self.assertIn('# [profiles."fast".capabilities]', self.path.read_text())

    def test_terminal_missing_route_adds_example_before_stopping(self):
        config = resolve(self.script)
        provider = Mock()
        provider.complete.return_value = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'stop', 'type': 'function', 'function': {'name': 'finish', 'arguments': json.dumps({
                'outcome': {'kind': 'error', 'reason': 'No text_to_speech route is configured',
                            'missing_capability': 'text_to_speech'}})}}]}
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ProviderError, 'aithon setup --capability text_to_speech'):
                run_script(self.script, config=config, agent_factory=lambda _: ToolAgent(provider))
        self.assertEqual(provider.complete.call_count, 1)
        self.assertIn('# text_to_speech = ', self.path.read_text())

    def test_other_project_config_is_not_modified(self):
        config = resolve(self.script)
        original = self.path.read_text()
        elsewhere = ResolvedConfig(self.path, self.root / 'different-project',
                                   config.default_profile, config.profiles)
        with self.assertRaisesRegex(ConfigError, 'No route for capability'):
            CapabilityRuntime(elsewhere).routes(config.profiles['default'], 'speech_to_text')
        self.assertEqual(self.path.read_text(), original)
