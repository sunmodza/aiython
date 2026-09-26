import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiython.agent import ToolAgent
from aiython.capabilities import CapabilityRuntime
from aiython.cli import run_script
from aiython.config import resolve
from aiython.config_hints import append_missing_route_example, example, setup_command
from aiython.models import ConfigError, ProfileConfig, ProviderError, ResolvedConfig


class ConfigHintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'aiython.toml'
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
                with self.assertRaisesRegex(ConfigError, 'aiython setup --capability speech_to_text'):
                    runtime.routes(profile, 'speech_to_text')
        updated = self.path.read_text()
        self.assertTrue(updated.startswith(original))
        self.assertEqual(updated.count('# >>> Aiython missing route:'), 1)
        self.assertIn('# [capabilities]', updated)
        self.assertIn('# speech_to_text = "openai/YOUR_TRANSCRIPTION_MODEL"', updated)
        self.assertEqual(tomllib.loads(updated)['model'], 'openai/test')
        self.assertEqual(output.getvalue().count('added commented'), 1)
        self.assertIn('aiython setup --capability speech_to_text', output.getvalue())

    def test_profile_hint_uses_profile_table(self):
        self.path.write_text(self.path.read_text() + '[profiles.fast]\nmodel="openai/fast"\n')
        config = resolve(self.script)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'aiython setup --capability image_generation --profile fast'):
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
            with self.assertRaisesRegex(ProviderError, 'aiython setup --capability text_to_speech'):
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

    def test_invalid_or_linked_config_is_never_modified(self):
        config = resolve(self.script)
        profile = config.profiles['default']
        self.assertIsNone(append_missing_route_example(ResolvedConfig(None, self.root),
                                                       profile, 'embedding'))
        linked = self.root / 'linked.toml'
        linked.symlink_to(self.path)
        self.assertIsNone(append_missing_route_example(
            ResolvedConfig(linked, self.root), profile, 'embedding'))
        for source in ('version=2\nmodel="openai/test"\n', 'not valid TOML = ['):
            with self.subTest(source=source):
                self.path.write_text(source)
                self.assertIsNone(append_missing_route_example(config, profile, 'embedding'))
                self.assertEqual(self.path.read_text(), source)
        self.path.write_text('version=3\nmodel="openai/test"\n')
        with patch('aiython.config_hints.stat.S_ISREG', return_value=False):
            self.assertIsNone(append_missing_route_example(config, profile, 'embedding'))

    def test_video_hint_quotes_profile_and_preserves_last_line(self):
        self.path.write_text('version=3\nmodel="openai/test"')
        profile = ProfileConfig('video profile', 'fake', 'model')
        config = ResolvedConfig(self.path, self.root)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(append_missing_route_example(config, profile, 'video'), self.path)
        updated = self.path.read_text()
        self.assertIn('model="openai/test"\n\n# >>>', updated)
        self.assertIn('# video = { understand = ', updated)
        self.assertIn('# [profiles."video profile".capabilities]', example(profile, 'video'))
        self.assertIn("--profile 'video profile'", setup_command(profile, 'video'))

    def test_hint_works_on_platform_without_nofollow_flag(self):
        config = resolve(self.script)
        portable_os = SimpleNamespace(O_WRONLY=os.O_WRONLY, O_APPEND=os.O_APPEND,
                                      open=os.open, fdopen=os.fdopen)
        with patch('aiython.config_hints.os', portable_os), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(append_missing_route_example(
                config, config.profiles['default'], 'embedding'), self.path)
        self.assertIn('# embedding = ', self.path.read_text())
