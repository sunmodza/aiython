import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiython.config import credential, describe, resolve
from aiython.models import ConfigError


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'aiython.toml'

    def test_default_and_capability_routes(self):
        self.path.write_text('''version = 3
model = "openrouter/vendor/reasoner"
api_key_env = "TEST_AI_KEY"
[capabilities]
embedding = "openai/text-embedding-3-small"
video = { understand = "gemini/understand", generate = "gemini/veo" }
[profiles.fast]
model = "openai/fast"
[profiles.fast.capabilities]
reranking = { model = "cohere/rerank", api_key_env = "COHERE_KEY" }
''')
        with patch.dict(os.environ, {'TEST_AI_KEY': 'secret'}):
            config = resolve(self.root / 'main.py')
            self.assertEqual(credential(config, config.profiles['default']), 'secret')
        self.assertEqual(config.profiles['default'].model, 'openrouter/vendor/reasoner')
        self.assertEqual(config.profiles['fast'].routes['reasoning'][0]['model'], 'openai/fast')
        self.assertEqual(config.profiles['default'].routes['video'][1]['modes'], ['generate'])
        self.assertEqual(config.profiles['fast'].routes['reranking'][0]['model'], 'cohere/rerank')
        self.assertNotIn('secret', str(describe(config)))

    def test_project_scoped_credentials_and_process_override(self):
        self.path.write_text('version=3\nmodel="openai/test"\nenv_file=".aiython/credentials.env"\napi_key_env="TEST_AI_KEY"\n')
        (self.root / '.aiython').mkdir()
        (self.root / '.aiython/credentials.env').write_text('TEST_AI_KEY="project-key"\n')
        config = resolve(self.root / 'main.py')
        self.assertEqual(credential(config, config.profiles['default']), 'project-key')
        with patch.dict(os.environ, {'TEST_AI_KEY': 'environment-key'}):
            self.assertEqual(credential(config, config.profiles['default']), 'environment-key')
        self.assertNotIn('project-key', repr(config))
        self.assertNotIn('project-key', str(describe(config)))

    def test_legacy_config_rejected_without_modification(self):
        for version in (1, 2):
            self.path.write_text(f'version={version}\nmodel="openai/test"\n')
            source = self.path.read_text()
            with self.assertRaisesRegex(ConfigError, 'version 1/2.*version 3'):
                resolve(self.root / 'main.py')
            self.assertEqual(self.path.read_text(), source)

    def test_invalid_schema_and_endpoint_are_redacted(self):
        for source in (
            'version=3\nmodel="openai/test"\napi_key_env="invalid-key"\n',
            'version=3\nmodel="openai/test"\napi_base="https://secret@example.test/"\n',
            'version=3\nmodel="openai/test"\nunknown="private"\n',
        ):
            self.path.write_text(source)
            with self.assertRaises(ConfigError) as caught:
                resolve(self.root / 'main.py')
            self.assertNotIn('secret@example', str(caught.exception))

    def test_config_discovery_and_selection(self):
        self.path.write_text('version=3\nmodel="openai/default"\n[profiles.fast]\nmodel="openai/fast"\n')
        nested = self.root / 'nested'
        nested.mkdir()
        self.assertEqual(resolve(nested / 'main.py').profiles['default'].model, 'openai/default')
        self.assertEqual(resolve(nested / 'main.py', profile='fast').cli_profile, 'fast')
        with self.assertRaises(ConfigError):
            resolve(nested / 'main.py', profile='fast', force_profile='default')
        with self.assertRaises(ConfigError):
            resolve(nested / 'main.py', config_path=str(self.root / 'missing.toml'))

    def test_example_directory_inherits_nearest_parent_config(self):
        self.path.write_text('version=3\nmodel="openrouter/test"\napi_key_env="TEST_AI_KEY"\n')
        example = self.root / 'examples'
        example.mkdir()
        selected = resolve(example / 'demo.py')
        self.assertEqual(selected.path, self.path)
        self.assertEqual(selected.project_root, self.root)
        (example / 'aiython.toml').write_text('version=3\nmodel="openrouter/other"\n')
        self.assertEqual(resolve(example / 'demo.py').path, example / 'aiython.toml')
