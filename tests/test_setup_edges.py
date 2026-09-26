"""Setup validation and safe configuration writes."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import tomlkit

from aiython.models import ConfigError, ProfileConfig
from aiython.setup import (
    ModelChoice, _ask, _catalog, _catalog_matches, _check_model, _choice,
    _choose_model, _current_key, _effective_key, _ignore_credentials, _missing_hints,
    _model, _model_id, _path, _provider, _route_fields, _route_value,
    _select_capability_route, _configure_capability, parser,
    _strip_missing_hint, _valid_url, _validated_content, _write_config,
    _write_keys, setup,
)


class SetupEdgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'aiython.toml'

    def test_prompts_model_ids_and_urls_validate_input(self):
        with patch('builtins.input', return_value=''):
            self.assertEqual(_ask('Model', 'openai/test'), 'openai/test')
        with patch('builtins.input', side_effect=EOFError):
            with self.assertRaisesRegex(ConfigError, 'Missing model'):
                _ask('Model')
        with patch('prompt_toolkit.shortcuts.choice', return_value='openai') as chooser:
            self.assertEqual(_choice('Provider', [('openai', 'OpenAI')], 'openai'), 'openai')
        chooser.assert_called_once()
        self.assertFalse(_valid_url('http://['))
        self.assertFalse(_valid_url('https://user:secret@example.test'))
        self.assertTrue(_valid_url('http://localhost:11434/v1'))
        with self.assertRaisesRegex(ConfigError, 'Model ID is required'):
            _model_id('openai', 'bad id')
        self.assertEqual(_model_id('openai', 'test'), 'openai/test')
        self.assertEqual(_provider('unknown/test', ''), 'other')
        self.assertEqual(_provider('unknown/test', 'http://localhost'), 'custom')

    def test_exact_config_path_rejects_wrong_name_missing_directory_and_symlink(self):
        for path in (self.root / 'other.toml', self.root / 'missing' / 'aiython.toml'):
            with self.subTest(path=path), self.assertRaisesRegex(ConfigError, 'existing project directory'):
                _path(argparse.Namespace(path=path))
        original = self.root / 'original.toml'
        original.write_text('private')
        self.path.symlink_to(original)
        with self.assertRaisesRegex(ConfigError, 'symlinked'):
            _path(argparse.Namespace(path=self.path))

    def test_current_model_and_route_values_accept_inline_tables(self):
        self.assertEqual(_model({}), '')
        self.assertEqual(_model({'model': 'openai/test'}), 'openai/test')
        document = {'model': {'model': 'openai/test', 'api_base': 'http://localhost'}}
        self.assertEqual(_model(document), 'openai/test')
        self.assertEqual(_route_value(document, 'api_base'), 'http://localhost')
        self.assertEqual(_route_value({}, 'api_key_env'), '')
        self.assertEqual(_route_fields([]), ('', '', ''))
        self.assertEqual(_route_fields([{'model': 'openai/test', 'api_key_env': 'KEY'}]),
                         ('openai/test', '', 'KEY'))
        self.assertEqual(_route_fields('openai/test'), ('openai/test', '', ''))

    def test_catalog_filters_incompatible_models_and_handles_incomplete_metadata(self):
        item = {'architecture': {'input_modalities': ['text'], 'output_modalities': ['text']},
                'supported_parameters': ['temperature']}
        self.assertFalse(_catalog_matches('openrouter', item, 'reasoning', None))
        self.assertFalse(_catalog_matches('openrouter', item, 'vision', None))
        self.assertTrue(_catalog_matches('openrouter', {}, 'reasoning', None))
        self.assertTrue(_catalog_matches('openrouter', item, 'not-listed', None))
        image = {'architecture': {'input_modalities': ['text'], 'output_modalities': ['image']}}
        self.assertFalse(_catalog_matches('openrouter', image, 'image_generation', None))
        self.assertFalse(_catalog_matches('gemini', {'supportedGenerationMethods': ['embedContent']},
                                          'reasoning', None))
        self.assertTrue(_catalog_matches('gemini', {'supportedGenerationMethods': ['embedContent']},
                                         'embedding', None))
        self.assertTrue(_catalog_matches('gemini', {'supportedGenerationMethods': ['generateContent']},
                                         'image_generation', None))
        self.assertTrue(_catalog_matches('gemini', {}, 'reasoning', None))

    def test_catalog_openai_gemini_and_custom_routes_are_optional(self):
        response = Mock()
        response.json.return_value = {'data': [{'id': 'model-a', 'name': 'First'}, {'id': 42}, None]}
        with patch('aiython.setup.httpx.get', return_value=response) as request:
            self.assertEqual([choice.id for choice in _catalog('openai', '', 'key')], ['openai/model-a'])
        self.assertEqual(request.call_args.args[0], 'https://api.openai.com/v1/models')
        response.json.return_value = {'models': [
            {'name': 'models/gemini-test', 'displayName': 'Gemini Test',
             'supportedGenerationMethods': ['generateContent']},
        ]}
        with patch('aiython.setup.httpx.get', return_value=response) as request:
            choices = _catalog('gemini', '', 'key')
        self.assertEqual(choices[0], ModelChoice('gemini/gemini-test', 'Gemini Test'))
        self.assertEqual(request.call_args.kwargs['headers'], {'x-goog-api-key': 'key'})
        response.json.return_value = {'data': [{'id': 'local-model'}]}
        with patch('aiython.setup.httpx.get', return_value=response) as request:
            self.assertEqual(_catalog('custom', 'http://localhost:11434/v1/', None)[0].id,
                             'openai/local-model')
        self.assertEqual(request.call_args.args[0], 'http://localhost:11434/v1/models')
        self.assertEqual(_catalog('openai', '', None), [])
        self.assertEqual(_catalog('other', '', None), [])
        with patch('aiython.setup.httpx.get', side_effect=httpx.ConnectError('offline')):
            self.assertEqual(_catalog('openai', '', 'key'), [])

    def test_model_search_can_be_cancelled_or_entered_manually(self):
        with patch('aiython.setup._catalog', return_value=[ModelChoice('openai/test')]), \
                patch('prompt_toolkit.prompt', side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'Model ID is required'):
                _choose_model('openai', '', '', None)
        with patch('aiython.setup._catalog', return_value=[]), \
                patch('aiython.setup._ask', return_value='manual-model'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_choose_model('openai', '', '', None), 'manual-model')

    def test_key_lookup_respects_environment_and_confined_project_file(self):
        document = {'env_file': '.aiython/credentials.env'}
        key_path = self.root / '.aiython' / 'credentials.env'
        key_path.parent.mkdir()
        key_path.write_text('KEY="saved"\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_current_key(self.path, document, 'KEY'), 'saved')
            self.assertEqual(_effective_key(self.path, document, 'KEY', 'new'), 'new')
        with patch.dict(os.environ, {'KEY': 'ambient'}):
            self.assertEqual(_current_key(self.path, document, 'KEY'), 'ambient')
            self.assertEqual(_effective_key(self.path, document, 'KEY', 'new'), 'ambient')
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_current_key(self.path, {'env_file': '../outside.env'}, 'KEY'))
            self.assertIsNone(_current_key(self.path, {}, ''))

    def test_credential_writes_reject_symlinks_and_nonfiles(self):
        outside = self.root.parent / 'outside-credentials.env'
        with self.assertRaisesRegex(ConfigError, 'outside the project'):
            _write_keys(self.root, '../outside-credentials.env', {'KEY': 'private'})
        directory = self.root / '.aiython'
        directory.mkdir()
        (directory / 'credentials.env').symlink_to(outside)
        with self.assertRaisesRegex(ConfigError, 'symlink'):
            _write_keys(self.root, '.aiython/credentials.env', {'KEY': 'private'})
        (directory / 'credentials.env').unlink()
        (directory / 'credentials.env').mkdir()
        with self.assertRaisesRegex(ConfigError, 'not a file'):
            _write_keys(self.root, '.aiython/credentials.env', {'KEY': 'private'})
        (directory / 'credentials.env').rmdir()
        ignore = self.root / '.gitignore'
        ignore.symlink_to(outside)
        with self.assertRaisesRegex(ConfigError, 'symlinked .gitignore'):
            _ignore_credentials(self.root, '.aiython/credentials.env')
        ignore.unlink()
        real = self.root / 'real'
        real.mkdir()
        (self.root / 'link').symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ConfigError, 'symlinked directory'):
            _write_keys(self.root, 'link/credentials.env', {'KEY': 'private'})
        with patch.object(Path, 'open', side_effect=OSError('denied')):
            with self.assertRaisesRegex(ConfigError, 'Cannot update .gitignore'):
                _ignore_credentials(self.root, '.aiython/credentials.env')

    def test_failed_config_and_credential_writes_remove_temp_files(self):
        with patch('aiython.setup.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(ConfigError, 'Cannot save API key'):
                _write_keys(self.root, '.aiython/credentials.env', {'KEY': 'private'})
        self.assertFalse((self.root / '.aiython' / 'credentials.env').exists())
        self.assertEqual(list((self.root / '.aiython').glob('.credentials-*')), [])
        with patch('aiython.setup.os.link', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(ConfigError, 'Cannot save'):
                _write_config(self.path, 'version=3\nmodel="openai/test"\n', existing=False)
        self.assertFalse(self.path.exists())

    def test_config_write_rechecks_symlink_and_validation(self):
        document = tomlkit.parse('version=2\nmodel="openai/test"\n')
        with self.assertRaisesRegex(ConfigError, 'Invalid setup configuration'):
            _validated_content(document)
        original = self.root / 'original.toml'
        original.write_text('version=3\nmodel="openai/test"\n')
        self.path.symlink_to(original)
        with self.assertRaisesRegex(ConfigError, 'changed during setup'):
            _write_config(self.path, 'version=3\nmodel="openai/new"\n', existing=True)
        self.assertEqual(original.read_text(), 'version=3\nmodel="openai/test"\n')

    def test_check_model_handles_credential_provider_and_tool_failures(self):
        with self.assertRaisesRegex(ConfigError, 'Missing credential'):
            _check_model('openai/test', '', 'KEY', None)
        failure = RuntimeError('private body')
        failure.status_code = 403
        with patch('aiython.providers.sdk', return_value=SimpleNamespace(
                completion=Mock(side_effect=failure))), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'access denied') as caught:
                _check_model('openai/test', '', '', None)
        self.assertNotIn('private body', str(caught.exception))
        with patch('aiython.providers.sdk', return_value=SimpleNamespace(
                completion=Mock(return_value={'choices': [{'message': {'tool_calls': []}}]}))), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'did not call a tool'):
                _check_model('openai/test', '', '', None)
        good = {'choices': [{'message': {'tool_calls': [
            {'function': {'name': 'aiython_probe'}},
        ]}}]}
        with patch('aiython.providers.sdk', return_value=SimpleNamespace(completion=Mock(return_value=good))), \
                contextlib.redirect_stdout(io.StringIO()):
            _check_model('openai/test', '', '', None)

    def test_missing_route_comment_is_removed_only_when_complete(self):
        marker = '# >>> Aiython missing route: profile="default" capability=embedding'
        self.assertEqual(_strip_missing_hint('plain\n', 'default', 'embedding'), 'plain\n')
        self.assertEqual(_strip_missing_hint(marker, 'default', 'embedding'), marker)
        content = marker + '\n# example\n# <<< Aiython missing route\nrest\n'
        self.assertEqual(_strip_missing_hint(content, 'default', 'embedding'), 'rest\n')
        document = tomlkit.parse('version=3\nmodel="openai/test"\n' +
                                 marker + '\n# example\n# <<< Aiython missing route\n')
        self.assertEqual(_missing_hints(document), [('default', 'embedding')])
        malformed = tomlkit.parse('version=3\nmodel="openai/test"\n'
                                  '# >>> Aiython missing route: profile=bad capability=embedding\n'
                                  '# >>> Aiython missing route: profile="default" capability=unknown\n')
        self.assertEqual(_missing_hints(malformed), [])
        configured = tomlkit.parse('version=3\nmodel="openai/test"\n[capabilities]\n'
                                   'embedding="openai/embedding"\n' + marker + '\n')
        self.assertEqual(_missing_hints(configured), [])

    def test_capability_route_provider_and_key_validation(self):
        profile = ProfileConfig('default', 'litellm', 'openai/test', base_url='')
        document = tomlkit.parse('version=3\nmodel="openai/test"\n')
        def choose(*, provider=None, base_url=None, key_env=None, set_key=False,
                   current=None, model='openai/embedding', interactive=False,
                   profile_config=profile, new_keys=None):
            args = SimpleNamespace(provider=provider, base_url=base_url,
                                   api_key_env=key_env, set_key=set_key)
            keys = {} if new_keys is None else new_keys
            return _select_capability_route(args, self.path, document, profile_config,
                                            current, 'Embedding', model, interactive, keys,
                                            capability='embedding')

        self.assertEqual(str(choose(model='anthropic/embed')['model']), 'anthropic/embed')
        custom = ProfileConfig('default', 'litellm', 'openai/local',
                               base_url='http://localhost:11434/v1')
        route = choose(provider='custom', model='local', profile_config=custom)
        self.assertEqual(str(route['api_base']), 'http://localhost:11434/v1')
        self.assertNotIn('api_key_env', route)
        with self.assertRaisesRegex(ConfigError, 'requires --base-url'):
            choose(provider='custom', model='local')
        with patch('aiython.setup._ask', side_effect=['http://localhost:11434/v1', '']):
            self.assertEqual(str(choose(provider='custom', model='local', interactive=True)['api_base']),
                             'http://localhost:11434/v1')
        with self.assertRaisesRegex(ConfigError, 'API base URL'):
            choose(provider='custom', base_url='http://[', model='local')
        with self.assertRaisesRegex(ConfigError, 'valid name'):
            choose(key_env='bad-name')
        with self.assertRaisesRegex(ConfigError, 'requires an API key'):
            choose(provider='openai', key_env='')
        with self.assertRaisesRegex(ConfigError, '--set-key requires'):
            choose(provider='other', key_env='', set_key=True)
        with patch('aiython.setup._ask', return_value='CUSTOM_KEY'), \
                patch('aiython.setup.getpass.getpass', return_value=''):
            route = choose(provider='custom', base_url='http://localhost:11434/v1',
                           model='local', interactive=True)
        self.assertEqual(str(route['api_key_env']), 'CUSTOM_KEY')
        current = {'model': 'openai/old', 'api_key_env': 'KEY', 'revision': 'r1'}
        route = choose(current=current, model=None)
        self.assertEqual(str(route['model']), 'openai/old')
        self.assertEqual(str(route['revision']), 'r1')
        self.assertEqual(str(choose(model='plain-model')['model']), 'openai/plain-model')

    def test_capability_setup_can_preserve_video_generation_and_decline_key(self):
        self.path.write_text('version=3\nmodel="openai/test"\n'
                             '[capabilities]\nvideo={understand="gemini/old",generate="gemini/generate"}\n')
        with patch('sys.stdin.isatty', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path), '--non-interactive', '--capability', 'video',
                   '--understand-model', 'gemini/new'])
        self.assertIn('gemini/generate', self.path.read_text())
        self.assertIn('gemini/new', self.path.read_text())
        self.path.write_text('version=3\nmodel="openai/test"\n'
                             '[capabilities]\nembedding={model="openai/embed",api_key_env="KEY"}\n')
        original = self.path.read_bytes()
        output = io.StringIO()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup.getpass.getpass', return_value=''), \
                contextlib.redirect_stdout(output):
            setup(['--path', str(self.path), '--capability', 'embedding', '--set-key'])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertIn('API key was not changed', output.getvalue())

    def test_capability_setup_reports_ambient_key_precedence(self):
        self.path.write_text('version=3\nmodel="openai/test"\n'
                             '[capabilities]\nembedding={model="openai/embed",api_key_env="KEY"}\n')
        output = io.StringIO()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup.getpass.getpass', return_value='saved-key'), \
                patch.dict(os.environ, {'KEY': 'ambient-key'}), \
                contextlib.redirect_stdout(output):
            setup(['--path', str(self.path), '--capability', 'embedding', '--set-key'])
        self.assertIn('process environment takes precedence', output.getvalue())
        self.assertIn('KEY="saved-key"', (self.root / '.aiython/credentials.env').read_text())

    def test_capability_internal_validation_and_cancellation(self):
        self.path.write_text('version=3\nmodel="openai/test"\n')
        document = tomlkit.parse(self.path.read_text())
        args = parser().parse_args(['--path', str(self.path), '--capability', 'embedding',
                                    '--understand-model', 'gemini/test'])
        with self.assertRaisesRegex(ConfigError, 'require --capability video'):
            _configure_capability(args, self.path, document, 'embedding', 'default', False)
        with patch('sys.stdin.isatty', return_value=False), \
                patch('aiython.setup._configure_capability', side_effect=EOFError):
            with self.assertRaisesRegex(ConfigError, 'Setup cancelled'):
                setup(['--path', str(self.path), '--non-interactive', '--capability', 'embedding'])

    def test_existing_config_parse_race_is_reported_without_rewrite(self):
        self.path.write_text('version=3\nmodel="openai/test"\n')
        original = self.path.read_bytes()
        with patch('aiython.setup.tomlkit.parse', side_effect=ValueError('private')):
            with self.assertRaisesRegex(ConfigError, 'Cannot edit'):
                setup(['--path', str(self.path), '--model', 'openai/new', '--non-interactive'])
        self.assertEqual(self.path.read_bytes(), original)

    def test_custom_provider_prompts_and_inferred_base_url(self):
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup._ask', side_effect=['http://localhost:11434/v1', '']), \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path), '--provider', 'custom', '--model', 'local'])
        self.assertIn('http://localhost:11434/v1', self.path.read_text())
        self.path.unlink()
        with patch('sys.stdin.isatty', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path), '--base-url', 'http://localhost:11434/v1',
                   '--model', 'local', '--non-interactive'])
        self.assertIn('http://localhost:11434/v1', self.path.read_text())

    def test_interactive_wizard_check_key_and_cancellation(self):
        self.path.write_text('version=3\nmodel="openai/test"\napi_key_env="KEY"\n')
        original = self.path.read_bytes()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup._choice', side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'Setup cancelled'):
                setup(['--path', str(self.path)])
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup._choice', return_value='check'), \
                patch('aiython.setup._check_model') as check, \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path)])
        check.assert_called_once()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup._choice', side_effect=['key', 'no']), \
                patch('aiython.setup.getpass.getpass', return_value='saved-key'), \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path)])
        self.assertIn('KEY="saved-key"', (self.root / '.aiython/credentials.env').read_text())
        self.assertNotEqual(self.path.read_bytes(), original)

    def test_interactive_wizard_declines_failed_model_test(self):
        self.path.write_text('version=3\nmodel="openai/test"\napi_key_env="KEY"\n')
        original = self.path.read_bytes()
        with patch('sys.stdin.isatty', return_value=True), \
                patch.dict(os.environ, {'KEY': 'ambient'}), \
                patch('aiython.setup._choice', side_effect=['model', 'yes', 'no']), \
                patch('aiython.setup._choose_model', return_value='openai/new'), \
                patch('aiython.setup._check_model', side_effect=ConfigError('unavailable')), \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path)])
        self.assertEqual(self.path.read_bytes(), original)
        with patch('sys.stdin.isatty', return_value=True), \
                patch.dict(os.environ, {'KEY': 'ambient'}), \
                patch('aiython.setup._choice', side_effect=['model', 'yes', 'yes']), \
                patch('aiython.setup._choose_model', return_value='openai/new'), \
                patch('aiython.setup._check_model', side_effect=ConfigError('unavailable')), \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path)])
        self.assertIn('openai/new', self.path.read_text())
        with patch('sys.stdin.isatty', return_value=True), \
                patch.dict(os.environ, {'KEY': 'ambient'}), \
                patch('aiython.setup._choice', return_value='model'), \
                patch('aiython.setup._choose_model', side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ConfigError, 'Setup cancelled'):
                setup(['--path', str(self.path)])

    def test_general_key_rotation_reports_ambient_precedence_and_blank_cancel(self):
        self.path.write_text('version=3\nmodel="openai/test"\napi_key_env="KEY"\n')
        original = self.path.read_bytes()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup.getpass.getpass', return_value=''), \
                contextlib.redirect_stdout(io.StringIO()):
            setup(['--path', str(self.path), '--set-key'])
        self.assertEqual(self.path.read_bytes(), original)
        output = io.StringIO()
        with patch('sys.stdin.isatty', return_value=True), \
                patch('aiython.setup.getpass.getpass', return_value='saved-key'), \
                patch.dict(os.environ, {'KEY': 'ambient'}), \
                contextlib.redirect_stdout(output):
            setup(['--path', str(self.path), '--set-key'])
        self.assertIn('process environment takes precedence', output.getvalue())

    def test_setup_rejects_flag_conflicts_before_writing(self):
        cases = [
            (['--profile', 'fast'], '--profile requires'),
            (['--understand-model', 'gemini/test'], 'Video model options'),
            (['--set-key'], '--set-key needs a terminal'),
            (['--capability', 'embedding', '--model', 'openai/test'], 'Create the main project'),
        ]
        for flags, message in cases:
            with self.subTest(flags=flags), patch('sys.stdin.isatty', return_value=False):
                with self.assertRaisesRegex(ConfigError, message):
                    setup(['--path', str(self.path), '--non-interactive', *flags])
            self.assertFalse(self.path.exists())

    def test_setup_rejects_invalid_provider_flags_without_writing(self):
        cases = [
            (['--provider', 'custom', '--model', 'local'], 'requires --base-url'),
            (['--provider', 'custom', '--model', 'local', '--base-url', 'http://['], 'API base URL'),
            (['--provider', 'openai', '--model', 'test', '--api-key-env', 'bad-name'],
             'valid name'),
            (['--provider', 'openai', '--model', 'test', '--api-key-env', ''],
             'requires an API key'),
            (['--provider', 'openai'], 'Model ID is required'),
        ]
        for flags, message in cases:
            with self.subTest(flags=flags), patch('sys.stdin.isatty', return_value=False):
                with self.assertRaisesRegex(ConfigError, message):
                    setup(['--path', str(self.path), '--non-interactive', *flags])
            self.assertFalse(self.path.exists())

    def test_capability_setup_rejects_incompatible_options(self):
        self.path.write_text('version=3\nmodel="openai/test"\n')
        cases = [
            (['--capability', 'embedding', '--understand-model', 'openai/test'],
             'require --capability video'),
            (['--capability', 'video', '--model', 'openai/test'],
             'Video requires'),
            (['--capability', 'embedding', '--profile', 'unknown', '--model', 'openai/test'],
             'not configured'),
        ]
        original = self.path.read_text()
        for flags, message in cases:
            with self.subTest(flags=flags), patch('sys.stdin.isatty', return_value=False):
                with self.assertRaisesRegex(ConfigError, message):
                    setup(['--path', str(self.path), '--non-interactive', *flags])
            self.assertEqual(self.path.read_text(), original)
