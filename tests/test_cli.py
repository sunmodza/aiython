import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from aithon.config import credential, resolve
from aithon.models import ConfigError
from aithon.setup import _catalog, _choose_model, setup


class CLITests(unittest.TestCase):
    def test_setup_from_subdirectory_reuses_parent_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'aithon.toml'
            parent.write_text('version=3\nmodel="openrouter/test"\n')
            child = root / 'examples'
            child.mkdir()
            output = io.StringIO()
            with patch('os.getcwd', return_value=str(child)), contextlib.redirect_stdout(output):
                selected = setup([])
            self.assertEqual(selected, parent)
            self.assertFalse((child / 'aithon.toml').exists())
            self.assertIn(str(parent), output.getvalue())

    def test_setup_creates_runnable_route_without_key_in_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--provider", "openrouter", "--model", "example/model", "--path", str(path)])
            config = resolve(path.parent / "main.py")
            self.assertEqual(config.default_profile, "default")
            route = config.profiles["default"].routes["reasoning"][0]
            self.assertEqual(route["model"], "openrouter/example/model")
            self.assertEqual(config.providers[route["provider"]]["api_key_env"], "OPENROUTER_API_KEY")
            self.assertNotIn("api_key =", path.read_text())

    def test_setup_keeps_saved_key_private_and_can_rotate_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="private-test-key"), \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                setup(["--provider", "gemini", "--model", "example-model", "--path", str(path)])
                config = resolve(path.parent / "main.py")
                self.assertEqual(credential(config, config.profiles["default"]), "private-test-key")
            key_path = path.parent / ".aithon" / "credentials.env"
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertIn(".aithon/", (path.parent / ".gitignore").read_text())
            self.assertNotIn("private-test-key", path.read_text())
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="replacement-key"), \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                setup(["--set-key", "--path", str(path)])
                config = resolve(path.parent / "main.py")
                self.assertEqual(credential(config, config.profiles["default"]), "replacement-key")
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("replacement-key", path.read_text())

    def test_setup_cli_creates_and_updates_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            args = [sys.executable, "-m", "aithon", "setup", "--provider", "custom",
                    "--base-url", "http://localhost:11434/v1", "--model", "local-model",
                    "--non-interactive"]
            first = subprocess.run(args, cwd=directory, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            path = Path(directory) / "aithon.toml"
            self.assertTrue(path.is_file())
            self.assertIn("local-model", path.read_text())
            second = subprocess.run(args, cwd=directory, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            changed = subprocess.run([sys.executable, "-m", "aithon", "setup", "--model", "another-model",
                                      "--non-interactive"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(changed.returncode, 0, changed.stderr)
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model, "openai/another-model")

    def test_setup_updates_parent_config_without_losing_comments_or_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "aithon.toml"
            path.write_text('version = 3\nmodel = "openrouter/old" # chosen model\n'
                            'api_key_env = "OPENROUTER_API_KEY"\n\n[capabilities]\n'
                            'embedding = "openai/embedding"\n\n[profiles.fast]\n'
                            'model = "openai/fast"\n')
            child = root / "examples"
            child.mkdir()
            with patch("pathlib.Path.cwd", return_value=child), patch("sys.stdin.isatty", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                selected = setup(["--model", "new", "--non-interactive"])
            self.assertEqual(selected, path)
            source = path.read_text()
            self.assertIn('# chosen model', source)
            self.assertIn('embedding = "openai/embedding"', source)
            self.assertIn('model = "openai/fast"', source)
            self.assertEqual(resolve(child / "main.py").profiles["default"].model, "openrouter/new")

    def test_setup_switches_provider_and_keeps_other_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel="openai/local"\napi_base="http://localhost:11434/v1"\n'
                            'api_key_env="LOCAL_KEY"\nenv_file=".aithon/credentials.env"\n')
            folder = path.parent / ".aithon"
            folder.mkdir()
            (folder / "credentials.env").write_text('LOCAL_KEY="old-local"\nOPENROUTER_API_KEY="saved-router"\n')
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--provider", "openrouter", "--model", "vendor/new", "--path", str(path), "--non-interactive"])
            config = resolve(path.parent / "main.py")
            self.assertEqual(config.profiles["default"].model, "openrouter/vendor/new")
            self.assertEqual(config.profiles["default"].base_url, "")
            self.assertEqual(config.profiles["default"].api_key_env, "OPENROUTER_API_KEY")
            self.assertEqual(credential(config, config.profiles["default"]), "saved-router")
            self.assertIn('LOCAL_KEY="old-local"', (folder / "credentials.env").read_text())

    def test_setup_switches_provider_in_inline_model_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel={model="openai/local",api_base="http://localhost:11434/v1",'
                            'api_key_env="LOCAL_KEY",revision="pinned"}\n')
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--provider", "openrouter", "--model", "vendor/new", "--path", str(path),
                       "--non-interactive"])
            config = resolve(path.parent / "main.py")
            self.assertEqual(config.profiles["default"].model, "openrouter/vendor/new")
            self.assertEqual(config.profiles["default"].base_url, "")
            self.assertIn('revision="pinned"', path.read_text())

    def test_setup_interactive_model_search_updates_existing_project(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n')
            with patch("pathlib.Path.cwd", return_value=path.parent), patch("sys.stdin.isatty", return_value=True), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aithon.setup._choice", side_effect=["model", "no"]), \
                    patch("aithon.setup._choose_model", return_value="openrouter/new") as search, \
                    contextlib.redirect_stdout(io.StringIO()):
                setup([])
            search.assert_called_once()
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model, "openrouter/new")

    def test_setup_initial_wizard_allows_key_later_without_testing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            with patch("sys.stdin.isatty", return_value=True), patch("aithon.setup._choice", return_value="openrouter"), \
                    patch("aithon.setup._choose_model", return_value="vendor/model"), \
                    patch("getpass.getpass", return_value=""), patch.dict(os.environ, {}, clear=True), \
                    patch("aithon.setup._check_model") as check, contextlib.redirect_stdout(io.StringIO()):
                setup(["--path", str(path)])
            check.assert_not_called()
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model,
                             "openrouter/vendor/model")

    def test_setup_full_model_id_selects_provider_and_preserves_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n'
                            'timeout=45\n')
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--model", "openai/new", "--path", str(path), "--non-interactive"])
            config = resolve(path.parent / "main.py")
            self.assertEqual(config.profiles["default"].model, "openai/new")
            self.assertEqual(config.profiles["default"].api_key_env, "OPENAI_API_KEY")
            self.assertEqual(config.profiles["default"].timeout, 45)

    def test_setup_accepts_another_litellm_provider_without_a_special_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--model", "anthropic/example-model", "--path", str(path), "--non-interactive"])
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model,
                             "anthropic/example-model")

    def test_setup_catalog_uses_provider_model_ids_and_falls_back_offline(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"data": [{"id": "openai/model-a"}, {"id": "vendor/model-b"}]}
        with patch("aithon.setup.httpx.get", return_value=response) as request:
            models = _catalog("openrouter", "", None)
        self.assertEqual(models, ["openrouter/openai/model-a", "openrouter/vendor/model-b"])
        self.assertEqual(request.call_args.kwargs["timeout"], 5)
        with patch("aithon.setup.httpx.get", side_effect=httpx.ConnectError("offline")):
            self.assertEqual(_catalog("openrouter", "", None), [])

    def test_setup_model_picker_offers_fuzzy_completion_and_manual_ids(self):
        with patch("aithon.setup._catalog", return_value=["openrouter/vendor/model"]), \
                patch("prompt_toolkit.prompt", return_value="openrouter/custom/preview") as prompt, \
                contextlib.redirect_stdout(io.StringIO()):
            selected = _choose_model("openrouter", "openrouter/old", "", None)
        self.assertEqual(selected, "openrouter/custom/preview")
        self.assertEqual(type(prompt.call_args.kwargs["completer"]).__name__, "FuzzyWordCompleter")

    def test_setup_check_uses_one_tool_call_and_never_writes_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n')
            original = path.read_bytes()
            provider = unittest.mock.Mock()
            provider.completion.return_value = {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "aithon_probe", "arguments": "{\"ok\":true}"}}]}}]}
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aithon.providers.sdk", return_value=provider), \
                    contextlib.redirect_stdout(io.StringIO()):
                setup(["--check", "--path", str(path), "--non-interactive"])
            provider.completion.assert_called_once()
            self.assertEqual(path.read_bytes(), original)
            provider.completion.side_effect = RuntimeError("secret raw provider detail")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aithon.providers.sdk", return_value=provider), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigError, "Model test failed: RuntimeError") as error:
                    setup(["--model", "new", "--check", "--path", str(path), "--non-interactive"])
            self.assertNotIn("secret raw", str(error.exception))
            self.assertEqual(path.read_bytes(), original)

    def test_setup_check_respects_empty_process_credential_over_saved_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aithon.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n'
                            'env_file=".aithon/credentials.env"\n')
            folder = path.parent / ".aithon"
            folder.mkdir()
            (folder / "credentials.env").write_text('OPENROUTER_API_KEY="saved-key"\n')
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
                    patch("aithon.providers.sdk") as sdk, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigError, "Missing credential OPENROUTER_API_KEY"):
                    setup(["--check", "--path", str(path), "--non-interactive"])
            sdk.assert_not_called()

    def test_stats_on_stderr_preserves_program_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text('print("program output")')
            result = subprocess.run([sys.executable, "-m", "aithon", "--stats", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "program output\n")
            self.assertEqual(result.stderr, "aithon stats: []\n")

    def test_cpython_equivalence_and_arguments(self):
        source = '''"docstring"
import sys
print(__name__, __doc__, __package__, __spec__)
print(sys.argv[1:])
class Example:
    def __init__(self, x): self.x = x
print([Example(i).x for i in range(3)])
try:
    {}["missing"]
except KeyError as error:
    print(type(error).__name__)
finally:
    print("done")
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text(source)
            python = subprocess.run([sys.executable, str(path), "--hello", "world"], capture_output=True, text=True)
            aithon = subprocess.run([sys.executable, "-m", "aithon", str(path), "--hello", "world"], capture_output=True, text=True)
            self.assertEqual((aithon.returncode, aithon.stdout, aithon.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_explain_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text('raise AssertionError("must not execute")\nx = choose best value')
            result = subprocess.run([sys.executable, "-m", "aithon", "--explain", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["blocks"][0]["statement"], "choose best value")
            self.assertEqual(len(data["checkpoints"]), 2)

    def test_missing_provider_only_when_ai_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            path = root / "main.py"
            path.write_text('if False:\n    choose best value\nprint("ok")')
            result = subprocess.run([sys.executable, "-m", "aithon", str(path)], capture_output=True, text=True,
                                    env={**os.environ, "XDG_CONFIG_HOME": str(root / "unused")})
            self.assertEqual((result.returncode, result.stdout), (0, "ok\n"))
            path.write_text('choose best value')
            result = subprocess.run([sys.executable, "-m", "aithon", str(path)], capture_output=True, text=True,
                                    env={**os.environ, "XDG_CONFIG_HOME": str(root / "unused")})
            self.assertEqual(result.returncode, 1)
            self.assertIn("requires a configured profile", result.stderr)
