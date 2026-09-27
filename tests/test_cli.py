import contextlib
import importlib.metadata
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
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from aiython.config import CAPABILITIES, credential, resolve
from aiython.cli import main
from aiython.models import ConfigError
from aiython.setup import ModelChoice, _catalog, _choose_model, setup


class CLITests(unittest.TestCase):
    def test_cli_without_arguments_shows_first_run_help_on_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        output = io.StringIO()
        with patch.object(sys, 'stdin', Terminal()), contextlib.redirect_stdout(output):
            main([])
        self.assertIn("aiython setup", output.getvalue())

    def test_cli_version_uses_distribution_metadata(self):
        result = subprocess.run([sys.executable, "-m", "aiython", "--version"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"aiython {importlib.metadata.version('aiython')}")

    def test_setup_from_subdirectory_reuses_parent_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'aiython.toml'
            parent.write_text('version=3\nmodel="openrouter/test"\n')
            child = root / 'examples'
            child.mkdir()
            output = io.StringIO()
            with patch('os.getcwd', return_value=str(child)), contextlib.redirect_stdout(output):
                selected = setup([])
            self.assertEqual(selected, parent)
            self.assertFalse((child / 'aiython.toml').exists())
            self.assertIn(str(parent), output.getvalue())

    def test_setup_creates_runnable_route_without_key_in_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
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
            path = Path(directory) / "aiython.toml"
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="private-test-key"), \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                setup(["--provider", "gemini", "--model", "example-model", "--path", str(path)])
                config = resolve(path.parent / "main.py")
                self.assertEqual(credential(config, config.profiles["default"]), "private-test-key")
            key_path = path.parent / ".aiython" / "credentials.env"
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertIn(".aiython/", (path.parent / ".gitignore").read_text())
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
            args = [sys.executable, "-m", "aiython", "setup", "--provider", "custom",
                    "--base-url", "http://localhost:11434/v1", "--model", "local-model",
                    "--non-interactive"]
            first = subprocess.run(args, cwd=directory, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            path = Path(directory) / "aiython.toml"
            self.assertTrue(path.is_file())
            self.assertIn("local-model", path.read_text())
            second = subprocess.run(args, cwd=directory, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            changed = subprocess.run([sys.executable, "-m", "aiython", "setup", "--model", "another-model",
                                      "--non-interactive"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(changed.returncode, 0, changed.stderr)
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model, "openai/another-model")

    def test_setup_updates_parent_config_without_losing_comments_or_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "aiython.toml"
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
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openai/local"\napi_base="http://localhost:11434/v1"\n'
                            'api_key_env="LOCAL_KEY"\nenv_file=".aiython/credentials.env"\n')
            folder = path.parent / ".aiython"
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
            path = Path(directory) / "aiython.toml"
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
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n')
            with patch("pathlib.Path.cwd", return_value=path.parent), patch("sys.stdin.isatty", return_value=True), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aiython.setup._choice", side_effect=["model", "no"]), \
                    patch("aiython.setup._choose_model", return_value="openrouter/new") as search, \
                    contextlib.redirect_stdout(io.StringIO()):
                setup([])
            search.assert_called_once()
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model, "openrouter/new")

    def test_setup_initial_wizard_allows_key_later_without_testing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            with patch("sys.stdin.isatty", return_value=True), patch("aiython.setup._choice", return_value="openrouter"), \
                    patch("aiython.setup._choose_model", return_value="vendor/model"), \
                    patch("getpass.getpass", return_value=""), patch.dict(os.environ, {}, clear=True), \
                    patch("aiython.setup._check_model") as check, contextlib.redirect_stdout(io.StringIO()):
                setup(["--path", str(path)])
            check.assert_not_called()
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model,
                             "openrouter/vendor/model")

    def test_setup_full_model_id_selects_provider_and_preserves_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
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
            path = Path(directory) / "aiython.toml"
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--model", "anthropic/example-model", "--path", str(path), "--non-interactive"])
            self.assertEqual(resolve(path.parent / "main.py").profiles["default"].model,
                             "anthropic/example-model")

    def test_setup_catalog_uses_provider_model_ids_and_falls_back_offline(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"data": [{"id": "openai/model-a"}, {"id": "vendor/model-b"}]}
        with patch("aiython.setup.httpx.get", return_value=response) as request:
            models = _catalog("openrouter", "", None)
        self.assertEqual([choice.id for choice in models],
                         ["openrouter/openai/model-a", "openrouter/vendor/model-b"])
        self.assertEqual(request.call_args.kwargs["timeout"], 5)
        self.assertEqual(request.call_args.kwargs["params"], {"output_modalities": "all"})
        with patch("aiython.setup.httpx.get", side_effect=httpx.ConnectError("offline")):
            self.assertEqual(_catalog("openrouter", "", None), [])

    def test_setup_openrouter_catalog_filters_each_capability_by_metadata(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"data": [
            {"id": "openai/gpt-transcribe", "name": "GPT Transcribe",
             "architecture": {"input_modalities": ["audio"], "output_modalities": ["transcription"]}},
            {"id": "cohere/rerank", "architecture": {"input_modalities": ["text"],
                                                   "output_modalities": ["rerank"]}},
            {"id": "openai/embed", "architecture": {"input_modalities": ["text"],
                                                   "output_modalities": ["embeddings"]}},
            {"id": "vendor/chat", "architecture": {"input_modalities": ["text", "image"],
                                                     "output_modalities": ["text"]},
             "supported_parameters": ["tools"]},
            {"id": "vendor/no-tools", "architecture": {"input_modalities": ["text"],
                                                         "output_modalities": ["text"]},
             "supported_parameters": ["temperature"]},
            {"id": "vendor/video", "architecture": {"input_modalities": ["text"],
                                                      "output_modalities": ["video"]}},
            {"id": "vendor/chat-image", "architecture": {"input_modalities": ["text", "image"],
                                                           "output_modalities": ["image", "text"]},
             "supported_parameters": ["temperature"]},
            {"id": "vendor/image-only", "architecture": {"input_modalities": ["text", "image"],
                                                           "output_modalities": ["image"]}},
        ]}
        expected = {"reasoning": ["vendor/chat"], "speech_to_text": ["openai/gpt-transcribe"],
                    "reranking": ["cohere/rerank"], "embedding": ["openai/embed"],
                    "vision": ["vendor/chat", "vendor/chat-image"], "image_generation": ["vendor/chat-image"],
                    "image_editing": ["vendor/chat-image"]}
        with patch("aiython.setup.httpx.get", return_value=response):
            for capability, names in expected.items():
                with self.subTest(capability=capability):
                    self.assertEqual([choice.id.removeprefix("openrouter/") for choice in
                                      _catalog("openrouter", "", None, capability=capability)], names)
        video_response = unittest.mock.Mock()
        video_response.json.return_value = {"data": [{"id": "vendor/video", "name": "Video"}]}
        with patch("aiython.setup.httpx.get", return_value=video_response) as request:
            self.assertEqual([choice.id for choice in _catalog("openrouter", "", None,
                                                                capability="video", mode="generate")],
                             ["openrouter/vendor/video"])
        self.assertEqual(str(request.call_args.args[0]), "https://openrouter.ai/api/v1/videos/models")

    def test_setup_gemini_catalog_uses_supported_methods_when_available(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"models": [
            {"name": "models/text-model", "displayName": "Text Model",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed-model", "displayName": "Embed Model",
             "supportedGenerationMethods": ["embedContent"]},
        ]}
        with patch("aiython.setup.httpx.get", return_value=response):
            self.assertEqual([choice.id for choice in _catalog("gemini", "", "test-key")],
                             ["gemini/text-model"])
            self.assertEqual([choice.id for choice in _catalog("gemini", "", "test-key", capability="embedding")],
                             ["gemini/embed-model"])

    def test_setup_model_picker_matches_words_and_accepts_manual_ids(self):
        choices = [ModelChoice("openrouter/openai/gpt-transcribe", "GPT Transcribe"),
                   ModelChoice("openrouter/qwen/qwen3-30b-a3b-instruct-2507", "Qwen Instruct"),
                   ModelChoice("openrouter/vendor/voice-model", "Speech Transcriber")]
        with patch("aiython.setup._catalog", return_value=choices), \
                patch("prompt_toolkit.prompt", return_value="openrouter/custom/preview") as prompt, \
                contextlib.redirect_stdout(io.StringIO()):
            selected = _choose_model("openrouter", "openrouter/old", "", None,
                                     capability="speech_to_text")
        self.assertEqual(selected, "openrouter/custom/preview")
        completer = prompt.call_args.kwargs["completer"]
        def matches(query):
            return [item.text for item in completer.get_completions(Document(query), CompleteEvent())]
        self.assertEqual(matches("transc"), ["openrouter/openai/gpt-transcribe",
                                             "openrouter/vendor/voice-model"])
        self.assertEqual(matches("QWEN 30b"), ["openrouter/qwen/qwen3-30b-a3b-instruct-2507"])
        self.assertEqual(matches("zzunknown"), [])
        self.assertEqual(prompt.call_args.kwargs["placeholder"], "openrouter/old")

    def test_setup_check_uses_one_tool_call_and_never_writes_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n')
            original = path.read_bytes()
            provider = unittest.mock.Mock()
            provider.completion.return_value = {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "aiython_probe", "arguments": "{\"ok\":true}"}}]}}]}
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aiython.providers.sdk", return_value=provider), \
                    contextlib.redirect_stdout(io.StringIO()):
                setup(["--check", "--path", str(path), "--non-interactive"])
            provider.completion.assert_called_once()
            self.assertEqual(path.read_bytes(), original)
            provider.completion.side_effect = RuntimeError("secret raw provider detail")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aiython.providers.sdk", return_value=provider), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigError, "Model test failed: RuntimeError") as error:
                    setup(["--model", "new", "--check", "--path", str(path), "--non-interactive"])
            self.assertNotIn("secret raw", str(error.exception))
            self.assertEqual(path.read_bytes(), original)

    def test_setup_check_respects_empty_process_credential_over_saved_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/old"\napi_key_env="OPENROUTER_API_KEY"\n'
                            'env_file=".aiython/credentials.env"\n')
            folder = path.parent / ".aiython"
            folder.mkdir()
            (folder / "credentials.env").write_text('OPENROUTER_API_KEY="saved-key"\n')
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
                    patch("aiython.providers.sdk") as sdk, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigError, "Missing credential OPENROUTER_API_KEY"):
                    setup(["--check", "--path", str(path), "--non-interactive"])
            sdk.assert_not_called()

    def test_setup_configures_every_non_video_capability_with_its_own_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\n'
                            'api_key_env="OPENROUTER_API_KEY"\n\n'
                            '# >>> Aiython missing route: profile="default" capability=speech_to_text\n'
                            '# Add this line to [capabilities], creating the section if necessary:\n'
                            '# [capabilities]\n# speech_to_text = "openai/YOUR_TRANSCRIPTION_MODEL"\n'
                            '# Replace the placeholder with a LiteLLM model ID and configure its credential.\n'
                            '# <<< Aiython missing route\n')
            for capability in sorted(CAPABILITIES - {"reasoning", "video"}):
                with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    setup(["--capability", capability, "--model", "openai/example-model",
                           "--path", str(path), "--non-interactive"])
            config = resolve(path.parent / "main.py")
            for capability in CAPABILITIES - {"reasoning", "video"}:
                route = config.profiles["default"].routes[capability][0]
                self.assertEqual(route["model"], "openai/example-model")
                self.assertEqual(config.providers[route["provider"]]["api_key_env"], "OPENAI_API_KEY")
            self.assertEqual(config.profiles["default"].model, "openrouter/tool-model")
            self.assertNotIn("# >>> Aiython missing route: profile=", path.read_text())

    def test_setup_video_modes_can_be_configured_and_changed_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\napi_key_env="OPENROUTER_API_KEY"\n')
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "video", "--understand-model", "gemini/understand",
                       "--generate-model", "gemini/generate", "--path", str(path), "--non-interactive"])
                setup(["--capability", "video", "--generate-model", "gemini/generate-v2",
                       "--path", str(path), "--non-interactive"])
            config = resolve(path.parent / "main.py")
            routes = config.profiles["default"].routes["video"]
            self.assertEqual([(route["modes"], route["model"]) for route in routes],
                             [(["understand"], "gemini/understand"), (["generate"], "gemini/generate-v2")])
            self.assertTrue(all(config.providers[route["provider"]]["api_key_env"] == "GEMINI_API_KEY"
                                for route in routes))

    def test_setup_video_picker_filters_each_mode_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\napi_key_env="OPENROUTER_API_KEY"\n')
            with patch("sys.stdin.isatty", return_value=True), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "GEMINI_API_KEY": "test-key"}), \
                    patch("aiython.setup._choice", side_effect=["openrouter", "gemini"]), \
                    patch("aiython.setup._choose_model", side_effect=["openrouter/understand", "gemini/generate"]) as chooser, \
                    contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "video", "--path", str(path)])
            self.assertEqual([(call.kwargs["capability"], call.kwargs["mode"])
                              for call in chooser.call_args_list],
                             [("video", "understand"), ("video", "generate")])

    def test_setup_saves_openrouter_video_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            source = 'version=3\nmodel="openrouter/tool-model"\napi_key_env="OPENROUTER_API_KEY"\n'
            path.write_text(source)
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "video", "--understand-model", "openrouter/google/gemini-2.5-flash",
                       "--generate-model", "openrouter/minimax/hailuo-3", "--path", str(path),
                       "--non-interactive"])
            self.assertIn('openrouter/minimax/hailuo-3', path.read_text())

    def test_setup_capability_uses_named_profile_and_saves_key_privately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\napi_key_env="OPENROUTER_API_KEY"\n'
                            'env_file=".aiython/credentials.env"\n\n[profiles.fast]\nmodel="openai/fast"\n')
            folder = path.parent / ".aiython"
            folder.mkdir()
            (folder / "credentials.env").write_text('OPENROUTER_API_KEY="router-key"\n')
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="openai-key"), \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "speech_to_text", "--profile", "fast", "--model", "openai/whisper-1",
                       "--path", str(path)])
                config = resolve(path.parent / "main.py")
                route = config.profiles["fast"].routes["speech_to_text"][0]
                self.assertEqual(config.providers[route["provider"]]["api_key_env"], "OPENAI_API_KEY")
                self.assertEqual(config.secrets["OPENAI_API_KEY"], "openai-key")
                self.assertEqual(config.secrets["OPENROUTER_API_KEY"], "router-key")
            self.assertNotIn("openai-key", path.read_text())
            self.assertEqual((folder / "credentials.env").stat().st_mode & 0o777, 0o600)

    def test_setup_capability_key_rotation_keeps_model_without_catalog_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\n'
                            '[capabilities]\nspeech_to_text={model="openai/whisper-1",'
                            'api_key_env="OPENAI_API_KEY"}\n')
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="new-key"), \
                    patch("aiython.setup._choose_model") as chooser, patch("aiython.setup._choice") as choice, \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "speech_to_text", "--set-key", "--path", str(path)])
            chooser.assert_not_called()
            choice.assert_not_called()
            config = resolve(path.parent / "main.py")
            self.assertEqual(config.profiles["default"].routes["speech_to_text"][0]["model"],
                             "openai/whisper-1")
            self.assertEqual(config.secrets["OPENAI_API_KEY"], "new-key")

    def test_setup_capability_keeps_revision_and_does_not_collapse_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\n'
                            '[capabilities]\nspeech_to_text={model="openai/whisper-1",'
                            'api_key_env="OPENAI_API_KEY",revision="pinned"}\n')
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                setup(["--capability", "speech_to_text", "--api-key-env", "SECOND_OPENAI_KEY",
                       "--path", str(path), "--non-interactive"])
            self.assertIn('revision = "pinned"', path.read_text())
            path.write_text('version=3\nmodel="openrouter/tool-model"\n'
                            '[capabilities]\nspeech_to_text=["openai/first","openai/second"]\n')
            original = path.read_bytes()
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigError, "fallback routes"):
                    setup(["--capability", "speech_to_text", "--api-key-env", "SECOND_OPENAI_KEY",
                           "--path", str(path), "--non-interactive"])
            self.assertEqual(path.read_bytes(), original)

    def test_setup_capability_menu_allows_multiple_routes_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\napi_key_env="OPENROUTER_API_KEY"\n')
            with patch("pathlib.Path.cwd", return_value=path.parent), patch("sys.stdin.isatty", return_value=True), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                    patch("aiython.setup._choice", side_effect=["capability", "speech_to_text", "openrouter",
                                                               "embedding", "openrouter", "done"]), \
                    patch("aiython.setup._choose_model", side_effect=["openrouter/transcribe", "openrouter/embed"]) as chooser, \
                    contextlib.redirect_stdout(io.StringIO()):
                setup([])
            routes = resolve(path.parent / "main.py").profiles["default"].routes
            self.assertIn("speech_to_text", routes)
            self.assertIn("embedding", routes)
            self.assertEqual([call.kwargs["capability"] for call in chooser.call_args_list],
                             ["speech_to_text", "embedding"])

    def test_setup_suggests_missing_route_from_runtime_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\n'
                            '\n# >>> Aiython missing route: profile="default" capability=speech_to_text\n'
                            '# <<< Aiython missing route\n')
            output = io.StringIO()
            with patch("pathlib.Path.cwd", return_value=path.parent), patch("sys.stdin.isatty", return_value=True), \
                    patch("aiython.setup._choice", return_value="done") as choice, \
                    contextlib.redirect_stdout(output):
                setup([])
            self.assertIn("Suggested next: configure speech_to_text", output.getvalue())
            self.assertEqual(choice.call_args.kwargs["default"], "capability")

    def test_setup_video_missing_mode_and_check_rejection_leave_config_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiython.toml"
            path.write_text('version=3\nmodel="openrouter/tool-model"\n')
            original = path.read_bytes()
            with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ConfigError):
                    setup(["--capability", "video", "--generate-model", "gemini/generate",
                           "--path", str(path), "--non-interactive"])
                with self.assertRaisesRegex(ConfigError, "reasoning model only"):
                    setup(["--capability", "speech_to_text", "--model", "openai/whisper-1", "--check",
                           "--path", str(path), "--non-interactive"])
            self.assertEqual(path.read_bytes(), original)

    def test_stats_on_stderr_preserves_program_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text('print("program output")')
            result = subprocess.run([sys.executable, "-m", "aiython", "--stats", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "program output\n")
            lines = result.stderr.splitlines()
            self.assertEqual(lines[0], "aiython stats: []")
            summary = json.loads(lines[1].removeprefix('aiython run stats: '))
            self.assertGreater(summary['total_seconds'], 0)
            self.assertGreaterEqual(summary['total_seconds'], summary['execution_seconds'])
            self.assertEqual(summary['preparation_cache_misses'], 1)

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
            aiython = subprocess.run([sys.executable, "-m", "aiython", str(path), "--hello", "world"], capture_output=True, text=True)
            self.assertEqual((aiython.returncode, aiython.stdout, aiython.stderr),
                             (python.returncode, python.stdout, python.stderr))

    def test_explain_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text('raise AssertionError("must not execute")\nx = choose best value')
            result = subprocess.run([sys.executable, "-m", "aiython", "--explain", str(path)], capture_output=True, text=True)
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
            result = subprocess.run([sys.executable, "-m", "aiython", str(path)], capture_output=True, text=True,
                                    env={**os.environ, "XDG_CONFIG_HOME": str(root / "unused")})
            self.assertEqual((result.returncode, result.stdout), (0, "ok\n"))
            path.write_text('choose best value')
            result = subprocess.run([sys.executable, "-m", "aiython", str(path)], capture_output=True, text=True,
                                    env={**os.environ, "XDG_CONFIG_HOME": str(root / "unused")})
            self.assertEqual(result.returncode, 1)
            self.assertIn("requires a configured profile", result.stderr)
