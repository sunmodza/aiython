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

from aithon.config import credential, resolve
from aithon.models import ConfigError
from aithon.setup import setup


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

    def test_setup_keeps_saved_key_private_and_never_overwrites(self):
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
            previous = path.read_bytes()
            with self.assertRaises(ConfigError):
                setup(["--provider", "openai", "--model", "other", "--path", str(path), "--non-interactive"])
            self.assertEqual(path.read_bytes(), previous)

    def test_setup_cli_creates_config_and_rejects_existing_file(self):
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
            self.assertEqual(second.returncode, 1)
            self.assertIn("will not overwrite", second.stderr)

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
