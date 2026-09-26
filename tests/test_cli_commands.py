"""CLI command dispatch and cleanup around a script run."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiython.cli import ProjectFinder, main, read_source, run_script
from aiython.models import AiythonError, ResolvedConfig
from aiython.runtime import Runtime


class CLICommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "main.py"
        self.script.write_text("answer = 42\n")
        (self.root / "aiython.toml").write_text('version=3\nmodel="openai/test"\n')

    def test_jobs_list_and_resume_use_selected_profile(self):
        with patch("aiython.cli.Runtime") as runtime_type:
            runtime = runtime_type.return_value
            runtime.capabilities.store.jobs.return_value = [{"id": "job-1"}]
            runtime.capabilities.resume_job.return_value.path = self.root / "result.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["jobs", "list", "--script", str(self.script)])
            self.assertEqual(json.loads(output.getvalue()), [{"id": "job-1"}])
            runtime.capabilities.require.assert_called_with(unittest.mock.ANY, "read_asset")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["jobs", "resume", "job-1", "--script", str(self.script)])
            self.assertIn("result.json", output.getvalue())
            runtime.capabilities.resume_job.assert_called_with(unittest.mock.ANY, "job-1")

    def test_jobs_missing_profile_and_operation_fail_cleanly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing_operation:
                main(["jobs", "resume", "--script", str(self.script)])
        self.assertEqual(missing_operation.exception.code, 2)
        (self.root / "aiython.toml").unlink()
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as missing_profile:
            main(["jobs", "list", "--script", str(self.script)])
        self.assertEqual(missing_profile.exception.code, 1)
        self.assertIn("Select a configured profile", output.getvalue())

    def test_config_show_and_script_errors(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["config", "show", "--script", str(self.script)])
        self.assertEqual(json.loads(output.getvalue())["default_profile"], "default")

        error = AiythonError("script failed")
        error.__cause__ = ValueError("underlying failure")
        output = io.StringIO()
        with patch("aiython.cli.run_script", side_effect=error), contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as exit_status:
                main([str(self.script)])
        self.assertEqual(exit_status.exception.code, 1)
        self.assertIn("ValueError: underlying failure", output.getvalue())
        self.assertIn("aiython: script failed", output.getvalue())

    def test_missing_script_read_is_redacted(self):
        with self.assertRaisesRegex(AiythonError, "Cannot read source file") as caught:
            read_source(self.root / "missing.py")
        self.assertNotIn("No such file", str(caught.exception))

    def test_project_finder_excludes_virtual_environment_files(self):
        environment = self.root / "venv"
        environment.mkdir()
        (environment / "private_module.py").write_text("secret = 1\n")
        finder = ProjectFinder(Runtime(ResolvedConfig(None, self.root)))
        self.assertIsNone(finder.find_spec("private_module", [str(environment)]))

    def test_script_restores_absent_main_and_previous_spawn_entry(self):
        old_main = sys.modules.pop("__main__", None)
        try:
            with patch.dict(os.environ, {"AIYTHON_SPAWN_ENTRY": "previous.py"}):
                result = run_script(self.script, config=ResolvedConfig(None, self.root))
                self.assertEqual(result["answer"], 42)
                self.assertNotIn("__main__", sys.modules)
                self.assertEqual(os.environ["AIYTHON_SPAWN_ENTRY"], "previous.py")
        finally:
            if old_main is not None:
                sys.modules["__main__"] = old_main
