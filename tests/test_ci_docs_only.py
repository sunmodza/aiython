import unittest

from scripts.ci_docs_only import is_docs_only


class DocsOnlyScopeTests(unittest.TestCase):
    def test_docs_and_readme_assets(self):
        self.assertTrue(
            is_docs_only(
                [
                    "README.md",
                    "README.pypi.md",
                    "docs/getting-started.md",
                    "docs/assets/icon.png",
                    "assets/readme/runtime-debug.gif",
                    ".github/ISSUE_TEMPLATE/bug_report.md",
                    "zensical.toml",
                ]
            )
        )

    def test_empty_or_code_changes_run_full_suite(self):
        for paths in (
            [],
            ["src/aiython/runtime.py"],
            ["README.md", "src/aiython/runtime.py"],
            ["docs/example.py"],
            ["assets/readme/build_runtime_debug.py"],
            ["examples/recipes/03_loop.py"],
            [".github/workflows/tests.yml"],
            ["pyproject.toml"],
            ["uv.lock"],
        ):
            with self.subTest(paths=paths):
                self.assertFalse(is_docs_only(paths))
