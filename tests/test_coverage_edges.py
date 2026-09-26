"""Behavior at process, asset, and safety boundaries."""

import ast
import hashlib
import os
from pathlib import Path
import pickle
import runpy
import tempfile
import threading
import tokenize
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

import aiython
from aiython._config_schema import ModelRoute, ProjectSettings
from aiython import _subprocess_worker
from aiython.assets import Asset, Image
from aiython.directives import Directives, tolerant_tokens
from aiython.models import DirectiveError
from aiython.object_metadata import source_hint, source_node, structural_hint
from aiython.source_guard import SourceWriteError, audit, protect_source, protected
from aiython.stats import record_request_bytes
from pydantic import ValidationError


class BoundaryTests(unittest.TestCase):
    def test_public_exports_and_unknown_attribute(self):
        self.assertTrue(callable(aiython.send_a2a))
        self.assertTrue(callable(aiython.group))
        with self.assertRaises(AttributeError):
            getattr(aiython, "missing_export")

    def test_asset_save_preserves_type_content_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(b"image contents")
            asset = Image(source)
            self.assertEqual(os.fspath(asset), str(source))
            self.assertEqual(asset.mime_type, "image/png")
            self.assertEqual(asset.content_hash(), hashlib.sha256(b"image contents").hexdigest())
            copy = asset.save(root / "copy.png")
            self.assertIsInstance(copy, Image)
            self.assertEqual(copy.path.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                asset.save(root / "copy.png")
            self.assertEqual(Asset(root / "unknown.customsuffix").mime_type,
                             "application/octet-stream")

    def test_config_schema_rejects_empty_model_and_old_version(self):
        with self.assertRaises(ValidationError):
            ModelRoute(model=" \t")
        with self.assertRaises(ValidationError):
            ProjectSettings(version=2, model="openai/test")

    def test_source_guard_handles_descriptors_and_directory_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.py"
            source.write_text("pass\n")
            manager = SimpleNamespace(units={str(source.resolve()): object()})
            self.assertFalse(protected(None, manager))
            self.assertTrue(protected(-1, manager))
            with source.open("rb") as file:
                self.assertTrue(protected(file.fileno(), manager))
            self.assertTrue(protected(source, manager))
            self.assertFalse(protected(root / "notes.txt", manager))
            with protect_source(manager):
                audit("open", (root / "notes.txt", "r", 0))
                with self.assertRaises(SourceWriteError):
                    audit("os.chmod", (source, 0o644))
                with self.assertRaises(SourceWriteError):
                    audit("os.rmdir", (root,))
            audit("os.chmod", (source, 0o644))

    def test_spawn_main_compiles_script_under_worker_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aiython.toml").write_text('version=3\nmodel="openai/test"\n')
            source = root / "main.py"
            source.write_text("worker_value = 42\n")
            with patch.dict(os.environ, {"AIYTHON_SPAWN_ENTRY": str(source)}):
                import sys
                original_path, original_finders = sys.path[:], sys.meta_path[:]
                try:
                    self.assertNotIn("worker_value", runpy.run_module(
                        "aiython._worker_main", run_name="ordinary_worker"))
                    namespace = runpy.run_module("aiython._worker_main", run_name="__mp_main__")
                    self.assertEqual(namespace["worker_value"], 42)
                finally:
                    sys.path[:] = original_path
                    sys.meta_path[:] = original_finders

    def test_worker_reports_unpickleable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.pkl"
            request.write_bytes(pickle.dumps(("ticket", "module", "function", (), {})))
            with patch("aiython.collaboration.worker_entry", return_value=lambda: None):
                result = pickle.loads(_subprocess_worker._run(request))
            self.assertEqual(result[0], "error")
            self.assertIn(result[1], {"AttributeError", "PicklingError"})

    def test_request_bytes_without_active_stats_is_ignored(self):
        record_request_bytes(5)

    def test_directive_lexer_recovers_from_broken_indentation(self):
        source = "  x=1\n y=2\n# aiython: profile=\"fast\"\n"
        tokens = tolerant_tokens(source)
        self.assertIn((2, 1), [token.start for token in tokens if token.string == "y"])
        self.assertTrue(any(token.type == tokenize.COMMENT and
                            token.string.startswith("# aiython:") for token in tokens))
        self.assertTrue(tolerant_tokens("  x=1\n y='''unclosed\n"))

    def test_directive_validation_reports_specific_invalid_forms(self):
        failures = {
            '# aiython: profile="fast" profile="slow"\nx=1': 'duplicate profile',
            '# aiython: profile=bad\nx=1': 'double-quoted',
            '# aiython: prompt=42\nx=1': 'must be strings',
            '# aiython: capability=" "\nx=1': 'must not be empty',
            '# aiython: profile="fast"prompt="hi"\nx=1': 'separate directive fields',
            '# aiython:\nx=1': 'empty directive',
            '# aiython: begin profile="fast"\n  # aiython: end': 'same indentation',
            ('if True:\n    # aiython: begin profile="fast"\n    x=1\n'
             'y=2\n    # aiython: end'): 'region crosses an execution suite',
        }
        for source, message in failures.items():
            with self.subTest(message=message), self.assertRaisesRegex(DirectiveError, message):
                Directives(source, "example.py")

    def test_statement_annotation_cannot_cross_region(self):
        source = ('# aiython: prompt="point"\n'
                  '# aiython: begin profile="region"\n'
                  'x=1\n'
                  '# aiython: end\n')
        directives = Directives(source, "example.py")
        with self.assertRaisesRegex(DirectiveError, 'cannot cross a region boundary'):
            directives.bind(ast.parse(source))

    def test_adjacent_directives_combine_with_intervening_comment(self):
        source = ('# aiython: profile="fast"\n'
                  '# explanatory comment\n'
                  '# aiython: prompt="short"\n'
                  '# another comment\n'
                  'x=1\n')
        directives = Directives(source, "example.py")
        directives.bind(ast.parse(source))
        self.assertEqual(directives.at(5).profile, "fast")
        self.assertEqual(directives.at(5).prompts, ("short",))

    def test_recovered_parse_cannot_bind_past_intervening_source(self):
        source = '# aiython: profile="fast"\nunparsed statement\n# comment\nx=1\n'
        directives = Directives(source, 'recovered.py')
        tree = ast.parse('x=1\n')
        tree.body[0].lineno = 4
        tree.body[0].end_lineno = 4
        with self.assertRaisesRegex(DirectiveError, 'must precede a statement'):
            directives.bind(tree)

    def test_structural_hints_limit_fields_and_signatures(self):
        manager = SimpleNamespace(source_nodes={}, source_hints={}, _lock=threading.Lock())
        field = lambda value, name: getattr(value, name)
        self.assertIsNone(source_node(42, manager, field))
        hint = structural_hint({"public": 1, 99: "excluded"}, manager, field)
        self.assertEqual(hint['fields'], {"public": "int"})
        self.assertTrue(hint['fields_truncated'])
        module = types.ModuleType("sample")
        vars(module)['__name__'] = 3
        self.assertNotIn('name', structural_hint(module, manager, field))
        cls = ast.parse('class Sample:\n' + ''.join(
            f'    field_{index}: int\n' for index in range(21))).body[0]
        class_hint = source_hint(cls)
        self.assertEqual(len(class_hint['fields']), 20)
        self.assertTrue(class_hint['fields_truncated'])
        function = ast.parse('def work(x: int, *args, **kwargs) -> str: pass').body[0]
        self.assertEqual(source_hint(function)['signature'],
                         '(x: int, *args, **kwargs) -> str')
        self.assertEqual(source_hint(ast.parse('x = 1').body[0]), {})

    def test_registered_function_hint_is_cached_without_values(self):
        def selected(value: int) -> int:
            return value

        node = ast.parse('def selected(value: int) -> int: return value').body[0]
        manager = SimpleNamespace(
            source_nodes={selected.__code__.co_filename: (
                {(selected.__name__, selected.__code__.co_firstlineno): (node,)}, {})},
            source_hints={}, _lock=threading.Lock())
        field = lambda value, name: getattr(value, name)
        self.assertEqual(structural_hint(selected, manager, field)['signature'],
                         '(value: int) -> int')
        self.assertIn(node, manager.source_hints)
        module = types.ModuleType('visible_module')
        self.assertEqual(structural_hint(module, manager, field)['name'], 'visible_module')
