import tokenize
import unittest
from unittest.mock import patch

from aiython.frontend import Frontend, parse


class FrontendEdgeTests(unittest.TestCase):
    def test_bullet_header_and_blank_header_candidates(self):
        source = 'describe task:\n- first point\n'
        frontend = Frontend(source, 'test.py')
        self.assertIn((0, len(source.rstrip()), False), frontend.candidates(0))
        self.assertTrue(Frontend('describe task:\n', 'test.py').candidates(0))
        self.assertIn((0, 0, False), Frontend('\n', 'test.py').candidates(0))
        source = 'print(1); y = choose best name'
        self.assertTrue(Frontend(source, 'test.py').candidates(source.index('choose')))

    def test_legacy_fstring_scanner_skips_valid_fields_and_recovers_invalid_one(self):
        source = 'f"{answer} {choose best name}"'
        token = tokenize.TokenInfo(tokenize.STRING, source, (1, 0), (1, len(source)), source)
        other = tokenize.TokenInfo(tokenize.NAME, 'ordinary', (1, 0), (1, 8), 'ordinary')
        frontend = Frontend(source, 'test.py')
        with patch('aiython.frontend.tolerant_tokens', return_value=[other, token]):
            spans = list(frontend.legacy_fstring_candidates())
        start = source.index('choose')
        self.assertEqual(spans, [(start, start + len('choose best name'))])

    def test_legacy_fstring_build_path_recovers_field(self):
        source = 'x = f"{choose best name}"'
        start = source.index('choose')
        frontend = Frontend(source, 'test.py')
        original_check = frontend.check
        def check(text):
            error = original_check(text)
            if error and text == source:
                error.msg = 'f-string: invalid syntax'
            return error
        with patch('aiython.frontend.sys.version_info', (3, 11)), \
                patch.object(frontend, 'check', side_effect=check), \
                patch.object(frontend, 'legacy_fstring_candidates',
                             return_value=iter([(start, start + len('choose best name'))])):
            unit = frontend.build()
        self.assertEqual(len(unit.blocks), 1)
        self.assertIn('execute', unit.transformed)

    def test_recovery_skips_empty_candidate_and_reports_unrecoverable_syntax(self):
        source = 'choose best name'
        frontend = Frontend(source, 'test.py')
        with patch.object(frontend, 'candidates', return_value=[(0, 0, True),
                                                                (0, len(source), True)]):
            self.assertEqual(len(frontend.build().blocks), 1)
        with patch.object(Frontend, 'candidates', return_value=[]):
            with self.assertRaises(SyntaxError):
                parse(source)

    def test_legacy_parser_falls_back_when_no_fstring_field_matches(self):
        source = 'choose best name'
        frontend = Frontend(source, 'test.py')
        original_check = frontend.check
        def check(text):
            error = original_check(text)
            if error and text == source:
                error.msg = 'f-string: invalid syntax'
            return error
        with patch('aiython.frontend.sys.version_info', (3, 11)), \
                patch.object(frontend, 'check', side_effect=check), \
                patch.object(frontend, 'legacy_fstring_candidates', return_value=iter(())):
            self.assertEqual(len(frontend.build().blocks), 1)
        with patch('aiython.frontend.sys.version_info', (3, 14)):
            self.assertEqual(len(parse(source).blocks), 1)


if __name__ == '__main__':
    unittest.main()
