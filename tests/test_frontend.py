import unittest

from aithon.frontend import parse
from aithon.models import DirectiveError


class FrontendTests(unittest.TestCase):
    def test_natural_language_assignment_with_bullets(self):
        source = 'summary = summarize the data and include:\n- the problem\n- observed patterns\n- responsible team\n\nprint(summary)\n'
        unit = parse(source)
        self.assertEqual(len(unit.blocks), 1)
        block = next(iter(unit.blocks.values()))
        self.assertTrue(block.expression)
        self.assertEqual(block.span.end_line, 4)
        self.assertTrue(unit.transformed.startswith('summary = ('))
        self.assertIn('print(summary)', unit.transformed)
        self.assertIn('- the problem', block.statement)

    def test_valid_unary_minus_and_blank_boundary_stay_python(self):
        self.assertFalse(parse('problem = 1\n- problem').blocks)
        unit = parse('summary = summarize data:\n- first item\n\n- separate_name')
        self.assertIn('- separate_name', unit.transformed)

    def test_large_multiline_assignment_keeps_target(self):
        source = 'analysis = analyze ticket and return a dict {\n' + ''.join(
            f'    "field{i}": choose an appropriate value from ticket,\n' for i in range(20)) + '}\nprint(analysis)'
        unit = parse(source)
        self.assertEqual(len(unit.blocks), 1)
        block = next(iter(unit.blocks.values()))
        self.assertTrue(block.expression)
        self.assertNotIn('analysis =', block.statement)
        self.assertTrue(unit.transformed.startswith('analysis = ('))

    def test_grammar_positions(self):
        cases = [
            'x = select best user',
            'if person weighs 10 KG:\n    pass',
            'while keep doing work:\n    break',
            'assert everything looks correct',
            'f(select best user, other=pick another user)',
            'x = [pick item for item in users]',
            'x = {item: pick value for item in users}',
            'x = f"hello {choose best name}"',
            'def f():\n    return choose best user',
            'async def f():\n    return await do asynchronous work',
            '@choose best decorator\ndef f():\n    pass',
            'def f(bad parameter):\n    pass',
            'class bad name:\n    pass',
            'pick this target = 10',
            'match obj:\n    case match this pattern:\n        pass',
            'x = "unclosed\nprint(x)',
            'x = (unclosed\nprint(x)',
            'if True:\n pass\n  broken indentation',
            'return 10',
            'try:\n    do some work\nexcept ValueError:\n    fix the error\nfinally:\n    clean up stuff',
        ]
        for source in cases:
            with self.subTest(source=source):
                unit = parse(source)
                self.assertTrue(unit.blocks)
                compile(unit.transformed, unit.filename, "exec")

    def test_statement_boundaries(self):
        unit = parse('do first thing\n\n# a comment\ndo second thing\nx = 1\ndo third thing')
        self.assertEqual(len(unit.blocks), 2)
        self.assertIn("do second thing", next(iter(unit.blocks.values())).statement)

    def test_directives_split_blocks(self):
        unit = parse('do first thing\n# aithon: profile="fast"\ndo second thing')
        self.assertEqual(len(unit.blocks), 2)

    def test_invalid_directives(self):
        for source in ['# aithon: end\nx=1', '# aithon: begin\nx=1',
                       '# aithon: typo="x"\nx=1', '# aithon: prompt="hello"',
                       '# aithon: profile="a"\n# aithon: profile="b"\nx=1']:
            with self.subTest(source=source), self.assertRaises(DirectiveError):
                parse(source)

    def test_comment_text_in_string_is_ignored(self):
        unit = parse('s = """\n# aithon: end\n"""\nx=1 # aithon: end')
        self.assertFalse(unit.directives.annotations)

    def test_no_ai_for_valid_python(self):
        unit = parse('naïve = 10\nanswer = naïve + 2')
        self.assertFalse(unit.blocks)
