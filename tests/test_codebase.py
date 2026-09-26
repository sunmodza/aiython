from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aiython.agent import ToolAgent
from aiython.capabilities import CapabilityPermissionError
from aiython.cli import run_script
from aiython.codebase import Codebase
from aiython.models import ProfileConfig, ResolvedConfig


def call(identifier, name, **args):
    return {'role':'assistant', 'tool_calls':[{'id':identifier, 'type':'function',
            'function':{'name':name, 'arguments':json.dumps(args)}}]}


class CodebaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = Codebase(self.root)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_search_and_read_without_executing_source(self):
        path = self.write('helpers.py', 'raise RuntimeError("do not execute")\ndef choose():\n    return 42\n')
        hits = self.code.search('def choose')
        self.assertEqual(hits['matches'][0]['line'], 2)
        self.assertEqual(hits['matches'][0]['path'], 'helpers.py')
        self.assertEqual(self.code.read('helpers.py',2,3)['lines'][1]['text'], '    return 42')
        self.assertEqual(self.code.read(str(path),2,3)['lines'][1]['text'], '    return 42')
        self.assertFalse(hits['truncated'])

    def test_exclusions_and_path_escape(self):
        self.write('safe.py', 'public')
        for name in ('.env', 'aiython.toml', 'secret.json', '.git/config',
                     '.hidden.py', 'node_modules/x.py', 'venv/x.py', '.aiython/x.md'):
            self.write(name, 'secret')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.code.read(name)
        self.assertEqual(self.code.list_files()['files'], ['safe.py'])
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other)/'outside.py'; outside.write_text('secret')
            (self.root/'link.py').symlink_to(outside)
            (self.root/'linked').symlink_to(Path(other), target_is_directory=True)
            for name in ('link.py','linked/outside.py','../outside.py',str(outside)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.code.read(name)
            self.assertEqual(self.code.search('secret')['matches'], [])

    def test_bounds_pagination_binary_and_literal_search(self):
        self.write('a.py', 'a.*b\na.*b\na.*b')
        self.write('b.py', 'not relevant')
        self.write('large.py', 'x' * 512001)
        self.write('binary.py', '\x00not source')
        self.assertEqual(self.code.list_files(limit=1)['next_offset'], 1)
        self.assertEqual(self.code.list_files(offset=1, limit=1)['files'], ['b.py'])
        hits = self.code.search('a.*b',limit=2)
        self.assertEqual(len(hits['matches']),2)
        self.assertTrue(hits['truncated'])
        self.assertEqual(self.code.search('nothing')['skipped_files'],2)
        with self.assertRaises(ValueError): self.code.read('a.py',1,201)
        self.write('long.py','x'*20000)
        read = self.code.read('long.py')
        self.assertTrue(read['lines'][0]['line_truncated'])
        self.assertEqual(len(read['lines'][0]['text']),8000)

    def test_agent_searches_then_uses_found_helper(self):
        self.write('helpers.py','def choose():\n    return 42\n')
        script = self.write('main.py','answer: int = choose using project rules please\n')
        provider = Mock()
        provider.complete.side_effect = [call('search','search_code',query='def choose'),
            call('read','read_code',path='helpers.py'),
            call('finish','finish',code="__import__('helpers').choose()")]
        profile = ProfileConfig('default','fake','model')
        config = ResolvedConfig(None,self.root,'default',{'default':profile})
        try:
            result = run_script(script,config=config,agent_factory=lambda _:ToolAgent(provider))
            self.assertEqual(result['answer'],42)
            messages = provider.complete.call_args.args[0]
            results = [json.loads(m['content']) for m in messages if m['role']=='tool']
            self.assertEqual(results[0]['matches'][0]['path'],'helpers.py')
            self.assertEqual(results[1]['lines'][0]['text'],'def choose():')
        finally:
            import sys
            sys.modules.pop('helpers',None)
        provider.complete.side_effect = None
        provider.complete.return_value = call('search','search_code',query='choose')
        restricted = replace(profile,permissions=('network',))
        config = replace(config,profiles={'default':restricted})
        with self.assertRaises(CapabilityPermissionError):
            run_script(script,config=config,agent_factory=lambda _:ToolAgent(provider))

    def test_scan_and_search_limits_report_truncation(self):
        self.write('first.py', 'needle\n')
        self.write('second.py', 'needle\n')
        with patch('aiython.codebase.MAX_ENTRIES', 1):
            self.assertIn('Project scan limit', self.code.list_files()['reason'])
            self.assertIn('Project scan limit', self.code.search('needle')['reason'])
        with patch('aiython.codebase.MAX_SEARCH_BYTES', 1):
            result = self.code.search('needle')
            self.assertTrue(result['truncated'])
            self.assertIn('Search byte limit', result['reason'])

    def test_specific_path_failures_and_bounded_read(self):
        for query in ('', 'two\nlines'):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, 'single-line'):
                self.code.search(query)
        with self.assertRaisesRegex(ValueError, 'regular source file'):
            self.code.read('missing.py')
        with self.assertRaisesRegex(ValueError, 'regular source file'):
            self.code.search('anything', path='missing.py')
        self.write('wide.py', '\n'.join(['a' * 8000] * 3))
        result = self.code.read('wide.py', 1, 3)
        self.assertEqual(len(result['lines']), 2)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['next_line'], 3)

    def test_scan_skips_special_files_and_unmatched_names(self):
        self.write('ordinary.py', 'hello')
        self.assertEqual(self.code.list_files(query='missing')['files'], [])
        if hasattr(__import__('os'), 'mkfifo'):
            __import__('os').mkfifo(self.root / 'pipe.py')
            self.assertNotIn('pipe.py', self.code.list_files()['files'])

    def test_path_stays_inside_project_if_symlink_changes_during_check(self):
        path = self.write('safe.py', 'public')
        original_resolve = Path.resolve

        def changed_target(candidate, *args, **kwargs):
            if candidate == path:
                return self.root.parent / 'outside.py'
            return original_resolve(candidate, *args, **kwargs)

        with patch.object(Path, 'resolve', changed_target):
            with self.assertRaisesRegex(ValueError, 'outside project'):
                self.code.path('safe.py')
