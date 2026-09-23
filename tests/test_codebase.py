from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from aithon.agent import ToolAgent
from aithon.capabilities import CapabilityPermissionError
from aithon.cli import run_script
from aithon.codebase import Codebase
from aithon.models import ProfileConfig, ResolvedConfig


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
        for name in ('.env', 'aithon.toml', 'secret.json', '.git/config',
                     '.hidden.py', 'node_modules/x.py', 'venv/x.py', '.aithon/x.md'):
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
