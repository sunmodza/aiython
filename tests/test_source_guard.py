import inspect
import json
import os  # noqa: F401 - accessed through the live frame during source guard tests
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from aithon.agent import ToolAgent
from aithon.cli import run_script
from aithon.models import ProfileConfig, ResolvedConfig
from aithon.runtime import Runtime, RuntimeBridge
from aithon.source_guard import SourceWriteError


def response(identifier, name, **args):
    return {'role':'assistant','tool_calls':[{'id':identifier,'type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}}]}


class SourceGuardTests(unittest.TestCase):
    def test_source_mutations_blocked_before_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'main.py'; source.write_text('original')
            temporary = root/'replacement.tmp'; temporary.write_text('replacement')
            alias = root/'alias.txt'; alias.symlink_to(source)
            bridge = RuntimeBridge(inspect.currentframe(), Runtime(ResolvedConfig(None,root)))
            for code in ("source.write_text('bad')", "open(source, 'w').write('bad')",
                         "os.open(source, os.O_WRONLY | os.O_TRUNC)", "source.unlink()",
                         "temporary.replace(source)", "source.rename(root/'moved.txt')",
                         "alias.write_text('bad')", "__import__('subprocess').run(['echo','bad'])"):
                with self.subTest(code=code), self.assertRaises(SourceWriteError):
                    bridge.eval(code)
                self.assertEqual(source.read_text(),'original')
            with self.assertRaises(SourceWriteError):
                bridge.exec("source.write_text('bad')")
            # Guard ends with AI execution; normal Python and data writes still work.
            bridge.eval("(root/'output.txt').write_text('data')")
            source.write_text('normal Python')
            self.assertEqual(source.read_text(),'normal Python')

    def test_agent_repairs_in_runtime_after_blocked_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'main.py'
            source = 'answer = compute answer please\n'
            path.write_text(source)
            provider = Mock()
            provider.complete.side_effect = [response('bad','execute',code="open(__file__, 'w').write('changed')"),
                response('good','finish',code='42')]
            profile = ProfileConfig('default','fake','model')
            result = run_script(path,config=ResolvedConfig(None,path.parent,'default',{'default':profile}),
                                agent_factory=lambda _:ToolAgent(provider))
            self.assertEqual(result['answer'],42)
            self.assertEqual(path.read_text(),source)
            errors = [json.loads(m['content']) for m in provider.complete.call_args.args[0] if m['role']=='tool']
            self.assertEqual(errors[0]['error'],'SourceWriteError')
