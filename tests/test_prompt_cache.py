import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from aiython.agent import ToolAgent
from aiython.cli import run_script
from aiython.models import ProfileConfig, ResolvedConfig
from aiython.prompt_cache import canonical


class PromptContextTests(unittest.TestCase):
    def test_loop_keeps_stable_context_but_reads_fresh_history(self):
        captured = []
        def complete(messages, tools):
            captured.append(copy.deepcopy(messages))
            state = json.loads(messages[2]['content'])
            history = state['frame_objects']['history']
            self.assertEqual(history['type'], 'list')
            self.assertNotIn('snapshot', history)
            self.assertNotIn('value', history)
            self.assertEqual(state['frame_objects']['evaluated']['size'], len(captured) - 1)
            self.assertIn('Python controls statement and loop order', messages[0]['content'])
            return {'role': 'assistant', 'tool_calls': [{'id': 'done', 'type': 'function',
                'function': {'name': 'finish', 'arguments': json.dumps({'outcome': {'kind': 'expression', 'code': 'len(history)'}})}}]}
        provider = Mock(complete=Mock(side_effect=complete))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('history: list[int] = []\nevaluated = []\ndef score(candidate):\n'
                            '    evaluated.append(candidate)\nfor trial in range(10):\n'
                            '    candidate: int = choose next candidate\n    history.append(candidate)\n'
                            '    score(candidate)\n')
            profile = ProfileConfig('default', 'fake', 'model')
            result = run_script(path, config=ResolvedConfig(None,path.parent,'default',{'default':profile}),
                                agent_factory=lambda _: ToolAgent(provider))
        self.assertEqual(result['history'], list(range(10)))
        self.assertEqual(result['evaluated'], list(range(10)))
        self.assertEqual(provider.complete.call_count, 10)
        self.assertTrue(all(messages[:2] == captured[0][:2] for messages in captured))

    def test_canonical_json_is_compact_and_deterministic(self):
        self.assertEqual(canonical({'b': 2, 'a': 1}), '{"a":1,"b":2}')
