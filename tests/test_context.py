import inspect
import json
from pathlib import Path
import tempfile
import unittest

from aithon.cli import run_script
from aithon.frontend import parse
from aithon.models import AgentRequest, ProfileConfig, RecoveryDecision, ResolvedConfig, SourceSpan
from aithon.runtime import Runtime, RuntimeBridge


class ContextTests(unittest.TestCase):
    def test_project_metadata_has_types_keys_and_signatures_without_internal_names(self):
        captured = {}
        class Agent:
            def execute(self, request, runtime):
                captured.update(runtime.describe_handles(request.frame_objects | request.related_objects))
                return None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.py'
            path.write_text('''from typing import TypedDict
class Params(TypedDict):
    learning_rate: float
    max_depth: int
history: list[dict] = []
search_space = {"learning_rate": [0.001, 0.3], "max_depth": [2, 16]}
def evaluate(p: Params) -> float:
    return 0.9
review this context please
''')
            profile = ProfileConfig('default', 'fake', 'model')
            run_script(path, config=ResolvedConfig(None,path.parent,'default',{'default':profile}),
                       agent_factory=lambda _: Agent())
        self.assertFalse(any(name.startswith('__') for name in captured))
        self.assertEqual(captured['Params']['type'], 'class')
        self.assertEqual(captured['Params']['name'], 'Params')
        self.assertEqual(captured['Params']['fields'], {'learning_rate':'float','max_depth':'int'})
        self.assertEqual(captured['evaluate']['signature'], '(p: Params) -> float')
        self.assertEqual(captured['history']['annotation'], 'list[dict]')
        self.assertEqual(captured['search_space']['fields'], {'learning_rate':'list','max_depth':'list'})
        self.assertNotIn('0.001', json.dumps(captured))

    def test_initial_metadata_never_serializes_values_or_traverses_objects(self):
        class Dangerous(list):
            def __iter__(self):
                raise AssertionError('must not traverse')
            def __repr__(self):
                raise AssertionError('must not repr')
        bridge = self.bridge()
        history = [{'private': 'sensitive'}] * 100000
        objects = {'history': history, 'alias': history, 'text': 'private-content',
                   'number': 987654321, 'custom': Dangerous(), 'huge': 10**10000}
        metadata = bridge.describe_handles(objects)
        encoded = json.dumps(metadata)
        self.assertLess(len(encoded), 1500)
        self.assertNotIn('private-content', encoded)
        self.assertNotIn('sensitive', encoded)
        self.assertNotIn('987654321', encoded)
        self.assertEqual(metadata['history'], metadata['alias'])
        for description in metadata.values():
            self.assertTrue({'handle', 'type'} <= description.keys())
            self.assertNotIn('value', description)
            self.assertNotIn('snapshot', description)
        self.assertEqual(metadata['history']['first_item_fields'], {'private': 'str'})
        self.assertEqual(metadata['history']['size'], 100000)
        self.assertEqual(bridge.inspect(metadata['text']['handle'])['value'], 'private-content')
        history.append({'private': 'latest'})
        self.assertEqual(bridge.dereference(metadata['history']['handle'])[-1]['private'], 'latest')

    def test_source_index_keeps_decorated_and_nested_function_signatures(self):
        source = '''def keep(fn): return fn
@keep
def selected(value: int) -> int: return value
def left():
    def choice(item: int) -> int: return item
    return choice
def right():
    def choice(item: str) -> str: return item
    return choice
'''
        filename = '<context-index>'
        manager = Runtime(ResolvedConfig(None, Path.cwd()))
        manager.register(parse(source, filename))
        namespace = {}
        exec(compile(source, filename, 'exec'), namespace)
        bridge = RuntimeBridge(inspect.currentframe(), manager)
        metadata = bridge.describe_handles({
            'selected': namespace['selected'],
            'left_choice': namespace['left'](),
            'right_choice': namespace['right'](),
        })
        self.assertEqual(metadata['selected']['signature'], '(value: int) -> int')
        self.assertEqual(metadata['left_choice']['signature'], '(item: int) -> int')
        self.assertEqual(metadata['right_choice']['signature'], '(item: str) -> str')

    def test_nested_inspection_in_one_call_and_cycles(self):
        bridge = self.bridge()
        tickets = [{'id': 'INC-1', 'message': 'payment failure'}, {'id': 'INC-2', 'message': 'slow'}]
        result = bridge.inspect(bridge.handle(tickets)['handle'], depth=2)
        values = {entry['key']['value']: entry['value']['value'] for entry in result['items'][0]['entries']}
        self.assertEqual(values['message'], 'payment failure')
        cycle = []
        cycle.append(cycle)
        handle = bridge.handle(cycle)['handle']
        self.assertEqual(bridge.inspect(handle, depth=3)['items'][0]['handle'], handle)

    def bridge(self):
        return RuntimeBridge(inspect.currentframe(), Runtime(ResolvedConfig(None, Path.cwd())))

    def test_handles_intern_identity_not_equality(self):
        bridge = self.bridge()
        value = []
        value.append(value)
        first = bridge.handle(value)
        self.assertEqual(first, bridge.handle(value))
        self.assertNotEqual(first["handle"], bridge.handle([])["handle"])
        self.assertIs(bridge.dereference(first["handle"]), value)
        self.assertEqual(bridge.inspect(first["handle"])["items"][0]["handle"], first["handle"])
        self.assertEqual((first["type"], first["module"], first["qualname"]), ("list", "builtins", "list"))

    def test_metadata_and_inspection_do_not_call_user_hooks(self):
        events = []
        class Meta(type):
            def __getattribute__(self, name):
                events.append("metaclass attribute")
                raise AssertionError(name)
        class Person(metaclass=Meta):
            def __getattribute__(self, name):
                events.append("attribute")
                raise AssertionError(name)
            def __repr__(self):
                events.append("repr")
                raise AssertionError()
            def __dir__(self):
                events.append("dir")
                raise AssertionError()
            @property
            def value(self):
                events.append("property")
                raise AssertionError()
        person = Person()
        bridge = self.bridge()
        descriptor = bridge.handle(person)
        self.assertEqual(descriptor["type"], "Person")
        bridge.inspect(descriptor["handle"])
        self.assertEqual(events, [])

    def run_source(self, source, agent, extra=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text(source)
            for name, text in (extra or {}).items():
                (path.parent / name).write_text(text)
            config = ResolvedConfig(None, path.parent, "default", {"default": ProfileConfig("default", "fake", "test")})
            return run_script(path, config=config, agent_factory=lambda p: agent)

    def test_nearby_context_and_full_function_source(self):
        outer = self
        source = "def work():\n" + "    # padding\n" * 1000 + "    x = choose a value\n    return x\nanswer = work()\n"
        class Agent:
            def execute(self, request, runtime):
                nearby = runtime.code_context(request)["active"]
                full = runtime.get_frame_code()
                outer.assertTrue(nearby["truncated"])
                outer.assertIn("choose a value", nearby["code"])
                outer.assertLessEqual(len(nearby["code"]), 12000)
                outer.assertEqual(nearby["start_line"], request.span.line - 12)
                outer.assertTrue(full["code"].startswith("def work():"))
                outer.assertNotIn("answer = work()", full["code"])
                outer.assertEqual(request.frame_code, full["code"])
                old = json.dumps({"frame_code": request.frame_code})
                new = json.dumps({"code_context": runtime.code_context(request)})
                outer.assertLess(len(new), len(old) // 5)
                return 42
        self.assertEqual(self.run_source(source, Agent())["answer"], 42)

    def test_short_frame_includes_preceding_definitions(self):
        outer = self
        source = ("def choose():\n    return 42\n" + "# context\n" * 20 +
                  "answer = choose a value\n" + "# later\n" * 20)
        class Agent:
            def execute(self, request, runtime):
                context = runtime.code_context(request)["active"]
                outer.assertEqual(context["start_line"], 1)
                outer.assertIn("def choose():", context["code"])
                outer.assertNotIn("# later\n" * 20, context["code"])
                return 42
        self.assertEqual(self.run_source(source, Agent())["answer"], 42)

    def test_recovery_source_origin_and_active_are_distinct(self):
        outer = self
        class Agent:
            def recover(self, request, runtime):
                active = runtime.get_frame_code()
                origin = runtime.get_frame_code("origin")
                outer.assertIn("answer = work()", active["code"])
                outer.assertEqual(origin["code"], 'def work():\n    raise ValueError("bad")\n')
                outer.assertEqual(origin["start_line"], 1)
                context = runtime.code_context(request)
                outer.assertIn("origin", context)
                return RecoveryDecision("complete", 12, True)
        result = self.run_source('def work():\n    raise ValueError("bad")\nanswer = work()\n', Agent())
        self.assertEqual(result["answer"], 12)

    def test_huge_line_context_is_bounded_and_statement_preserved(self):
        bridge = self.bridge()
        statement = "do " + "a" * 13000
        request = AgentRequest(statement, statement, {}, {}, SourceSpan("test.py", 1, 0, 1, len(statement)),
                               ProfileConfig("default", "fake", "test"), ())
        context = bridge.code_context(request)["active"]
        self.assertEqual(len(context["code"]), 12000)
        self.assertTrue(context["truncated"])
        self.assertEqual(request.statement, statement)
