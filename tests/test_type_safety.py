import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import Mock

from aiython.agent import ToolAgent
from aiython.cli import run_script
from aiython.models import ProfileConfig, ResolvedConfig
from aiython.type_constraints import TypeViolation, UnsupportedType, describe_output, validate_output


class TypeSafetyTests(unittest.TestCase):
    def run_source(self,source,agent=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'main.py'; path.write_text(source)
            profile = ProfileConfig('default','fake','model')
            config = ResolvedConfig(None,path.parent,'default',{'default':profile})
            return run_script(path,config=config,agent_factory=lambda p:agent)

    def test_assignment_reassignment_and_alias_mutation(self):
        for source in ('x: int = "bad"', 'x: int = 1\nx = "bad"',
                       'xs: list[int] = [1]\nalias = xs\nalias.append("bad")',
                       'xs: dict[str, list[int]] = {"a":[1]}\nxs["a"].append("bad")'):
            with self.subTest(source=source), self.assertRaises(TypeViolation): self.run_source(source)

    def test_argument_return_and_implicit_return(self):
        for source in ('def work(x: int):\n    return x\nwork("bad")',
                       'def work() -> int:\n    return "bad"\nwork()',
                       'def work() -> int:\n    pass\nwork()'):
            with self.subTest(source=source), self.assertRaises(TypeViolation): self.run_source(source)

    def test_shadowed_global_is_not_a_local_contract(self):
        result = self.run_source('x: int = 1\ndef work():\n    x = "allowed"\n    return x\nanswer = work()')
        self.assertEqual(result['answer'],'allowed')

    def test_global_write_is_checked(self):
        with self.assertRaises(TypeViolation):
            self.run_source('x: int = 1\ndef work():\n    global x\n    x = "bad"\nwork()')

    def test_class_attribute_checked_before_write(self):
        result = self.run_source('''from aiython.type_constraints import TypeViolation
class Person:
    age: int
    def __init__(self):
        self.age = 10
person = Person()
try:
    person.age = 'bad'
except TypeViolation:
    pass
answer = person.age
''')
        self.assertEqual(result['answer'],10)

    def test_parameter_variants_async_and_typevar(self):
        self.assertEqual(self.run_source('''import asyncio
from typing import TypeVar
T = TypeVar('T')
def identity(x: T) -> T:
    return x
async def work(x: int, /, *values: int, flag: bool = True, **extras: str) -> list[int]:
    return [x, *values]
answer = asyncio.run(work(identity(1),2,flag=True,name='ok'))
''')['answer'],[1,2])
        with self.assertRaises(TypeViolation):
            self.run_source('''from typing import TypeVar
T = TypeVar('T')
def broken(x: T) -> T:
    return 'wrong'
broken(1)
''')

    def test_unpack_typed_dict_keyword_arguments(self):
        prelude = '''from typing import NotRequired, TypedDict, Unpack
class Options(TypedDict):
    count: int
    label: NotRequired[str]
def describe(**kwargs: Unpack[Options]) -> tuple[int, str | None]:
    return kwargs['count'], kwargs.get('label')
'''
        result = self.run_source(prelude + "answer = describe(count=2, label='ready')\n")
        self.assertEqual(result['answer'], (2, 'ready'))
        for call in ('describe(count="bad")', 'describe(label="missing")',
                     'describe(count=2, label=3)'):
            with self.subTest(call=call), self.assertRaises(TypeViolation):
                self.run_source(prelude + call + '\n')

    def test_self_annotated_field_uses_instance_class(self):
        prelude = '''from typing import Self
class Node:
    next: Self | None
    def __init__(self):
        self.next = None
class Child(Node):
    pass
child = Child()
'''
        result = self.run_source(prelude + 'child.next = Child()\nanswer = isinstance(child.next, Child)\n')
        self.assertTrue(result['answer'])
        with self.assertRaises(TypeViolation):
            self.run_source(prelude + 'child.next = Node()\n')

    def test_self_in_methods_with_renamed_receiver(self):
        source = '''from typing import Self
class Base:
    def clone(this) -> Self:
        return type(this)()
    @classmethod
    def create(klass) -> Self:
        return klass()
    @property
    def same(this) -> Self:
        return this
    def keep(this):
        local: Self = this
        return local
class Child(Base):
    pass
child = Child()
answer = (type(child.clone()), type(Child.create()), type(child.same), type(child.keep()))
'''
        result = self.run_source(source)
        self.assertEqual(result['answer'], (result['Child'],) * 4)
        for method in ('''def clone(this) -> Self:
        return Base()''', '''@classmethod
    def create(klass) -> Self:
        return Base()'''):
            with self.subTest(method=method), self.assertRaises(TypeViolation):
                self.run_source('from typing import Self\nclass Base:\n    ' + method +
                                '\nclass Child(Base): pass\n' +
                                ('Child.create()\n' if 'classmethod' in method else 'Child().clone()\n'))

    def test_self_in_methods_assigned_after_class_creation(self):
        prelude = '''from typing import Self
class Base: pass
class Child(Base): pass
def clone(this) -> Self:
    return type(this)()
def create(klass) -> Self:
    return klass()
Base.clone = clone
Base.create = classmethod(create)
'''
        result = self.run_source(prelude + '''child = Child()
answer = (type(child.clone()), type(Child.create()))
''')
        self.assertEqual(result['answer'], (result['Child'], result['Child']))
        late = self.run_source('''from typing import Self as S
class Base: pass
class Child(Base): pass
def clone(this) -> S:
    return type(this)()
setattr(Base, 'clone', clone)
answer = type(Child().clone())
''')
        self.assertIs(late['answer'], late['Child'])
        with self.assertRaises(TypeViolation):
            self.run_source('''from typing import Self
class Base: pass
class Child(Base): pass
def wrong(this) -> Self:
    return Base()
Base.clone = wrong
Child().clone()
''')
        with self.assertRaises(TypeViolation):
            self.run_source('''from typing import Self
class Base: pass
class Child(Base): pass
def wrong(this) -> Self:
    return Base()
setattr(Base, 'clone', wrong)
Child().clone()
''')

    @unittest.skipIf(sys.version_info < (3, 12), "The type statement requires Python 3.12")
    def test_forward_local_alias_is_captured(self):
        self.assertEqual(self.run_source('''def factory():
    type Number = int
    def work(x: Number) -> Number:
        return x
    return work
answer = factory()(3)
''')['answer'],3)

    def test_final_and_loop_bindings(self):
        with self.assertRaises(TypeViolation): self.run_source('from typing import Final\nx: Final[int] = 1\nx = 2')
        with self.assertRaises(TypeViolation): self.run_source('x: int = 1\nfor x in ["bad"]:\n    pass')

    @unittest.skipIf(sys.version_info < (3, 12), "The type statement requires Python 3.12")
    def test_recursive_alias_and_generic_alias(self):
        namespace = {}
        exec('type Tree = int | list[Tree]\ntype Box[T] = list[T]',namespace)
        validate_output([1,[2]],'Tree',namespace)
        validate_output([1,2],'Box[int]',namespace)
        with self.assertRaises(TypeViolation): validate_output([1,['bad']],'Tree',namespace)

    def test_typed_dict_schema_literals_and_descriptions(self):
        namespace = {}
        exec('''from typing import TypedDict, Literal, Annotated, NotRequired
class TicketAnalysis(TypedDict):
    severity: Literal['high','low']
    summary: Annotated[str,'One short English sentence']
    note: NotRequired[str]
''',namespace)
        schema = describe_output('TicketAnalysis',namespace)
        self.assertEqual(schema['properties']['severity']['enum'],['high','low'])
        self.assertEqual(schema['properties']['summary']['description'],'One short English sentence')
        self.assertNotIn('note',schema['required'])
        validate_output({'severity':'high','summary':'short'},'TicketAnalysis',namespace)
        with self.assertRaisesRegex(TypeViolation,'severity'):
            validate_output({'severity':'wrong','summary':'short'},'TicketAnalysis',namespace)

    def test_future_typeddict_optional_fields(self):
        namespace = {}
        exec('''from __future__ import annotations
from typing import TypedDict, NotRequired
class Record(TypedDict):
    value: int
    note: NotRequired[str]
''',namespace)
        validate_output({'value':1},'Record',namespace)

    def test_annotation_calls_never_execute(self):
        touched = []
        with self.assertRaises(UnsupportedType):
            describe_output('danger()',{'danger':lambda:touched.append(1)})
        self.assertEqual(touched,[])

    def test_ai_gets_schema_from_return_annotation_and_repairs_before_return(self):
        provider = Mock()
        calls = [
            [{'id':'e1','type':'function','function':{'name':'evaluate','arguments':json.dumps({'code':'{"severity":"wrong"}','result_id':'r'})}},
             {'id':'f1','type':'function','function':{'name':'finish','arguments':json.dumps({'result_from':'r'})}}],
            [{'id':'e2','type':'function','function':{'name':'evaluate','arguments':json.dumps({'code':'{"severity":"high"}','result_id':'r'})}},
             {'id':'f2','type':'function','function':{'name':'finish','arguments':json.dumps({'result_from':'r'})}}]]
        payloads = []
        def complete(messages,tools):
            payloads.append(json.loads(messages[1]['content']))
            return {'role':'assistant','tool_calls':calls[len(payloads)-1]}
        provider.complete.side_effect = complete
        result = self.run_source('''from typing import TypedDict, Literal
class Result(TypedDict):
    severity: Literal['high','low']
def analyze() -> Result:
    return analyze this incident
answer = analyze()
''',ToolAgent(provider))
        self.assertEqual(result['answer'],{'severity':'high'})
        self.assertEqual(payloads[0]['output_schema']['properties']['severity']['enum'],['high','low'])
        self.assertEqual(provider.complete.call_count,2)

    def test_ai_exec_cannot_break_declared_binding(self):
        class Agent:
            def execute(self,request,runtime): runtime.exec('x = "bad"')
        with self.assertRaises(TypeViolation):
            self.run_source('x: int = 1\nchange x now',Agent())

    def test_returned_closure_nonlocal_binding_is_checked(self):
        with self.assertRaises(TypeViolation):
            self.run_source('''def factory():
    x: int = 1
    def change():
        nonlocal x
        x = 'bad'
    return change
factory()()
''')

    def test_generators_yield_send_return_and_async_generator(self):
        self.assertEqual(self.run_source('''import asyncio
from typing import Generator, AsyncIterator
async def ag() -> AsyncIterator[int]:
    yield 1
async def consume():
    return [item async for item in ag()]
def gen() -> Generator[int, str, int]:
    received = yield 1
    return len(received)
g = gen()
first = next(g)
try:
    g.send('ok')
except StopIteration as stop:
    returned = stop.value
answer = (first, returned, asyncio.run(consume()))
''')['answer'],(1,2,[1]))
        for source in ('from typing import Iterator\ndef gen() -> Iterator[int]:\n    yield "bad"\nnext(gen())',
                       'from typing import Generator\ndef gen() -> Generator[int, str, None]:\n    yield 1\ng = gen()\nnext(g)\ng.send(2)'):
            with self.assertRaises(TypeViolation): self.run_source(source)

    def test_yield_from_forwards_send_and_return(self):
        result = self.run_source('''from typing import Generator
def child():
    text = yield 1
    return len(text)
def parent() -> Generator[int, str, int]:
    return (yield from child())
g = parent()
next(g)
try:
    g.send('ok')
except StopIteration as stop:
    answer = stop.value
''')
        self.assertEqual(result['answer'],2)

    def test_generic_class_fields(self):
        namespace = {}
        exec('''from typing import Generic, TypeVar
T = TypeVar('T')
class Box(Generic[T]):
    value: T
    def __init__(self,value): self.value = value
''',namespace)
        validate_output(namespace['Box'](1),'Box[int]',namespace)
        with self.assertRaises(TypeViolation): validate_output(namespace['Box']('bad'),'Box[int]',namespace)

    def test_finally_cannot_invalidate_a_checked_return(self):
        with self.assertRaisesRegex(TypeViolation,'return'):
            self.run_source('''def work() -> list[int]:
    result = [1]
    try:
        return result
    finally:
        result.append('bad')
work()
''')

    def test_class_field_mutations_inside_untyped_containers(self):
        with self.assertRaises(TypeViolation):
            self.run_source('''class Group:
    values: list[int]
    def __init__(self):
        self.values = [1]
groups = [Group()]
groups[0].values.append('bad')
''')

    def test_dataclass_constructor_fields_checked_without_variable_annotation(self):
        with self.assertRaises(TypeViolation):
            self.run_source('''from dataclasses import dataclass
@dataclass
class Item:
    count: int
item = Item('bad')
''')

    def test_typed_natural_language_keeps_subscript_inside_statement(self):
        from aiython.frontend import parse
        unit = parse('analysis: TicketAnalysis = analyze the ticket from ticket["message"]\n','test.py')
        block = next(iter(unit.blocks.values()))
        self.assertEqual(block.output_type,'TicketAnalysis')
        self.assertTrue(block.statement.endswith('ticket["message"]'))

    def test_enum_types_and_enum_literals(self):
        from enum import Enum
        class Level(Enum):
            HIGH = 'high'
            LOW = 'low'
        namespace = {'Level':Level}
        validate_output(Level.HIGH,'Literal[Level.HIGH]',namespace)
        with self.assertRaises(TypeViolation): validate_output(Level.LOW,'Literal[Level.HIGH]',namespace)
        self.assertEqual(describe_output('Level',namespace)['enum'],['high','low'])
        json.dumps(describe_output('Literal[Level.HIGH]',namespace))

    def test_unsupported_contract_fails_before_model_call(self):
        provider = Mock()
        with self.assertRaises(UnsupportedType):
            self.run_source('from typing import LiteralString\nanswer: LiteralString = choose a string',ToolAgent(provider))
        provider.complete.assert_not_called()

    def test_ai_expected_type_from_function_argument_and_generic_return(self):
        class Agent:
            def __init__(self): self.types = []
            def execute(self,request,runtime):
                self.types.append(describe_output(request.output_type,runtime.manager.types.namespace(runtime.frame)))
                return 3
        agent = Agent()
        self.run_source('''from typing import TypeVar
T = TypeVar('T')
def consume(value: int):
    return value
def choose(value: T) -> T:
    return choose a similar value
answer = consume(choose a number)
other = choose(1)
''',agent)
        self.assertEqual([schema['type'] for schema in agent.types],['integer','integer'])

    def test_semicolon_after_natural_assignment_stays_python(self):
        class Agent:
            def execute(self,request,runtime): return 4
        result = self.run_source('x: int = choose a number; answer = x + 1',Agent())
        self.assertEqual(result['answer'],5)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 syntax requires Python 3.12")
    def test_pep695_generic_class_and_function(self):
        with self.assertRaises(TypeViolation):
            self.run_source('''class Box[T]:
    value: T
    def __init__(self,value):
        self.value = value
box = Box[int]('wrong')
''')
        result = self.run_source('''def identity[T](value: T) -> T:
    return value
answer = identity(3)
''')
        self.assertEqual(result['answer'],3)
