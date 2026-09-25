import sys
import tempfile
import unittest
from pathlib import Path

from aiython.cli import run_script
from aiython.models import AiythonError, ProfileConfig, RecoveryDecision, ResolvedConfig


class FakeAgent:
    def __init__(self, execute=None, recover=None):
        self.execution = execute
        self.recovery = recover
        self.requests = []
        self.errors = []

    def execute(self, request, runtime):
        self.requests.append(request)
        if self.execution:
            return self.execution(request, runtime)

    def recover(self, request, runtime):
        self.errors.append(request)
        if self.recovery:
            return self.recovery(request, runtime)
        return RecoveryDecision("reraise")


class RuntimeTests(unittest.TestCase):
    def run_source(self, source, agent=None, extra=None, force=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "main.py"
            path.write_text(source)
            for name, text in (extra or {}).items():
                (root / name).write_text(text)
            profiles = {n: ProfileConfig(n, "fake", n, prompt=f"{n} hint") for n in ("fast", "quality", "local")}
            config = ResolvedConfig(None, root, "fast", profiles, force_profile=force)
            return run_script(path, config=config, agent_factory=lambda p: agent or FakeAgent())

    def test_plain_python_and_handled_error_never_call_agent(self):
        agent = FakeAgent()
        result = self.run_source('''"module doc"
from __future__ import annotations
naïve = 3
try:
    1 / 0
except ZeroDivisionError:
    answer = naïve + 2
''', agent)
        self.assertEqual(result["answer"], 5)
        self.assertEqual(result["__doc__"], "module doc")
        self.assertFalse(agent.requests or agent.errors)

    def test_grouped_statement_module_and_identity(self):
        def execute(request, runtime):
            self.assertEqual(request.related_objects["items"], [])
            runtime.exec("x += 3\nz = x + y\nitems.append(z)")
        agent = FakeAgent(execute)
        result = self.run_source('''x = 10
y = 20
items = []
add 3 to x
set z to the sum of x plus y
append z to items
answer = (z, items)
''', agent)
        self.assertEqual(result["answer"], (33, [33]))
        self.assertEqual(len(agent.requests), 1)
        self.assertIs(agent.requests[0].related_objects["items"], result["items"])

    def test_function_locals_and_new_names(self):
        def execute(request, runtime):
            runtime.exec("x += 3\nz = x + y")
        agent = FakeAgent(execute)
        result = self.run_source('''def work(y):
    x = 10
    add 3 to x
    set z to the sum of x plus y
    return x, z
answer = [work(20), work(40)]
''', agent)
        self.assertEqual(result["answer"], [(13, 33), (13, 53)])
        self.assertEqual(len(agent.requests), 2)

    def test_expression_branch_loop(self):
        agent = FakeAgent(lambda request, runtime: runtime.eval("x < 3"))
        result = self.run_source('''x = 0
while x stays below 3:
    x += 1
if False:
    do impossible thing
''', agent)
        self.assertEqual(result["x"], 3)
        self.assertEqual(len(agent.requests), 4)

    def test_directives_and_force_profile(self):
        source = '''# aiython: begin profile="quality" prompt="outer"
# aiython: prompt="point"
one = do first thing
# aiython: begin profile="fast" prompt="inner"
two = do second thing
# aiython: end
# aiython: end
three = do third thing
'''
        agent = FakeAgent(lambda request, runtime: 1)
        self.run_source(source, agent)
        self.assertEqual([r.profile.name for r in agent.requests], ["quality", "fast", "fast"])
        self.assertEqual(agent.requests[0].prompts, ("quality hint", "outer", "point"))
        self.assertEqual(agent.requests[1].prompts, ("fast hint", "outer", "inner"))
        forced = FakeAgent(lambda request, runtime: 1)
        self.run_source(source, forced, force="local")
        self.assertTrue(all(r.profile.name == "local" for r in forced.requests))

    def test_recovery_original_finally_and_replacement(self):
        agent = FakeAgent(recover=lambda request, runtime: RecoveryDecision("complete", 42, True))
        result = self.run_source('''events = []
# aiython: profile="quality"
def work():
    try:
        events.append("called")
        raise ValueError("broken")
    finally:
        events.append("finally")
answer = work()
events.append(answer)
''', agent)
        self.assertEqual(result["events"], ["called", "finally", 42])
        self.assertEqual(agent.errors[0].profile.name, "quality")

    def test_retry_only_failed_checkpoint(self):
        def recover(request, runtime):
            runtime.exec("divisor = 2")
            return RecoveryDecision("retry")
        result = self.run_source('''events = ["before"]
divisor = 0
answer = 10 / divisor
events.append(answer)
''', FakeAgent(recover=recover))
        self.assertEqual(result["events"], ["before", 5])

    def test_loop_recovery_retries_only_failed_statement(self):
        def repair(request, bridge):
            self.assertEqual(request.statement, 'ticket.age')
            self.assertTrue(request.retry_allowed)
            bridge.exec('ticket.age = 7')
            return RecoveryDecision('retry')
        agent = FakeAgent(recover=repair)
        result = self.run_source('''class Ticket:
    pass
tickets = [Ticket(), Ticket(), Ticket()]
seen = []
for ticket in tickets:
    seen.append(ticket)
    ticket.age
''', agent)
        self.assertEqual(result['seen'], result['tickets'])
        self.assertTrue(all(ticket.age == 7 for ticket in result['tickets']))
        self.assertEqual(len(agent.errors), 3)
        self.assertTrue(all(request.attempt == 1 for request in agent.errors))

    def test_method_recovery_resumes_inside_method(self):
        def repair(request, bridge):
            self.assertEqual(request.statement, 'return self.age')
            bridge.exec('self.age = 7')
            return RecoveryDecision('retry')
        agent = FakeAgent(recover=repair)
        result = self.run_source('''seen = []
class Ticket:
    def read_age(self):
        seen.append(self)
        return self.age
ticket = Ticket()
answer = ticket.read_age()
''', agent)
        self.assertEqual(result['answer'], 7)
        self.assertEqual(result['seen'], [result['ticket']])
        self.assertEqual(len(agent.errors), 1)

    def test_class_and_method_docstrings_survive_nested_checkpoints(self):
        result = self.run_source('''class Ticket:
    """A ticket."""
    age: int = 7
    def read_age(self):
        """Read its age."""
        return self.age
answer = Ticket().read_age()
''')
        self.assertEqual(result['Ticket'].__doc__, 'A ticket.')
        self.assertEqual(result['Ticket'].read_age.__doc__, 'Read its age.')
        self.assertEqual(result['answer'], 7)
        self.assertFalse(any(name.startswith('__aiython_recovery_attempt_')
                             for name in vars(result['Ticket'])))

    def test_break_and_continue_keep_loop_control(self):
        result = self.run_source('''seen = []
for item in range(5):
    if item == 1:
        continue
    if item == 4:
        break
    seen.append(item)
''')
        self.assertEqual(result['seen'], [0, 2, 3])

    def test_user_try_handler_runs_before_ai_recovery(self):
        agent = FakeAgent()
        result = self.run_source('''class Ticket:
    pass
ticket = Ticket()
events = []
for _ in range(2):
    try:
        ticket.age
    except AttributeError:
        events.append('handled')
''', agent)
        self.assertEqual(result['events'], ['handled', 'handled'])
        self.assertFalse(agent.errors)

    def test_enclosing_loop_retry_is_rejected_without_replaying_effects(self):
        seen = []
        def repair(request, bridge):
            self.assertFalse(request.retry_allowed)
            seen.extend(bridge.eval('events'))
            return RecoveryDecision('retry')
        agent = FakeAgent(recover=repair)
        with self.assertRaisesRegex(AiythonError, 'Retry would replay'):
            self.run_source('''events = []
def fail():
    events.append('once')
    raise ValueError('broken')
for item in fail():
    pass
''', agent)
        self.assertEqual(len(agent.errors), 1)
        self.assertEqual(seen, ['once'])

    def test_recovery_limit(self):
        agent = FakeAgent(recover=lambda request, runtime: RecoveryDecision("retry"))
        with self.assertRaises(ZeroDivisionError):
            self.run_source("answer = 1 / 0", agent)
        self.assertEqual(len(agent.errors), 2)

    def test_exit_and_cancellation(self):
        for statement, error in ["raise SystemExit(7)", SystemExit], ["raise KeyboardInterrupt", KeyboardInterrupt], ["import concurrent.futures\nraise concurrent.futures.CancelledError()", __import__('concurrent.futures').futures.CancelledError]:
            agent = FakeAgent()
            with self.assertRaises(error):
                self.run_source(statement, agent)
            self.assertFalse(agent.errors)

    def test_import_syntax_and_runtime_directive(self):
        agent = FakeAgent(lambda request, runtime: 9,
                          lambda request, runtime: RecoveryDecision("complete", 15, True))
        try:
            result = self.run_source("import sample_aiython_module\nx = sample_aiython_module.answer\ny = sample_aiython_module.work()", agent, {
                "sample_aiython_module.py": '''answer = choose a number
# aiython: profile="quality"
def work():
    raise ValueError("broken")
'''
            })
            self.assertEqual((result["x"], result["y"]), (9, 15))
            self.assertEqual(agent.errors[0].profile.name, "quality")
        finally:
            sys.modules.pop("sample_aiython_module", None)

    def test_generator_coroutine_class_and_comprehension(self):
        agent = FakeAgent(lambda request, runtime: runtime.exec("new_value = seed + 1"))
        result = self.run_source('''import asyncio
def gen(seed):
    create new_value from seed
    yield new_value
    yield new_value + 1
async def coro(seed):
    create new_value from seed
    await asyncio.sleep(0)
    return new_value
class Thing:
    seed = 5
    create new_value from seed
    answer = new_value
a = gen(10)
b = gen(20)
answer = (next(a), next(b), next(a), asyncio.run(coro(30)), Thing.answer)
''', agent)
        self.assertEqual(result["answer"], (11, 21, 12, 31, 6))

    def test_nonlocal_and_global_preserved(self):
        agent = FakeAgent(lambda request, runtime: runtime.exec("x += 1"))
        result = self.run_source('''x = 100
def outer():
    x = 5
    def inner():
        nonlocal x
        increment x please
        return x
    return inner(), x
answer = outer()
''', agent)
        self.assertEqual(result["answer"], (6, 6))
        self.assertEqual(result["x"], 100)

    def test_explicit_global_write_through_exec(self):
        agent = FakeAgent(lambda request, runtime: runtime.exec("x += 1"))
        result = self.run_source('''x = 5
def f():
    global x
    increment x please
    return x
answer = f()
''', agent)
        self.assertEqual((result["x"], result["answer"]), (6, 6))

    def test_comprehensions_fstrings_and_argument_order(self):
        def execute(request, runtime):
            if "item" in request.statement:
                return runtime.eval("item * 2")
            if "name" in request.statement:
                return "Ada"
            return len(agent.requests)
        agent = FakeAgent(execute)
        result = self.run_source('''users = [1, 2, 3]
answer = [pick best item for item in users]
greeting = f"hello {choose best name}"
def collect(*args, **kwargs):
    return args, kwargs
arguments = collect(select first value, other=pick second value)
''', agent)
        self.assertEqual(result["answer"], [2, 4, 6])
        self.assertEqual(result["greeting"], "hello Ada")
        self.assertEqual(result["arguments"], ((5,), {"other": 6}))

    def test_super_closure_recursion_and_comprehension_scope(self):
        agent = FakeAgent(lambda request, runtime: runtime.exec("extra = n + 1"))
        result = self.run_source('''class Base:
    def value(self): return 10
class Child(Base):
    def value(self): return super().value() + 1
def recursive(n):
    create extra from n
    values = [extra + item for item in range(2)]
    if n:
        child = recursive(n - 1)
    return extra, values
answer = Child().value(), recursive(2)
''', agent)
        self.assertEqual(result["answer"], (11, (3, [3, 4])))

    def test_decorated_function_directives(self):
        agent = FakeAgent(recover=lambda request, runtime: RecoveryDecision("complete"))
        self.run_source('''def decorator(value):
    raise ValueError("failed decorator")
# aiython: profile="quality"
@decorator(
    1
)
def work():
    pass
''', agent)
        # Deepest project origin is the decorator's body, whose own context is
        # the default, rather than the annotated call site.
        self.assertEqual(agent.errors[0].profile.name, "fast")

    def test_exception_groups_and_cancellation(self):
        agent = FakeAgent(recover=lambda request, runtime: RecoveryDecision("complete"))
        self.run_source('raise ExceptionGroup("bad", [ValueError("one"), TypeError("two")])\nanswer=1', agent)
        self.assertEqual(len(agent.errors), 1)
        agent.errors.clear()
        with self.assertRaises(BaseExceptionGroup):
            self.run_source('raise BaseExceptionGroup("stop", [ValueError("bad"), KeyboardInterrupt()])', agent)
        self.assertFalse(agent.errors)
