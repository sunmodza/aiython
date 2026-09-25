import asyncio
from concurrent.futures import ThreadPoolExecutor
try:
    from concurrent.futures import InterpreterPoolExecutor
except ImportError:
    InterpreterPoolExecutor = None
from dataclasses import replace
import json
import inspect
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from aiython import current, group, join, worker_entry
from aiython.agent import ToolAgent
from aiython.capabilities import CapabilityResult
from aiython.cli import run_script
from aiython.models import AgentRequest, ProfileConfig, ProviderError, ResolvedConfig, SourceSpan
from aiython.runtime import Runtime, RuntimeBridge
from aiython.providers import LiteLLMProvider
from tests._plain_worker import send as plain_interpreter_send


class CollaborationTests(unittest.TestCase):
    def test_closing_one_group_keeps_other_group_available(self):
        with group() as first:
            first_ticket = first.invite("worker")
            with group() as second:
                second.invite("worker")
            with join(first_ticket) as participant:
                participant.send("main", "alive")
            self.assertEqual(first.read()[0]["payload"], "alive")

    def test_thread_messages_preserve_per_sender_order_and_identity(self):
        with group() as team:
            ticket = team.invite("worker")
            def work():
                with join(ticket) as participant:
                    first = participant.send("main", {"number": 1}, message_id="one")
                    duplicate = participant.send("main", {"number": 1}, message_id="one")
                    with self.assertRaisesRegex(ValueError, "different content"):
                        participant.send("main", {"number": 999}, message_id="one")
                    participant.send("main", {"number": 2})
                    return first["id"] == duplicate["id"]
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertTrue(pool.submit(work).result())
            messages = team.read()
            self.assertEqual([item["payload"]["number"] for item in messages], [1, 2])
            self.assertEqual([item["sequence"] for item in messages], [1, 2])

    def test_ticket_is_scoped_to_one_participant_and_mailbox_is_bounded(self):
        with group() as team:
            ticket = team.invite("worker")
            with self.assertRaisesRegex(ValueError, "ticket is invalid"):
                with join(replace(ticket, participant="main")):
                    pass
            with join(ticket) as participant:
                for number in range(100):
                    participant.send("main", number)
                with self.assertRaisesRegex(BufferError, "mailbox is full"):
                    participant.send("main", 100)
            self.assertEqual(len(team.read(50)) + len(team.read(50)), 100)

    def test_asyncio_tasks_can_wait_without_blocking_the_event_loop(self):
        async def scenario():
            with group() as team:
                ticket = team.invite("worker")
                async def work():
                    self.assertIsNone(current())
                    async with join(ticket) as participant:
                        messages = await participant.wait_async(2)
                        participant.send("main", messages[0]["payload"] + 1)
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(work())
                    await asyncio.sleep(0)
                    team.send("worker", 41)
                return team.read()[0]["payload"]
        self.assertEqual(asyncio.run(scenario()), 42)

    def test_cancelled_async_wait_does_not_consume_future_message(self):
        async def scenario():
            with group() as team:
                ticket = team.invite("worker")
                ready = asyncio.Event()
                async def wait_for_message():
                    async with join(ticket) as participant:
                        ready.set()
                        await participant.wait_async(30)
                waiter = asyncio.create_task(wait_for_message())
                await ready.wait()
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                team.send("worker", "still here")
                async with join(ticket) as participant:
                    return (await participant.wait_async(1))[0]["payload"]
        self.assertEqual(asyncio.run(scenario()), "still here")

    def test_process_start_methods_import_aiython_source_and_exchange_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aiython.toml").write_text('version = 3\nmodel = "openai/test"\n')
            (root / "worker.py").write_text(
                'from aiython import join\n'
                'def work(ticket):\n'
                '    with join(ticket) as me:\n'
                '        me.send("main", {"ok": True})\n'
                '        return 12\n'
                'def unused():\n'
                '    return choose a value\n')
            (root / "main.py").write_text(
                'from concurrent.futures import ProcessPoolExecutor\n'
                'from multiprocessing import get_all_start_methods, get_context\n'
                'from aiython import group, worker_entry\n'
                'if __name__ == "__main__":\n'
                '    with group() as team:\n'
                '        answer = []\n'
                '        for method in ("spawn", "forkserver"):\n'
                '            if method not in get_all_start_methods():\n'
                '                continue\n'
                '            ticket = team.invite(method)\n'
                '            with ProcessPoolExecutor(max_workers=1, mp_context=get_context(method)) as pool:\n'
                '                result = pool.submit(worker_entry, ticket, "worker", "work").result(timeout=20)\n'
                '                answer.append((result, team.read()[0]["payload"]))\n')
            self.assertEqual(run_script(root / "main.py")["answer"],
                             [(12, {"ok": True}), (12, {"ok": True})])

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_plain_interpreter_worker_can_join_directly(self):
        with group() as team:
            ticket = team.invite("worker")
            with InterpreterPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(plain_interpreter_send, ticket).result(timeout=20), 7)
            self.assertEqual(team.read()[0]["payload"], 42)

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_interpreter_worker_entry_plain_python_uses_same_mailbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text(
                'from aiython import join\n'
                'def work(ticket):\n'
                '    with join(ticket) as me:\n'
                '        me.send("main", 42)\n'
                '        return 7\n')
            with group(root) as team:
                ticket = team.invite("worker")
                with InterpreterPoolExecutor(max_workers=1) as pool:
                    self.assertEqual(pool.submit(worker_entry, ticket, "worker", "work").result(timeout=20), 7)
                self.assertEqual(team.read()[0]["payload"], 42)

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_interpreter_worker_entry_reuses_one_fallback_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "effects.txt"
            (root / "worker.py").write_text('''import os
from pathlib import Path
def work(ticket, marker, fail=False):
    path = Path(marker)
    path.write_text((path.read_text() if path.exists() else "") + "x")
    if fail:
        raise ValueError("test failure")
    return os.getpid()
''')
            with group(root) as team:
                ticket = team.invite("worker")
                with InterpreterPoolExecutor(max_workers=1) as pool:
                    first = pool.submit(worker_entry, ticket, "worker", "work", str(marker)).result(timeout=20)
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        pool.submit(worker_entry, ticket, "worker", "work", str(marker), True).result(timeout=20)
                    third = pool.submit(worker_entry, ticket, "worker", "work", str(marker)).result(timeout=20)
            self.assertNotEqual(first, os.getpid())
            self.assertEqual(first, third)
            self.assertEqual(marker.read_text(), "xxx")

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_crashed_fallback_worker_does_not_replay_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "effects.txt"
            (root / "worker.py").write_text('''import os
from pathlib import Path
def work(ticket, marker, crash=False):
    path = Path(marker)
    path.write_text((path.read_text() if path.exists() else "") + "x")
    if crash:
        os._exit(17)
    return os.getpid()
''')
            with group(root) as team:
                ticket = team.invite("worker")
                with InterpreterPoolExecutor(max_workers=1) as pool:
                    with self.assertRaisesRegex(RuntimeError, "status 17"):
                        pool.submit(worker_entry, ticket, "worker", "work", str(marker), True).result(timeout=20)
                    self.assertIsInstance(pool.submit(worker_entry, ticket, "worker", "work", str(marker)).result(timeout=20), int)
            self.assertEqual(marker.read_text(), "xx")

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_interpreter_worker_runs_aiython_source_with_native_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aiython.toml").write_text('version = 3\nmodel = "openai/test"\n')
            (root / "worker.py").write_text('''import json
from unittest.mock import patch
from aiython import join
class FakeSDK:
    def completion(self, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "done", "type": "function", "function": {
                "name": "finish", "arguments": json.dumps({"outcome": {"kind": "literal", "value": 42}})}}]}}]}
def work(ticket):
    with join(ticket) as participant:
        with patch("aiython.providers.sdk", return_value=FakeSDK()):
            value = choose a number
        participant.send("main", value)
        return value
''')
            with group(root) as team:
                ticket = team.invite("worker")
                with InterpreterPoolExecutor(max_workers=1) as pool:
                    self.assertEqual(pool.submit(worker_entry, ticket, "worker", "work").result(timeout=20), 42)
                self.assertEqual([item["payload"] for item in team.read()], [42])

    def test_worker_runtime_syntax_error_does_not_replay_import_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "imports.txt"
            (root / "worker.py").write_text(
                f'from pathlib import Path\nmarker = Path({str(marker)!r})\n'
                'marker.write_text((marker.read_text() if marker.exists() else "") + "x")\n'
                'exec("this is ?")\ndef work(ticket):\n    return 1\n')
            with group(root) as team:
                with self.assertRaises(SyntaxError):
                    worker_entry(team.invite("worker"), "worker", "work")
            self.assertEqual(marker.read_text(), "x")

    @unittest.skipUnless(InterpreterPoolExecutor, "InterpreterPoolExecutor requires Python 3.14")
    def test_interpreter_worker_loads_imported_aiython_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aiython.toml").write_text('version = 3\nmodel = "openai/test"\n')
            (root / "nested.py").write_text(
                'def get_number():\n    value = choose a number\n    return value\n')
            (root / "worker.py").write_text('''import json
from unittest.mock import patch
from aiython import join
class FakeSDK:
    def completion(self, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "done", "type": "function", "function": {
                "name": "finish", "arguments": json.dumps({"outcome": {"kind": "literal", "value": 42}})}}]}}]}
def work(ticket):
    with join(ticket) as participant:
        with patch("aiython.providers.sdk", return_value=FakeSDK()):
            from nested import get_number
            value = get_number()
        participant.send("main", value)
        return value
''')
            with group(root) as team:
                ticket = team.invite("worker")
                with InterpreterPoolExecutor(max_workers=1) as pool:
                    self.assertEqual(pool.submit(worker_entry, ticket, "worker", "work").result(timeout=20), 42)
                self.assertEqual([item["payload"] for item in team.read()], [42])

    def test_agent_message_tools_are_only_available_when_joined(self):
        calls = []
        class Provider:
            def complete(self, messages, tools):
                names = {tool["function"]["name"] for tool in tools}
                calls.append(names)
                operations = ([{"id": "send", "type": "function", "function": {
                    "name": "send_message", "arguments": json.dumps({"recipient": "worker", "payload": 42})}}]
                    if "send_message" in names else [])
                operations.append({"id": "finish", "type": "function", "function": {
                    "name": "finish", "arguments": json.dumps({"outcome": {"kind": "literal", "value": 7}})}})
                return {"role": "assistant", "content": None, "tool_calls": operations}
        config = ResolvedConfig(None, Path.cwd())
        runtime = Runtime(config)
        request = AgentRequest("compute", "answer = 7", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 7),
                               ProfileConfig("test", "fake", "test", max_rounds=2), ())
        agent = ToolAgent(Provider())
        self.assertEqual(agent.execute(request, RuntimeBridge(__import__("inspect").currentframe(), runtime)), 7)
        with group() as team:
            team.invite("worker")
            self.assertEqual(agent.execute(request, RuntimeBridge(__import__("inspect").currentframe(), runtime)), 7)
            self.assertEqual(team.pending(), 0)
        self.assertNotIn("send_message", calls[0])
        self.assertIn("send_message", calls[1])

    def test_agent_can_return_messages_in_one_model_response(self):
        class Provider:
            calls = 0
            def complete(self, messages, tools):
                self.calls += 1
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "read", "type": "function", "function": {"name": "read_messages",
                     "arguments": json.dumps({"result_id": "inbox"})}},
                    {"id": "done", "type": "function", "function": {"name": "finish",
                     "arguments": json.dumps({"outcome": {"kind": "reference", "id": "inbox"}})}},
                ]}
        config = ResolvedConfig(None, Path.cwd())
        runtime = Runtime(config)
        request = AgentRequest("return inbox", "return inbox", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 12),
                               ProfileConfig("test", "fake", "test", max_rounds=2), ())
        provider = Provider()
        with group() as team:
            ticket = team.invite("worker")
            with join(ticket) as participant:
                participant.send("main", 42)
            result = ToolAgent(provider).execute(
                request, RuntimeBridge(__import__("inspect").currentframe(), runtime))
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result[0]["payload"], 42)

    @unittest.skipUnless(importlib.util.find_spec("a2a"), "optional A2A SDK is not installed")
    def test_a2a_uses_sdk_once_and_preserves_group_metadata(self):
        from a2a.helpers import new_text_message
        from a2a.types import Role, StreamResponse
        from aiython.a2a import send_a2a

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def send_message(self, request):
                content = json.loads(request.message.parts[0].text)
                self_outer.assertEqual(content, {"payload": {"question": "hi"},
                                                 "group_id": "group", "sender": "worker"})
                yield StreamResponse(message=new_text_message("answer", role=Role.ROLE_AGENT))

        self_outer = self
        factory = AsyncMock(return_value=Client())
        with patch("a2a.client.create_client", factory):
            response = asyncio.run(send_a2a("https://example.test", {"question": "hi"},
                                             group_id="group", sender="worker"))
        self.assertEqual(factory.await_count, 1)
        self.assertEqual(response[0]["message"]["parts"][0]["text"], "answer")
        with patch.dict(os.environ, {"AIYTHON_TEST_A2A_KEY": "private-key"}), \
                patch("a2a.client.create_client", factory):
            asyncio.run(send_a2a("https://example.test", {"question": "hi"},
                                 group_id="group", sender="worker",
                                 api_key_env="AIYTHON_TEST_A2A_KEY"))
        client_config = factory.await_args.kwargs["client_config"]
        self.assertEqual(client_config.httpx_client.headers["Authorization"], "Bearer private-key")


class AsyncRuntimeTests(unittest.TestCase):
    def test_threads_execute_separate_ai_frames_concurrently(self):
        barrier = threading.Barrier(2)
        class Agent:
            def execute(self, request, bridge):
                barrier.wait(timeout=5)
                bridge.exec("effects.append(n)")
                return bridge.eval("n * 2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "main.py"
            path.write_text('''from concurrent.futures import ThreadPoolExecutor
effects = []
def work(n):
    value = choose a value for n
    return value
with ThreadPoolExecutor(max_workers=2) as pool:
    answer = list(pool.map(work, [1, 2]))
''')
            profile = ProfileConfig("default", "fake", "test")
            config = ResolvedConfig(None, root, "default", {"default": profile})
            result = run_script(path, config=config, agent_factory=lambda _: Agent())
            self.assertEqual(result["answer"], [2, 4])
            self.assertEqual(sorted(result["effects"]), [1, 2])

    def test_async_ai_calls_overlap_and_touch_own_live_frames(self):
        events = []
        class Agent:
            async def aexecute(self, request, bridge):
                events.append(("start", bridge.eval("n")))
                await asyncio.sleep(0.01)
                bridge.exec("effects.append(n)")
                events.append(("end", bridge.eval("n")))
                return bridge.eval("n * 2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "main.py"
            path.write_text('''import asyncio
effects = []
async def work(n):
    value = choose a value for n
    return value
async def run():
    return await asyncio.gather(work(1), work(2))
answer = asyncio.run(run())
''')
            profile = ProfileConfig("default", "fake", "test")
            config = ResolvedConfig(None, root, "default", {"default": profile})
            result = run_script(path, config=config, agent_factory=lambda _: Agent())
            self.assertEqual(result["answer"], [2, 4])
            self.assertEqual(sorted(result["effects"]), [1, 2])
            self.assertEqual([event[0] for event in events], ["start", "start", "end", "end"])


class AsyncProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_capability_adapter_preserves_live_object_identity(self):
        owner = threading.get_ident()
        entered = threading.Event()
        release = threading.Event()
        records = [{"text": "hello", "source": "catalog:1"}]
        class Provider:
            async def acomplete(self, messages, tools):
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "plan", "type": "function", "function": {"name": "run_plan",
                     "arguments": json.dumps({"steps": [{"id": "rank", "capability": "reranking",
                                                       "params": {"query": "hello",
                                                                  "documents": {"$binding": "records"}}}],
                                              "output": {"$ref": "rank"}, "result_id": "answer"})}},
                    {"id": "done", "type": "function", "function": {"name": "finish",
                     "arguments": json.dumps({"outcome": {"kind": "reference", "id": "answer"}})}},
                ]}
        class Adapter:
            version = "test"
            api_base = None
            def capabilities(self):
                return {"reranking"}
            def invoke(self, request, context):
                self_outer.assertNotEqual(threading.get_ident(), owner)
                self_outer.assertIs(request.params["documents"], records)
                entered.set()
                self_outer.assertTrue(release.wait(3))
                return CapabilityResult(request.params["documents"])
        self_outer = self
        profile = ProfileConfig("test", "fake", "base", max_rounds=1,
                                routes={"reranking": [{"provider": "fake", "model": "other"}]})
        config = ResolvedConfig(None, Path.cwd(), "test", {"test": profile})
        runtime = Runtime(config)
        runtime.capabilities.adapters[("test", "fake")] = Adapter()
        request = AgentRequest("rank records", "rank records", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 12), profile, (), requires_result=True)
        bridge = RuntimeBridge(inspect.currentframe(), runtime)
        task = asyncio.create_task(ToolAgent(Provider()).aexecute(request, bridge))
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        release.set()
        self.assertIs(await task, records)

    async def test_invalid_async_batch_has_no_capability_side_effect(self):
        calls = []
        class Provider:
            round = 0
            async def acomplete(self, messages, tools):
                self.round += 1
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"plan-{self.round}", "type": "function", "function": {"name": "run_plan",
                     "arguments": json.dumps({"steps": [{"id": "one", "capability": "reasoning",
                                                       "params": {"prompt": "test"}}],
                                              "output": {"$ref": "one"}})}},
                    {"id": f"finish-{self.round}", "type": "function", "function": {"name": "finish",
                     "arguments": json.dumps({"outcome": {"kind": "literal", "value": 1,
                                                         "code": "2"}})}},
                ]}
        profile = ProfileConfig("test", "fake", "test", max_rounds=2)
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        request = AgentRequest("compute", "compute", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 7), profile, ())
        with patch.object(runtime.capabilities, "run_plan", lambda *a, **k: calls.append(1)):
            with self.assertRaisesRegex(ProviderError, "repeated invalid tool batches"):
                await ToolAgent(Provider()).aexecute(request,
                    RuntimeBridge(inspect.currentframe(), runtime))
        self.assertEqual(calls, [])

    async def test_async_plan_failure_reports_completed_steps_without_replay(self):
        seen_results = []
        class Provider:
            async def acomplete(self, messages, tools):
                errors = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
                if errors:
                    seen_results.extend(errors)
                    return {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "done", "type": "function", "function": {"name": "finish",
                         "arguments": json.dumps({"outcome": {"kind": "literal", "value": 42}})}}]}
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "plan", "type": "function", "function": {"name": "run_plan",
                     "arguments": json.dumps({"steps": [{"id": "one", "capability": "reasoning",
                                                       "params": {"prompt": "test"}}],
                                              "output": {"$ref": "one"}})}}]}
        def run_plan(profile, steps, output, worker_bridge, **kwargs):
            worker_bridge.completed_steps["one"] = ("fingerprint", 42)
            raise ValueError("later step failed")
        profile = ProfileConfig("test", "fake", "test", max_rounds=2)
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        request = AgentRequest("compute", "compute", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 7), profile, ())
        bridge = RuntimeBridge(inspect.currentframe(), runtime)
        with patch.object(runtime.capabilities, "run_plan", run_plan):
            self.assertEqual(await ToolAgent(Provider()).aexecute(request, bridge), 42)
        self.assertEqual(bridge.completed_steps["one"][1], 42)
        self.assertEqual(seen_results[0]["completed_steps"]["one"]["value"], 42)

    async def test_capability_plan_does_not_block_event_loop_and_preserves_batch_order(self):
        owner = threading.get_ident()
        entered = threading.Event()
        release = threading.Event()
        effects = []
        class Provider:
            async def acomplete(self, messages, tools):
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "mutate", "type": "function", "function": {"name": "execute",
                     "arguments": json.dumps({"code": "effects.append(1)\nassigned = 42"})}},
                    {"id": "plan", "type": "function", "function": {"name": "run_plan",
                     "arguments": json.dumps({"steps": [{"id": "one", "capability": "reasoning",
                                                       "params": {"prompt": "test"}}],
                                              "output": {"$ref": "one"}, "result_id": "answer"})}},
                    {"id": "finish", "type": "function", "function": {"name": "finish",
                     "arguments": json.dumps({"outcome": {"kind": "reference", "id": "answer"}})}},
                ]}
        def run_plan(profile, steps, output, bridge, **kwargs):
            self.assertNotEqual(threading.get_ident(), owner)
            self.assertEqual(bridge.namespace()["effects"], [1])
            self.assertEqual(bridge.namespace()["assigned"], 42)
            entered.set()
            self.assertTrue(release.wait(3))
            bridge.completed_steps["one"] = ("fingerprint", 42)
            return 42
        profile = ProfileConfig("test", "fake", "test", max_rounds=2)
        config = ResolvedConfig(None, Path.cwd())
        runtime = Runtime(config)
        request = AgentRequest("compute", "answer = compute", {}, {},
                               SourceSpan("test.py", 1, 0, 1, 7), profile, (), requires_result=True)
        bridge = RuntimeBridge(inspect.currentframe(), runtime)
        with patch.object(runtime.capabilities, "run_plan", run_plan):
            task = asyncio.create_task(ToolAgent(Provider()).aexecute(request, bridge))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            self.assertEqual(effects, [1])
            release.set()
            self.assertEqual(await task, 42)
        self.assertEqual(bridge.completed_steps["one"][1], 42)

    async def test_resume_job_does_not_block_event_loop(self):
        owner = threading.get_ident()
        entered = threading.Event()
        release = threading.Event()
        profile = ProfileConfig("test", "fake", "test")
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        bridge = RuntimeBridge(inspect.currentframe(), runtime)
        bridge.ai_request = AgentRequest("resume", "resume", {}, {},
                                         SourceSpan("test.py", 1, 0, 1, 6), profile, ())
        def resume_job(profile, operation):
            self.assertNotEqual(threading.get_ident(), owner)
            self.assertEqual(operation, "job-1")
            entered.set()
            self.assertTrue(release.wait(3))
            return {"done": True}
        with patch.object(runtime.capabilities, "resume_job", resume_job):
            task = asyncio.create_task(ToolAgent(None).dispatch_async(
                "resume_job", {"operation": "job-1"}, bridge, {}))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            release.set()
            result = await task
        self.assertIs(bridge.dereference(result["handle"])["done"], True)

    async def test_cancelled_plan_waits_for_worker_and_retains_completed_steps(self):
        entered = threading.Event()
        release = threading.Event()
        profile = ProfileConfig("test", "fake", "test")
        runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        bridge = RuntimeBridge(inspect.currentframe(), runtime)
        bridge.ai_request = AgentRequest("compute", "compute", {}, {},
                                         SourceSpan("test.py", 1, 0, 1, 7), profile, ())
        def run_plan(profile, steps, output, worker_bridge, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            worker_bridge.completed_steps["one"] = ("fingerprint", 42)
            return 42
        with patch.object(runtime.capabilities, "run_plan", run_plan):
            task = asyncio.create_task(ToolAgent(None).dispatch_async(
                "run_plan", {"steps": [], "output": {}}, bridge, {}))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(bridge.completed_steps["one"][1], 42)

    async def test_litellm_async_gateway_does_not_call_sync_completion(self):
        profile = ProfileConfig("test", "litellm", "openai/test")
        profile.routes["reasoning"] = [{"provider": "route", "model": "openai/test"}]
        config = ResolvedConfig(None, Path.cwd(), providers={"route": {"model": "openai/test"}})
        sdk = SimpleNamespace(acompletion=AsyncMock(return_value={"choices": [{"message": {
            "role": "assistant", "content": None, "tool_calls": []}}]}))
        with patch("aiython.providers.sdk", return_value=sdk):
            result = await LiteLLMProvider(config, profile).acomplete([{"role": "user", "content": "hi"}], [])
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(sdk.acompletion.await_count, 1)
        self.assertEqual(sdk.acompletion.await_args.kwargs["max_retries"], 0)
