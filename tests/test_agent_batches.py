import contextlib
import inspect
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from aiython.agent import ToolAgent, validate_schema
from aiython.models import AgentRequest, ProfileConfig, ProviderError, RecoveryRequest, ResolvedConfig, SourceSpan
from aiython.prompt_cache import canonical
from aiython.providers import LiteLLMProvider
from aiython.runtime import Runtime, RuntimeBridge
from aiython.stats import CURRENT_STATS, InvocationStats, record_token_usage


def call(id, tool_name, **args):
    return {"id": id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}


def response(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class BatchTests(unittest.TestCase):
    def test_repeated_failed_terminal_stops_before_round_limit(self):
        provider = Mock()
        bad = '(_ for _ in ()).throw(RuntimeError("capability unavailable"))'
        provider.complete.side_effect = [
            response(call('first', 'finish', code=bad)),
            response(call('second', 'finish', code=bad)),
        ]

        with self.assertRaisesRegex(ProviderError, 'repeated failed finish call.*capability unavailable'):
            ToolAgent(provider).execute(self.request(requires_result=True), self.bridge(inspect.currentframe()))

        self.assertEqual(provider.complete.call_count, 2)

    def test_successful_tool_call_resets_repeated_failure_detection(self):
        provider = Mock()
        provider.complete.side_effect = [
            response(call('first', 'evaluate', code='1 / 0')),
            response(call('progress', 'evaluate', code='42')),
            response(call('third', 'evaluate', code='1 / 0')),
            response(call('done', 'finish', value=42)),
        ]

        result = ToolAgent(provider).execute(self.request(requires_result=True), self.bridge(inspect.currentframe()))

        self.assertEqual(result, 42)
        self.assertEqual(provider.complete.call_count, 4)

    def test_unknown_capability_can_be_reported_without_fake_result(self):
        provider = Mock()
        provider.complete.side_effect = [
            response(call('plan', 'run_plan', steps=[
                {'id': 'audio', 'capability': 'speak', 'params': {'text': 'hello'}}],
                output={'$ref': 'audio'})),
            response(call('stop', 'finish', error='No speech route is configured')),
        ]

        with self.assertRaisesRegex(ProviderError, 'No speech route is configured'):
            ToolAgent(provider).execute(self.request(requires_result=True), self.bridge(inspect.currentframe()))

        messages = provider.complete.call_args.args[0]
        feedback = [json.loads(item['content']) for item in messages if item['role'] == 'tool']
        self.assertIn("Unknown capability 'speak'", feedback[0]['message'])
        self.assertIn('available in this profile', feedback[0]['message'])
        self.assertEqual(provider.complete.call_count, 2)

    def test_provider_failure_is_not_retried(self):
        provider = Mock()
        provider.complete.side_effect = [ProviderError('Provider HTTP 429'), response(call('done', 'finish'))]
        stats = InvocationStats('test.py', 1, 'syntax')
        with self.assertRaises(ProviderError):
            ToolAgent(provider).complete_once([], [], self.request(), stats)
        self.assertEqual(stats.model_calls, 1)
        self.assertEqual(provider.complete.call_count, 1)
        self.assertEqual(stats.retry_backoff_seconds, 0)

    def test_recovery_code_rejected_before_effects_without_replacement(self):
        for action, target in (("complete", None), ("retry", "answer"), ("reraise", "answer")):
            with self.subTest(action=action):
                items = []
                provider = Mock()
                provider.complete.side_effect = [response(
                    call('change', 'execute', code='items.append(1)'),
                    call('bad', 'recover', action=action, code='items.append(2)')),
                    response(call('corrected', 'recover', action='reraise'))]
                request = RecoveryRequest(**vars(self.request()), exception=NameError('bad'), traceback=None,
                    origin=self.request().span, attempt=1, replacement_target=target)
                ToolAgent(provider).recover(request, self.bridge(inspect.currentframe()))
                self.assertEqual(items, [])

    def test_disallowed_retry_batch_is_rejected_before_effects(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [response(
            call('change', 'execute', code='items.append(1)'),
            call('bad', 'recover', action='retry')),
            response(call('corrected', 'recover', action='reraise'))]
        request = RecoveryRequest(**vars(self.request()), exception=NameError('bad'), traceback=None,
                                  origin=self.request().span, attempt=1, retry_allowed=False)
        bridge = self.bridge(inspect.currentframe(), stats=True)
        decision = ToolAgent(provider).recover(request, bridge)
        self.assertEqual(decision.action, 'reraise')
        self.assertEqual(items, [])
        self.assertEqual(bridge.manager.stats.invocations[0].invalid_batches, 1)
        self.assertFalse(json.loads(provider.complete.call_args_list[0].args[0][1]['content'])['retry_allowed'])

    def test_token_usage_records_only_valid_counts(self):
        stats = InvocationStats('test.py', 1, 'syntax')
        token = CURRENT_STATS.set(stats)
        try:
            for _ in range(2):
                record_token_usage({'prompt_tokens': 10, 'completion_tokens': 5,
                    'total_tokens': 15, 'completion_tokens_details': {'reasoning_tokens': 3},
                    'prompt_tokens_details': {'cached_tokens': 2}, 'secret': 'must not record'})
            record_token_usage({'prompt_tokens': True, 'completion_tokens': -1, 'total_tokens': 'secret'})
            record_token_usage(None)
        finally:
            CURRENT_STATS.reset(token)
        self.assertEqual(stats.token_usage, {'prompt_tokens': 20, 'completion_tokens': 10,
            'total_tokens': 30, 'reasoning_tokens': 6, 'cached_tokens': 4, 'uncached_prompt_tokens': 16})

    def test_named_result_does_not_depend_on_provider_call_id(self):
        provider = Mock()
        provider.complete.return_value = response(
            call('provider-generated-872', 'evaluate', code='42', result_id='answer'),
            call('provider-generated-991', 'finish', result_from='answer'))
        result = ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertEqual(result, 42)
        self.assertEqual(provider.complete.call_count, 1)

    def test_blank_optional_result_id_is_omitted(self):
        provider = Mock()
        provider.complete.return_value = response(
            call('value', 'evaluate', code='42', result_id=''),
            call('done', 'finish', result_from='value', code=None, handle=''))

        result = ToolAgent(provider).execute(self.request(requires_result=True), self.bridge(inspect.currentframe()))

        self.assertEqual(result, 42)
        self.assertEqual(provider.complete.call_count, 1)

    def test_result_alias_collisions_reject_whole_batch(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [response(
            call('change', 'execute', code='items.append(1)'),
            call('first', 'evaluate', code='1', result_id='answer'),
            call('second', 'evaluate', code='2', result_id='answer'),
            call('done', 'finish', result_from='answer')),
            response(call('corrected', 'finish'))]
        ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertEqual(items, [])

    def test_invalid_recovery_value_caught_before_side_effect(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [response(
            call('change', 'execute', code='items.append(1)'),
            call('value', 'evaluate', code='42', result_id='answer'),
            call('done', 'recover', action='complete', result_from='answer')),
            response(call('corrected', 'recover', action='complete'))]
        request = RecoveryRequest(**vars(self.request()), exception=NameError('bad'), traceback=None,
                                  origin=self.request().span, attempt=1)
        bridge = self.bridge(inspect.currentframe(), stats=True)
        decision = ToolAgent(provider).recover(request, bridge)
        self.assertEqual(items, [])
        self.assertFalse(decision.has_value)
        self.assertEqual(bridge.manager.stats.invocations[0].invalid_batches, 1)

    def request(self, **changes):
        values = dict(statement="compute result", frame_code="x = 10", related_objects={}, frame_objects={},
                      span=SourceSpan("example.py", 1, 0, 1, 10),
                      profile=ProfileConfig("default", "fake", "model", max_rounds=4), prompts=())
        values.update(changes)
        return AgentRequest(**values)

    def bridge(self, frame, stats=False):
        return RuntimeBridge(frame, Runtime(ResolvedConfig(None, Path.cwd()), stats=stats))

    def test_one_response_value_keeps_identity(self):
        items = []
        provider = Mock()
        provider.complete.return_value = response(call("calculate", "evaluate", code="items"),
                                                  call("done", "finish", result_from="calculate"))
        result = ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertIs(result, items)
        self.assertEqual(provider.complete.call_count, 1)

    def test_one_response_mutation(self):
        items = []
        provider = Mock()
        provider.complete.return_value = response(call("change", "execute", code="items.append(3)"), call("done", "finish"))
        result = ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertIsNone(result)
        self.assertEqual(items, [3])
        self.assertEqual(provider.complete.call_count, 1)

    def test_one_response_recovery(self):
        provider = Mock()
        provider.complete.return_value = response(call("value", "evaluate", code="42"),
                                                  call("done", "recover", action="complete", result_from="value"))
        request = RecoveryRequest(**vars(self.request()), exception=ValueError("bad"), traceback=None,
                                  origin=self.request().span, attempt=1, replacement_target="answer")
        result = ToolAgent(provider).recover(request, self.bridge(inspect.currentframe()))
        self.assertEqual((result.action, result.has_value, result.value), ("complete", True, 42))
        self.assertEqual(provider.complete.call_count, 1)

    def test_alias_metadata_and_get_binding_result(self):
        items = []
        request = self.request(related_objects={"items": items}, frame_objects={"alias": items})
        payloads = []
        def complete(messages, tools):
            payloads.append(json.loads(messages[2]["content"]))
            return response(call("get", "get_binding", name="items"), call("done", "finish", result_from="get"))
        provider = Mock(complete=Mock(side_effect=complete))
        self.assertIs(ToolAgent(provider).execute(request, self.bridge(inspect.currentframe())), items)
        original = payloads[0]["related_objects"]["items"]
        self.assertEqual(original['type'], 'list')
        self.assertEqual(original['size'], 0)
        self.assertEqual(original['sample_item_types'], [])
        self.assertEqual(payloads[0]["frame_objects"]["alias"], original)

    def test_invalid_batches_do_nothing(self):
        invalid = [
            [call("end", "finish"), call("later", "inspect", handle="bad")],
            [call("end", "finish"), call("other_end", "finish")],
            [call("unknown", "unknown")],
            [call("bad_args", "evaluate", code=42)],
            [call("bad_ref", "finish", result_from="future")],
            [call("bad_ref", "finish", result_from="change")],
            [call("both", "finish", handle="object-1", result_from="change")],
            [call("wrong_mode", "recover", action="complete")],
        ]
        for suffix in invalid:
            with self.subTest(suffix=suffix):
                items = []
                provider = Mock()
                provider.complete.side_effect = [response(call("change", "execute", code="items.append(1)"), *suffix),
                                                  response(call("corrected", "finish"))]
                ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
                self.assertEqual(items, [])
                messages = provider.complete.call_args.args[0]
                replies = [m for m in messages if m["role"] == "tool"]
                self.assertEqual(len(replies), 1 + len(suffix))

    def test_tagged_outcome_conflict_rejects_whole_batch(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [response(
            call('change', 'execute', code='items.append(1)'),
            call('invalid', 'finish', outcome={'kind': 'expression', 'code': 'items', 'value': None})),
            response(call('fixed', 'finish', outcome={'kind': 'literal', 'value': 0}))]
        self.assertEqual(ToolAgent(provider).execute(self.request(requires_result=True),
                                                      self.bridge(inspect.currentframe())), 0)
        self.assertEqual(items, [])

    def test_video_job_has_separate_deadline(self):
        provider = Mock()
        provider.complete.return_value = response(
            call('video', 'run_plan', steps=[{'id': 'job', 'capability': 'video',
                'params': {'mode': 'generate', 'prompt': 'test'}}],
                output={'$ref': 'job'}, result_id='result'),
            call('done', 'finish', outcome={'kind': 'reference', 'id': 'result'}))
        bridge = self.bridge(inspect.currentframe())
        def slow_video(*args, **kwargs):
            time.sleep(0.3)
            return 'video completed'
        bridge.manager.capabilities.run_plan = Mock(side_effect=slow_video)
        profile = ProfileConfig('default', 'fake', 'model', timeout=0.2)
        request = self.request(profile=profile, requires_result=True)
        self.assertEqual(ToolAgent(provider).execute(request, bridge), 'video completed')
        self.assertEqual(provider.complete.call_count, 1)

    def test_duplicate_ids_reject_batch_before_mutation(self):
        items = []
        provider = Mock()
        provider.complete.return_value = response(call("same", "execute", code="items.append(1)"), call("same", "finish"))
        with self.assertRaises(ProviderError):
            ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertEqual(items, [])

    def test_failure_skips_rest_without_replay(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [
            response(call("first", "execute", code="items.append(1)"), call("fail", "evaluate", code="1/0"),
                     call("later", "execute", code="items.append(2)"), call("done", "finish")),
            response(call("corrected", "finish")),
        ]
        ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertEqual(items, [1])
        replies = [json.loads(m["content"]) for m in provider.complete.call_args.args[0] if m["role"] == "tool"]
        self.assertEqual([r.get("status") for r in replies], [None, "error", "skipped", "skipped"])

    def test_reraise_handle_rejected_without_dereference(self):
        items = []
        bridge = self.bridge(inspect.currentframe())
        bridge.dereference = Mock(side_effect=AssertionError("must not dereference"))
        provider = Mock()
        provider.complete.side_effect = [
            response(call("change", "execute", code="items.append(1)"), call("bad", "recover", action="reraise", handle="bad_handle")),
            response(call("corrected", "recover", action="reraise")),
        ]
        request = RecoveryRequest(**vars(self.request()), exception=ValueError("bad"), traceback=None,
                                  origin=self.request().span, attempt=1)
        result = ToolAgent(provider).recover(request, bridge)
        self.assertEqual(result.action, "reraise")
        self.assertEqual(items, [])
        bridge.dereference.assert_not_called()

    def test_schema_primitive_and_nested_types(self):
        schema = {"type": "object", "properties": {
            "depth": {"type": "integer", "minimum": 0}, "timeout": {"type": "number"},
            "enabled": {"type": "boolean"}, "items": {"type": "array", "items": {"type": "null"}}},
            "required": ["depth"], "additionalProperties": False}
        validate_schema({"depth": 2, "timeout": 0.5, "enabled": True, "items": [None]}, schema)
        for value in ({"depth": True}, {"depth": -1}, {"depth": 1, "timeout": float("nan")},
                      {"depth": 1, "items": ["not null"]}, {"depth": 1, "extra": 5}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_schema(value, schema)

    def test_stats_and_sdk_request_bytes(self):
        key = "test-private-key"
        code = "private-object-value"
        request = self.request(frame_objects={"value": code})
        request.profile.routes['reasoning'] = [{'provider': 'route', 'model': 'openai/test'}]
        config = ResolvedConfig(None, Path.cwd(), providers={'route': {'model': 'openai/test', 'api_key_env': 'TEST_AI_KEY'}})
        provider = LiteLLMProvider(config, request.profile)
        sdk = Mock(completion=Mock(return_value={"choices": [{"message": response(call("done", "finish"))}]}))
        bridge = self.bridge(inspect.currentframe(), stats=True)
        progress = io.StringIO()
        with contextlib.redirect_stderr(progress), patch.dict('os.environ', {'TEST_AI_KEY': key}), \
             patch('aiython.providers.sdk', return_value=sdk):
            ToolAgent(provider).execute(request, bridge)
        record = bridge.manager.stats.invocations[0]
        self.assertEqual((record.model_calls, record.tools), (1, 1))
        self.assertEqual(record.provider_requests, 1)
        self.assertIn('response received', progress.getvalue())
        self.assertNotIn(key, progress.getvalue())
        self.assertNotIn(code, progress.getvalue())
        body = sdk.completion.call_args.kwargs
        self.assertGreater(record.request_bytes, 0)
        self.assertEqual(body['max_retries'], 0)
        self.assertEqual(record.context_bytes, {
            'system': len(body['messages'][0]['content'].encode()),
            'stable': len(body['messages'][1]['content'].encode()),
            'live': len(body['messages'][2]['content'].encode()),
            'tools': len(canonical(body['tools']).encode()),
        })
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            bridge.manager.stats.report()
        self.assertNotIn(key, output.getvalue())
        self.assertNotIn(code, output.getvalue())
        self.assertGreaterEqual(record.provider_seconds, 0)
        self.assertGreaterEqual(record.context_build_seconds, 0)
        self.assertEqual(set(record.context_bytes), {"system", "stable", "live", "tools"})
        self.assertTrue(all(size > 0 for size in record.context_bytes.values()))

    def test_model_calls_reduced_for_same_value(self):
        legacy = Mock()
        legacy.complete.side_effect = [response(call("value", "evaluate", code="42")), response(call("done", "finish", handle="object-1"))]
        optimized = Mock()
        optimized.complete.return_value = response(call("value", "evaluate", code="42"), call("done", "finish", result_from="value"))
        for provider in (legacy, optimized):
            self.assertEqual(ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe())), 42)
        self.assertEqual((legacy.complete.call_count, optimized.complete.call_count), (2, 1))

    def test_stats_remain_available_after_provider_failure(self):
        provider = Mock()
        provider.complete.side_effect = ProviderError("connection failed")
        bridge = self.bridge(inspect.currentframe(), stats=True)
        with self.assertRaises(ProviderError):
            ToolAgent(provider).execute(self.request(), bridge)
        self.assertEqual(bridge.manager.stats.invocations[0].model_calls, 1)
        self.assertEqual(bridge.manager.stats.invocations[0].tools, 0)
        self.assertIsNone(CURRENT_STATS.get())

    def test_lazy_code_tool_protocol(self):
        bridge = self.bridge(inspect.currentframe())
        provider = Mock()
        provider.complete.side_effect = [response(call("source", "get_frame_code")), response(call("done", "finish"))]
        ToolAgent(provider).execute(self.request(), bridge)
        replies = [m for m in provider.complete.call_args.args[0] if m["role"] == "tool"]
        self.assertEqual(len(replies), 1)
        self.assertFalse(json.loads(replies[0]["content"])["available"])
