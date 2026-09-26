import asyncio
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiython.agent import (
    BatchError, RunState, TerminalResult, ToolAgent, ValidatedCall, _PlanBridge,
    normalize_optional_arguments, reject_constant, validate_batch, validate_outcome, validate_schema,
)
from aiython.capabilities import CapabilityPermissionError, InvocationError
from aiython.models import AgentRequest, ProfileConfig, ProviderError, RecoveryRequest, ResolvedConfig, SourceSpan
from aiython.providers import INVOCATION_DEADLINE
from aiython.runtime import Runtime, RuntimeBridge
from aiython.stats import InvocationStats


def call(identifier, name, arguments=None):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments or {})}}


class AgentValidationEdgeTests(unittest.TestCase):
    def test_schema_and_legacy_argument_validation(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite JSON constant'):
            reject_constant('NaN')
        with self.assertRaisesRegex(ValueError, 'expected object'):
            validate_schema([], {'type': 'object'})
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            validate_schema({'nested': [{'value': float('nan')}]}, {'type': 'object'})
        self.assertEqual(normalize_optional_arguments('finish', {'code': '  ', 'value': 3}),
                         {'outcome': {'kind': 'literal', 'value': 3}})
        with self.assertRaisesRegex(ValueError, 'only one result field'):
            normalize_optional_arguments('finish', {'value': 1, 'code': '2'})
        self.assertEqual(normalize_optional_arguments('finish', {'error': 'missing',
                         'missing_capability': 'vision'})['outcome']['missing_capability'], 'vision')
        self.assertEqual(normalize_optional_arguments('finish', {'result_from': 'earlier'})['outcome'],
                         {'kind': 'reference', 'id': 'earlier'})
        self.assertEqual(normalize_optional_arguments('finish', {'handle': 'object-1'})['outcome'],
                         {'kind': 'handle', 'id': 'object-1'})
        self.assertEqual(normalize_optional_arguments('finish', {})['outcome'], {'kind': 'none'})
        self.assertEqual(normalize_optional_arguments('execute', {'result_id': ' '}), {})
        self.assertEqual(normalize_optional_arguments('finish', {'outcome': {'kind': 'error',
                         'reason': 'no', 'missing_capability': ' '}}),
                         {'outcome': {'kind': 'error', 'reason': 'no'}})
        self.assertEqual(normalize_optional_arguments('finish', None), None)

    def test_outcome_and_batch_failures_are_rejected_before_effects(self):
        bad_outcomes = [
            ({'kind': 'literal'}, 'missing'),
            ({'kind': 'literal', 'value': 1, 'id': 'x'}, 'conflicting'),
            ({'kind': 'expression', 'code': ' '}, 'nonempty'),
            ({'kind': 'none'}, 'requires a result'),
            ({'kind': 'reference', 'id': 'missing'}, 'earlier'),
        ]
        for outcome, message in bad_outcomes:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(ValueError, message):
                validate_outcome(outcome, {}, requires_result=outcome['kind'] == 'none')
        invalid = [
            ([{'id': '', 'type': 'function'}], ProviderError, 'nonempty'),
            ([{'id': 'x', 'type': 'not-function', 'function': {}}], BatchError, 'function'),
            ([{'id': 'x', 'type': 'function', 'function': {'name': 'finish', 'arguments': 4}}], BatchError, 'JSON string'),
            ([call('x', 'recover', {'action': 'complete', 'outcome': {'kind': 'error', 'reason': 'bad'}})], BatchError, 'reraise'),
        ]
        for calls, error_type, message in invalid:
            with self.subTest(calls=calls), self.assertRaisesRegex(error_type, message):
                validate_batch(calls, {'finish', 'recover'}, set())
        with self.assertRaisesRegex(ProviderError, 'list'):
            validate_batch(None, {'finish'}, set())


class ToolAgentEdgeTests(unittest.TestCase):
    def setUp(self):
        self.agent = ToolAgent(None)
        self.profile = ProfileConfig('test', 'fake', 'model', max_rounds=2)
        self.runtime = Runtime(ResolvedConfig(None, Path.cwd()))
        self.bridge = RuntimeBridge(inspect.currentframe(), self.runtime)
        self.request = AgentRequest('compute', 'compute', {}, {},
                                    SourceSpan('test.py', 1, 0, 1, 7), self.profile, ())
        self.bridge.ai_request = self.request

    def test_plan_bridge_and_group_tools(self):
        value = object()
        handle = self.bridge.handle(value)['handle']
        plan = _PlanBridge(self.bridge)
        self.assertIs(plan.dereference(handle), value)
        with self.assertRaisesRegex(ValueError, 'Unknown object handle'):
            plan.dereference('missing')
        with self.assertRaisesRegex(ValueError, 'not attached'):
            self.agent._dispatch('list_peers', {}, self.bridge, {})
        participant = SimpleNamespace(name='worker', peers=Mock(return_value=['main']),
                                      send=Mock(return_value={'id': 'message'}),
                                      read=Mock(return_value=[{'payload': 3}]))
        self.bridge.participant = participant
        self.assertEqual(self.agent._dispatch('list_peers', {}, self.bridge, {}), ['main'])
        self.assertEqual(self.agent._dispatch('send_message', {'recipient': 'main', 'payload': 3},
                                              self.bridge, {})['id'], 'message')
        self.assertEqual(self.agent._dispatch('read_messages', {}, self.bridge, {})['messages'],
                         [{'payload': 3}])

    def test_dispatch_remaining_tools_and_error_hint(self):
        caps = self.runtime.capabilities
        with patch.object(caps.store, 'jobs', return_value=['job']), \
                patch.object(caps, 'resume_job', return_value='done'):
            self.assertEqual(self.agent._dispatch('jobs', {}, self.bridge, {}), ['job'])
            self.assertEqual(self.bridge.dereference(self.agent._dispatch('resume_job',
                             {'operation': 'job'}, self.bridge, {})['handle']), 'done')
        self.assertEqual(self.agent._dispatch('get_binding', {'name': 'self'}, self.bridge, {})['type'],
                         'ToolAgentEdgeTests')
        handle = self.bridge.handle(12)['handle']
        self.assertTrue(self.agent._dispatch('set_binding', {'name': 'value', 'handle': handle},
                                             self.bridge, {})['ok'])
        self.assertEqual(self.bridge.get('value'), 12)
        self.assertEqual(self.agent._dispatch('inspect', {'handle': handle}, self.bridge, {})['value'], 12)
        self.assertIn('available', self.agent._dispatch('get_frame_code', {}, self.bridge, {}))
        self.assertEqual(self.agent._dispatch('frames', {}, self.bridge, {}), [])
        with self.assertRaisesRegex(ValueError, 'Unknown tool'):
            self.agent._dispatch('unknown', {}, self.bridge, {})

        with patch.object(caps, 'suggest_missing_route', return_value='aiython.toml'):
            result = self.agent._dispatch('finish', {'outcome': {'kind': 'error',
                'reason': 'vision route unavailable', 'missing_capability': 'vision'}}, self.bridge, {})
        self.assertIsInstance(result, TerminalResult)
        self.assertIn('aiython.toml', result.error)
        with patch.object(caps, 'suggest_missing_route', return_value='aiython.toml'):
            inferred = self.agent._dispatch('finish', {'outcome': {'kind': 'error',
                'reason': 'vision route unavailable'}}, self.bridge, {})
        self.assertIn('aiython.toml', inferred.error)

    def test_response_protocol_and_round_limits(self):
        messages = []
        state = RunState(set())
        with self.assertRaisesRegex(ProviderError, 'assistant message'):
            self.agent._prepare_response({'role': 'user'}, self.request, messages,
                                         state, recovery=False, stats=None)
        for _ in range(2):
            if state.empty_responses:
                with self.assertRaisesRegex(ProviderError, 'repeated empty'):
                    self.agent._prepare_response({'role': 'assistant'}, self.request, messages,
                                                 state, recovery=False, stats=None)
            else:
                self.assertIsNone(self.agent._prepare_response({'role': 'assistant'},
                                   self.request, messages, state, recovery=False, stats=None))
        provider = Mock(complete=Mock(return_value={'role': 'assistant', 'tool_calls': []}))
        self.agent.provider = provider
        with self.assertRaisesRegex(ProviderError, 'repeated empty'):
            self.agent.execute(self.request, self.bridge)

    def test_provider_errors_and_timing_are_recorded(self):
        stats = InvocationStats('test.py', 1, 'syntax')
        self.agent.provider = SimpleNamespace(complete=Mock(side_effect=RuntimeError('secret')))
        with self.assertRaisesRegex(ProviderError, 'provider failed') as caught:
            self.agent.complete_once([], [], self.request, stats)
        self.assertNotIn('secret', str(caught.exception))
        self.assertEqual(stats.model_calls, 1)
        self.assertEqual(len(stats.model_call_seconds), 1)

        async def scenario():
            self.agent.provider = SimpleNamespace(complete=Mock(return_value={'role': 'assistant'}))
            self.assertEqual((await self.agent.complete_once_async([], [], self.request, stats))['role'],
                             'assistant')
            self.agent.provider = SimpleNamespace(acomplete=AsyncMock(side_effect=RuntimeError('secret')))
            with self.assertRaisesRegex(ProviderError, 'provider failed'):
                await self.agent.complete_once_async([], [], self.request, stats)
            self.agent.provider = SimpleNamespace(acomplete=AsyncMock(side_effect=ProviderError('api')))
            with self.assertRaisesRegex(ProviderError, 'api'):
                await self.agent.complete_once_async([], [], self.request, stats)
        asyncio.run(scenario())
        self.assertEqual(stats.model_calls, 4)
        self.assertEqual(len(stats.model_call_seconds), 4)

    def test_failure_diagnostics_keep_submission_state(self):
        messages = []
        stats = InvocationStats('test.py', 1, 'syntax')
        self.bridge.completed_steps = {'one': ('fingerprint', 42)}
        failure = self.agent._record_tool_failure(
            ValidatedCall('plan', 'run_plan', {}),
            InvocationError('provider uncertain', accepted=True), self.bridge, messages, stats)
        detail = json.loads(messages[-1]['content'])
        self.assertEqual(failure.name, 'run_plan')
        self.assertTrue(detail['acceptance_unknown_or_submitted'])
        self.assertEqual(detail['completed_steps']['one']['value'], 42)
        self.assertEqual(stats.tool_failures, 1)
        self.assertEqual(stats.tool_errors[0]['error'], 'OtherError')

    def test_empty_recovery_guidance_and_round_exhaustion(self):
        stats = InvocationStats('test.py', 1, 'recovery')
        for replacement in ('answer', None):
            recovery = RecoveryRequest(**vars(self.request), exception=ValueError('bad'),
                traceback=None, origin=self.request.span, attempt=1,
                replacement_target=replacement)
            messages = []
            self.assertIsNone(self.agent._prepare_response({'role': 'assistant'}, recovery,
                messages, RunState(set()), recovery=True, stats=stats))
            self.assertIn('recover', messages[-1]['content'])
        self.assertEqual(stats.empty_responses, 2)
        required = AgentRequest(**{**vars(self.request), 'requires_result': True})
        messages = []
        self.agent._prepare_response({'role': 'assistant'}, required, messages,
                                     RunState(set()), recovery=False, stats=None)
        self.assertNotIn('side effects only', messages[-1]['content'])

        one_round = ProfileConfig('test', 'fake', 'model', max_rounds=1)
        request = AgentRequest(**{**vars(self.request), 'profile': one_round})
        self.agent.provider = SimpleNamespace(complete=Mock(return_value={'role': 'assistant'}))
        with self.assertRaisesRegex(ProviderError, 'exceeded 1 agent rounds'):
            self.agent.execute(request, self.bridge)
        async def async_round():
            self.runtime.stats.enabled = True
            self.agent.provider = SimpleNamespace(acomplete=AsyncMock(return_value={'role': 'assistant'}))
            with self.assertRaisesRegex(ProviderError, 'exceeded 1 agent rounds'):
                await self.agent.aexecute(request, self.bridge)
            recovery = RecoveryRequest(**vars(request), exception=ValueError('bad'),
                traceback=None, origin=request.span, attempt=1)
            self.agent.provider = SimpleNamespace(acomplete=AsyncMock(return_value={
                'role': 'assistant', 'tool_calls': [call('done', 'recover', {'action': 'reraise'})]}))
            self.assertEqual((await self.agent.arecover(recovery, self.bridge)).action, 'reraise')
        asyncio.run(async_round())

    def test_async_batch_records_failure_skips_and_video_deadline(self):
        async def scenario():
            stats = InvocationStats('test.py', 1, 'syntax')
            messages = []
            calls = [call('first', 'evaluate', {'code': '1 / 0'}),
                     call('second', 'finish', {'outcome': {'kind': 'none'}})]
            result = await self.agent.process_batch_async(calls, {'evaluate', 'finish'}, set(),
                self.request, self.bridge, messages, recovery=False, stats=stats)
            self.assertEqual(result.name, 'evaluate')
            self.assertEqual(json.loads(messages[-1]['content'])['status'], 'skipped')
            self.assertEqual(stats.tools, 1)
            self.assertEqual(stats.tool_failures, 1)

            with patch.object(self.agent, 'dispatch_async', new_callable=AsyncMock,
                              side_effect=CapabilityPermissionError('denied')):
                with self.assertRaisesRegex(CapabilityPermissionError, 'denied'):
                    await self.agent.process_batch_async([call('read', 'jobs')], {'jobs'}, set(),
                        self.request, self.bridge, [], recovery=False, stats=stats)
            with patch.object(self.agent, 'dispatch_async', new_callable=AsyncMock,
                              return_value={'done': True}):
                token = INVOCATION_DEADLINE.set(time.monotonic() + 1)
                try:
                    await self.agent.process_batch_async([call('video', 'resume_job',
                        {'operation': 'job'})], {'resume_job'}, set(), self.request,
                        self.bridge, [], recovery=False, stats=stats)
                finally:
                    INVOCATION_DEADLINE.reset(token)
            self.assertGreater(stats.runtime_seconds, 0)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
