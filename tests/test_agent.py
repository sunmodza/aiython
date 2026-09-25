import inspect
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from aiython.agent import ToolAgent
from aiython.models import AgentRequest, ConfigError, ProfileConfig, ProviderError, ResolvedConfig, SourceSpan
from aiython.providers import INVOCATION_DEADLINE, LiteLLMProvider, load_provider, parse_completion
from aiython.runtime import Runtime, RuntimeBridge
from aiython.stats import CURRENT_STATS, InvocationStats


def tool(identifier, name, **args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


class AgentTests(unittest.TestCase):
    def request(self):
        return AgentRequest('compute result', 'items = []', {}, {}, SourceSpan('test.py', 1, 0, 1, 8),
                            ProfileConfig('test', 'litellm', 'openai/test', max_rounds=4), ())

    def bridge(self, frame):
        return RuntimeBridge(frame, Runtime(ResolvedConfig(None, Path.cwd())))

    def test_completion_validation_redacts_response_and_keeps_usage(self):
        cases = [
            ({'error': {'message': 'private'}}, 'error envelope'),
            ({'choices': []}, 'no completion choices'),
            ({'choices': [{'message': None}]}, 'no usable assistant message'),
            ({'choices': [{'finish_reason': 'length', 'message': tool('x', 'execute', code='1')}]}, 'did not complete'),
            ({'choices': [{'message': {'refusal': 'private'}}]}, 'no usable assistant message'),
            ({'choices': [{'message': {'tool_calls': 'private'}}]}, 'must be an array'),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                stats = InvocationStats('test.py', 1, 'syntax')
                token = CURRENT_STATS.set(stats)
                try:
                    result['usage'] = {'prompt_tokens': 100, 'completion_tokens': 20}
                    with self.assertRaisesRegex(ProviderError, expected) as caught:
                        parse_completion(result, self.request().profile)
                    self.assertEqual(stats.token_usage['completion_tokens'], 20)
                    self.assertNotIn('private', str(caught.exception))
                finally:
                    CURRENT_STATS.reset(token)

    def test_truncated_batch_does_not_execute_side_effect(self):
        items = []
        config = ResolvedConfig(None, Path.cwd(), providers={'test:reasoning:0': {'model': 'openai/test'}})
        profile = self.request().profile
        profile.routes['reasoning'] = [{'provider': 'test:reasoning:0', 'model': 'openai/test'}]
        response = {'choices': [{'finish_reason': 'length', 'message': tool('change', 'execute', code='items.append(1)')}]}
        fake_sdk = Mock(completion=Mock(return_value=response))
        with patch('aiython.providers.sdk', return_value=fake_sdk):
            with self.assertRaisesRegex(ProviderError, 'did not complete'):
                ToolAgent(LiteLLMProvider(config, profile)).execute(self.request_with_profile(profile), self.bridge(inspect.currentframe()))
        self.assertEqual(items, [])

    def request_with_profile(self, profile):
        request = self.request()
        request.profile = profile
        return request

    def test_litellm_completion_contract_and_no_hidden_retry(self):
        profile = self.request().profile
        profile.routes['reasoning'] = [{'provider': 'route', 'model': 'openai/test'}]
        config = ResolvedConfig(None, Path.cwd(), providers={'route': {'model': 'openai/test', 'api_key_env': 'TEST_AI_KEY', 'api_base': 'https://example.test/v1'}})
        provider = load_provider(config, profile)
        self.assertIsInstance(provider, LiteLLMProvider)
        fake_sdk = Mock(completion=Mock(return_value={'choices': [{'message': tool('done', 'finish', outcome={'kind': 'literal', 'value': 7})}]}))
        with patch.dict('os.environ', {'TEST_AI_KEY': 'private-key'}), patch('aiython.providers.sdk', return_value=fake_sdk):
            self.assertEqual(ToolAgent(provider).execute(self.request_with_profile(profile), self.bridge(inspect.currentframe())), 7)
        kwargs = fake_sdk.completion.call_args.kwargs
        self.assertEqual(kwargs['model'], 'openai/test')
        self.assertEqual(kwargs['api_key'], 'private-key')
        self.assertEqual(kwargs['api_base'], 'https://example.test/v1')
        self.assertEqual(kwargs['max_retries'], 0)
        self.assertLessEqual(kwargs['timeout'], 120)
        failure = Mock(completion=Mock(side_effect=TimeoutError('private-key')))
        with patch.dict('os.environ', {'TEST_AI_KEY': 'private-key'}), patch('aiython.providers.sdk', return_value=failure):
            with self.assertRaises(ProviderError) as caught:
                provider.complete([], [])
        self.assertEqual(failure.completion.call_count, 1)
        self.assertNotIn('private-key', str(caught.exception))

        unauthorized = Exception('private-key')
        unauthorized.status_code = 401
        failure.completion.side_effect = unauthorized
        with patch.dict('os.environ', {'TEST_AI_KEY': 'private-key'}), patch('aiython.providers.sdk', return_value=failure):
            with self.assertRaisesRegex(ProviderError, 'authentication failed; check the configured API key'):
                provider.complete([], [])

    def test_missing_credential_is_not_masked_as_provider_failure(self):
        profile = self.request().profile
        profile.routes['reasoning'] = [{'provider': 'route', 'model': 'openrouter/test'}]
        config = ResolvedConfig(None, Path.cwd(), providers={
            'route': {'model': 'openrouter/test', 'api_key_env': 'AIYTHON_TEST_MISSING_KEY'}})
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ConfigError, 'Missing credential AIYTHON_TEST_MISSING_KEY'):
                ToolAgent(LiteLLMProvider(config, profile)).execute(
                    self.request_with_profile(profile), self.bridge(inspect.currentframe()))

    def test_tagged_terminal_and_live_object_identity(self):
        items = []
        provider = Mock()
        provider.complete.side_effect = [
            tool('change', 'execute', code='items.append(3)'),
            tool('return', 'finish', outcome={'kind': 'expression', 'code': 'items'}),
        ]
        result = ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertIs(result, items)
        self.assertEqual(items, [3])

    def test_remaining_deadline_reaches_sdk_and_expiry_prevents_call(self):
        profile = self.request().profile
        profile.routes['reasoning'] = [{'provider': 'route', 'model': 'openai/test'}]
        config = ResolvedConfig(None, Path.cwd(), providers={'route': {'model': 'openai/test'}})
        provider = LiteLLMProvider(config, profile)
        sdk = Mock(completion=Mock(return_value={'choices': [{'message': tool('done', 'finish', outcome={'kind': 'none'})}]}))
        with patch('aiython.providers.sdk', return_value=sdk):
            token = INVOCATION_DEADLINE.set(time.monotonic() + 0.1)
            try:
                provider.complete([], [])
            finally:
                INVOCATION_DEADLINE.reset(token)
            self.assertLessEqual(sdk.completion.call_args.kwargs['timeout'], 0.1)
            sdk.completion.reset_mock()
            token = INVOCATION_DEADLINE.set(time.monotonic() - 1)
            try:
                with self.assertRaisesRegex(ProviderError, 'deadline'):
                    provider.complete([], [])
            finally:
                INVOCATION_DEADLINE.reset(token)
            sdk.completion.assert_not_called()

    def test_invalid_terminal_is_bounded(self):
        provider = Mock()
        provider.complete.side_effect = [
            tool('one', 'finish', outcome={'kind': 'expression', 'code': '42', 'value': 42}),
            tool('two', 'finish', outcome={'kind': 'expression', 'code': '42', 'value': 42}),
        ]
        with self.assertRaisesRegex(ProviderError, 'repeated invalid tool batches'):
            ToolAgent(provider).execute(self.request(), self.bridge(inspect.currentframe()))
        self.assertEqual(provider.complete.call_count, 2)
