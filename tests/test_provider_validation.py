"""Provider response and route failure behavior without network calls."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiython.models import ConfigError, ProfileConfig, ProviderError, ResolvedConfig
from aiython.providers import LiteLLMProvider, parse_completion


def provider_with_routes():
    profile = ProfileConfig('test', 'litellm', 'openai/first')
    profile.routes['reasoning'] = [
        {'provider': 'first', 'model': 'openai/first'},
        {'provider': 'second', 'model': 'openai/second'},
    ]
    config = ResolvedConfig(None, Path.cwd(), providers={
        'first': {'model': 'openai/first'},
        'second': {'model': 'openai/second'},
    })
    return LiteLLMProvider(config, profile)


class ProviderValidationTests(unittest.TestCase):
    def test_invalid_response_shapes_are_rejected_without_payload_leaks(self):
        profile = ProfileConfig('test', 'litellm', 'openai/test')
        cases = [
            (SimpleNamespace(model_dump=lambda **_: 'private'), 'root must be an object'),
            ({'choices': [None]}, 'no completion choices'),
            ({'choices': [{'error': 'private', 'message': {'content': 'private'}}]},
             'did not complete'),
            ({'choices': [{'finish_reason': 'content_filter', 'message': {}}]},
             'did not complete'),
        ]
        for response, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ProviderError, message) as caught:
                parse_completion(response, profile)
            self.assertNotIn('private', str(caught.exception))

    def test_sync_rate_limit_moves_to_next_route_once(self):
        provider = provider_with_routes()
        rate_limit = RuntimeError('secret')
        rate_limit.status_code = 429
        sdk = SimpleNamespace(completion=Mock(side_effect=[
            rate_limit, {'choices': [{'message': {'content': 'done'}}]},
        ]))
        with patch('aiython.providers.sdk', return_value=sdk):
            result = provider.complete([], [])
        self.assertEqual(result['content'], 'done')
        self.assertEqual(sdk.completion.call_count, 2)

    def test_missing_route_and_forbidden_access(self):
        provider = provider_with_routes()
        provider.profile.routes.clear()
        with self.assertRaisesRegex(ConfigError, 'Reasoning model'):
            provider.complete([], [])
        provider = provider_with_routes()
        denied = RuntimeError('secret')
        denied.status_code = 403
        with patch('aiython.providers.sdk', return_value=SimpleNamespace(
                completion=Mock(side_effect=denied))):
            with self.assertRaisesRegex(ProviderError, 'access denied') as caught:
                provider.complete([], [])
        self.assertNotIn('secret', str(caught.exception))


class AsyncProviderValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_rate_limit_moves_to_next_route_once(self):
        provider = provider_with_routes()
        rate_limit = RuntimeError('secret')
        rate_limit.status_code = 429
        sdk = SimpleNamespace(acompletion=AsyncMock(side_effect=[
            rate_limit, {'choices': [{'message': {'content': 'done'}}]},
        ]))
        with patch('aiython.providers.sdk', return_value=sdk):
            result = await provider.acomplete([], [])
        self.assertEqual(result['content'], 'done')
        self.assertEqual(sdk.acompletion.await_count, 2)

    async def test_async_missing_route_and_missing_model(self):
        provider = provider_with_routes()
        provider.profile.routes.clear()
        with self.assertRaisesRegex(ConfigError, 'Reasoning model'):
            await provider.acomplete([], [])
        provider = provider_with_routes()
        missing = RuntimeError('secret')
        missing.status_code = 404
        with patch('aiython.providers.sdk', return_value=SimpleNamespace(
                acompletion=AsyncMock(side_effect=missing))):
            with self.assertRaisesRegex(ProviderError, 'unavailable') as caught:
                await provider.acomplete([], [])
        self.assertNotIn('secret', str(caught.exception))
