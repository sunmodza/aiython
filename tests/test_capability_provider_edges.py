"""Provider boundary failures and media safety checks."""

import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from aiython.assets import Audio, Document, Image
from aiython.capabilities import CapabilityError, CapabilityRequest, CapabilityRuntime, InvocationError
from aiython.capability_providers import (
    LiteLLMAdapter, _openrouter_video_policy_error, _public_https, download_image,
    make_adapter, plain, public_context, text_of,
)
from aiython.models import ConfigError, ProfileConfig, ResolvedConfig


class CapabilityProviderEdgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = ProfileConfig('default', 'litellm', 'gemini/test')
        self.config = ResolvedConfig(None, self.root, 'default', {'default': self.profile},
                                     providers={'route': {'model': 'gemini/test'}})
        self.context = CapabilityRuntime(self.config)
        self.adapter = LiteLLMAdapter(self.config, self.profile, 'route')
        self.sdk = Mock()
        sdk_patch = patch('aiython.capability_providers.sdk', return_value=self.sdk)
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)

    def asset(self, name, data=b'data', kind=Document):
        path = self.root / name
        path.write_bytes(data)
        return kind(path)

    def invoke(self, capability, params, model='gemini/test'):
        request = CapabilityRequest(capability, model, params, provider='route')
        return self.adapter.invoke(request, self.context).value

    def test_text_and_public_context_reject_unserializable_values(self):
        self.assertEqual(text_of({'text': 'public', 'object': object()}), 'public')
        self.assertEqual(text_of('public'), 'public')
        with self.assertRaises(CapabilityError):
            text_of({'text': 42})
        self.assertEqual(public_context({'nested': [{'text': 'public', 'object': object()}]}),
                         {'nested': [{'text': 'public'}]})
        with self.assertRaises(CapabilityError):
            public_context({'bad': object()})
        self.assertEqual(plain(SimpleNamespace(model_dump=lambda **_: {'value': 1})), {'value': 1})
        with self.assertRaisesRegex(ConfigError, 'Unknown LiteLLM route'):
            LiteLLMAdapter(self.config, self.profile, 'missing')
        self.assertIn('embedding', self.adapter.capabilities())
        self.assertIsInstance(make_adapter(self.config, self.profile, 'route'), LiteLLMAdapter)

    def test_public_https_requires_global_address_and_no_credentials(self):
        self.assertFalse(_public_https('http://example.test/image.png'))
        self.assertFalse(_public_https('https://user:secret@example.test/image.png'))
        with patch('aiython.capability_providers.socket.getaddrinfo', return_value=[]):
            self.assertFalse(_public_https('https://example.test/image.png'))
        address = lambda host: [(socket.AF_INET, socket.SOCK_STREAM, 0, '', (host, 443))]
        with patch('aiython.capability_providers.socket.getaddrinfo', return_value=address('127.0.0.1')):
            self.assertFalse(_public_https('https://example.test/image.png'))
        with patch('aiython.capability_providers.socket.getaddrinfo', return_value=address('8.8.8.8')):
            self.assertTrue(_public_https('https://example.test/image.png'))
        with patch('aiython.capability_providers.socket.getaddrinfo', side_effect=OSError('private')):
            self.assertFalse(_public_https('https://example.test/image.png'))

    def test_image_download_limits_and_errors(self):
        url = 'https://example.test/image.png'
        with self.assertRaisesRegex(InvocationError, 'Untrusted'):
            download_image('http://localhost/private.png')

        class Response:
            def __init__(self, chunks, error=None):
                self.chunks, self.error = chunks, error

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def raise_for_status(self):
                if self.error:
                    raise self.error

            def iter_bytes(self):
                return iter(self.chunks)

        class Client:
            def __init__(self, response):
                self.response = response

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def stream(self, method, requested):
                self_outer.assertEqual((method, requested), ('GET', url))
                return self.response

        self_outer = self
        with patch('aiython.capability_providers._public_https', return_value=True), \
                patch('aiython.capability_providers.httpx.Client', return_value=Client(Response([b'a', b'b']))):
            self.assertEqual(download_image(url), b'ab')

        class OversizedChunk:
            def __len__(self):
                return 32 * 1024 * 1024 + 1

        with patch('aiython.capability_providers._public_https', return_value=True), \
                patch('aiython.capability_providers.httpx.Client', return_value=Client(Response([OversizedChunk()]))):
            with self.assertRaisesRegex(InvocationError, 'too large'):
                download_image(url)
        error = httpx.ConnectError('private')
        with patch('aiython.capability_providers._public_https', return_value=True), \
                patch('aiython.capability_providers.httpx.Client', return_value=Client(Response([], error))):
            with self.assertRaisesRegex(InvocationError, 'download failed'):
                download_image(url)

    def test_video_policy_error_requires_structured_reason(self):
        self.assertIsNone(_openrouter_video_policy_error(RuntimeError('private')))
        error = SimpleNamespace(data=SimpleNamespace(error=SimpleNamespace(metadata={
            'ineligibility_reasons': [{'reason': 'other'}],
        })))
        self.assertIsNone(_openrouter_video_policy_error(error))
        error.data.error.metadata['ineligibility_reasons'] = 'not a list'
        self.assertIsNone(_openrouter_video_policy_error(error))
        error.data.error.metadata['ineligibility_reasons'] = [{'reason': 'zdr-violation-account'}]
        self.assertIn('Zero Data Retention', _openrouter_video_policy_error(error))

    def test_sdk_errors_keep_acceptance_and_parameter_details(self):
        class UnsupportedParamsError(Exception):
            pass

        self.sdk.embedding.side_effect = UnsupportedParamsError('Setting `dimensions` is unsupported')
        with self.assertRaisesRegex(ConfigError, 'dimensions'):
            self.adapter._call('embedding', 'gemini/test', input=['one'])
        self.sdk.embedding.side_effect = ValueError('Unmapped provider passed: secret')
        with self.assertRaisesRegex(InvocationError, 'does not support') as caught:
            self.adapter._call('embedding', 'gemini/test', input=['one'])
        self.assertFalse(caught.exception.accepted)
        rate_limit = RuntimeError('secret')
        rate_limit.status_code = 429
        self.sdk.embedding.side_effect = rate_limit
        with self.assertRaises(InvocationError) as caught:
            self.adapter._call('embedding', 'gemini/test', input=['one'])
        self.assertTrue(caught.exception.retryable)
        self.assertFalse(caught.exception.accepted)
        original = InvocationError('already classified', accepted=True)
        self.sdk.embedding.side_effect = original
        with self.assertRaises(InvocationError) as caught:
            self.adapter._call('embedding', 'gemini/test', input=['one'])
        self.assertIs(caught.exception, original)

    def test_openrouter_video_preserves_classified_errors_and_http_status(self):
        original = InvocationError('already classified', accepted=True)
        with patch('openrouter.OpenRouter') as factory:
            client = factory.return_value.__enter__.return_value
            client.video_generation.generate.side_effect = original
            with self.assertRaises(InvocationError) as caught:
                self.adapter._openrouter_video('generate', 'openrouter/test', prompt='scene')
            self.assertIs(caught.exception, original)
            failure = RuntimeError('private provider body')
            failure.raw_response = SimpleNamespace(status_code=503)
            client.video_generation.generate.side_effect = failure
            with self.assertRaisesRegex(InvocationError, 'HTTP 503') as caught:
                self.adapter._openrouter_video('generate', 'openrouter/test', prompt='scene')
            self.assertTrue(caught.exception.accepted)
            self.assertNotIn('private provider body', str(caught.exception))

    def test_media_artifact_and_large_upload_failures(self):
        with self.assertRaisesRegex(InvocationError, 'empty media'):
            self.adapter.artifact(self.context, b'', 'audio/pcm', Audio)
        audio = self.adapter.artifact(self.context, b'\x00\x00', 'audio/pcm', Audio)
        self.assertEqual(audio.path.read_bytes()[:4], b'RIFF')
        large = self.root / 'large.mp3'
        with large.open('wb') as file:
            file.truncate(10 * 1024 * 1024 + 1)
        asset = Audio(large)
        with self.assertRaisesRegex(CapabilityError, 'file upload API'):
            self.adapter._part(asset, 'openai/test')
        self.sdk.create_file.side_effect = RuntimeError('private')
        with self.assertRaisesRegex(InvocationError, 'Media upload failed') as caught:
            self.adapter._part(asset, 'gemini/test')
        self.assertTrue(caught.exception.accepted)

    def test_large_upload_timeout_and_failed_processing_are_accepted(self):
        large = self.root / 'large.mp3'
        with large.open('wb') as file:
            file.truncate(10 * 1024 * 1024 + 1)
        asset = Audio(large)
        self.sdk.create_file.return_value = {'id': 'files/1', 'status': 'uploaded'}
        fake_time = SimpleNamespace(monotonic=Mock(side_effect=[0, 100000]), sleep=Mock())
        with patch('aiython.capability_providers.time', fake_time):
            with self.assertRaisesRegex(InvocationError, 'Media upload failed') as caught:
                self.adapter._part(asset, 'gemini/test')
        self.assertTrue(caught.exception.accepted)
        self.sdk.create_file.return_value = {'id': 'files/2', 'status': 'failed'}
        with self.assertRaisesRegex(InvocationError, 'Media upload failed') as caught:
            self.adapter._part(asset, 'gemini/test')
        self.assertTrue(caught.exception.accepted)

    def test_completion_and_malformed_capability_responses(self):
        request = CapabilityRequest('reasoning', 'gemini/test', {}, provider='route')
        self.sdk.completion.return_value = {'choices': [{'message': {'content': [
            {'text': 'first'}, {'text': 'second'}, 'ignored',
        ]}}]}
        self.assertEqual(self.adapter._completion(request, 'prompt')[0], 'first\nsecond')
        self.sdk.completion.return_value = {'choices': [{'message': {'content': None}}]}
        with self.assertRaisesRegex(InvocationError, 'no text'):
            self.adapter._completion(request, 'prompt')
        self.sdk.embedding.return_value = {'data': []}
        with self.assertRaisesRegex(InvocationError, 'Malformed capability response'):
            self.invoke('embedding', {'inputs': ['one']})
        self.sdk.rerank.return_value = {'results': [{'index': 5}]}
        with self.assertRaisesRegex(InvocationError, 'Malformed capability response'):
            self.invoke('reranking', {'query': 'q', 'documents': ['one']})
        self.sdk.completion.return_value = {'choices': [{'message': {'content': '[]'}}]}
        with self.assertRaisesRegex(InvocationError, 'Malformed capability response'):
            self.invoke('document_understanding', {'assets': [self.asset('doc.pdf')]})
        self.sdk.completion.return_value = {'choices': [{'message': {'content': json.dumps([
            {'text': 'x', 'asset_index': 1, 'page': 1},
        ])}}]}
        with self.assertRaisesRegex(InvocationError, 'Malformed capability response'):
            self.invoke('document_understanding', {'assets': [self.asset('doc2.pdf')]})

    def test_mixed_embedding_input_and_nonreasoning_context(self):
        image = self.asset('photo.png', b'image', Image)
        self.sdk.embedding.side_effect = [
            {'data': [{'embedding': [1.0, 0.0]}]},
            {'data': [{'embedding': [0.0, 1.0]}]},
        ]
        vectors = self.invoke('embedding', {'inputs': ['word', image]})
        self.assertEqual(vectors.vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertIn('data:image/png;base64,', str(self.sdk.embedding.call_args.kwargs['input']))
        self.sdk.completion.return_value = {'choices': [{'message': {'content': 'caption'}}]}
        self.assertEqual(self.invoke('vision', {'prompt': 'describe', 'context': {'source': 'page 1'},
                                                'assets': [image]}), 'caption')
        self.assertIn('page 1', str(self.sdk.completion.call_args.kwargs['messages']))

    def test_video_submission_and_speech_stream_edges(self):
        first = self.asset('first.png', b'one', Image)
        second = self.asset('second.png', b'two', Image)
        with self.assertRaisesRegex(CapabilityError, 'one starting image'):
            self.invoke('video', {'mode': 'generate', 'prompt': 'scene', 'assets': [first, second]})
        self.sdk.video_generation.return_value = {}
        with self.assertRaisesRegex(InvocationError, 'no job ID') as caught:
            self.invoke('video', {'mode': 'generate', 'prompt': 'scene'})
        self.assertTrue(caught.exception.accepted)
        self.sdk.speech.return_value = SimpleNamespace(read=lambda: b'ID3audio')
        self.assertEqual(self.invoke('text_to_speech', {'text': 'hello'}).path.read_bytes(), b'ID3audio')

    def test_video_job_unknown_failed_and_download_error(self):
        with self.assertRaisesRegex(CapabilityError, 'Unknown video job'):
            self.adapter.wait_video('unknown', self.context)
        self.context.store.job('failed-job', {'status': 'pending', 'model': 'gemini/test'})
        self.sdk.video_status.return_value = {'status': 'failed'}
        with self.assertRaisesRegex(InvocationError, 'Video job failed'):
            self.adapter.wait_video('failed-job', self.context)
        self.assertEqual(self.context.store.job('failed-job')['status'], 'failed')
        self.context.store.job('download-job', {'status': 'pending', 'model': 'gemini/test'})
        self.sdk.video_status.return_value = {'status': 'completed'}
        self.sdk.video_content.side_effect = RuntimeError('private')
        with self.assertRaisesRegex(InvocationError, 'Video download failed'):
            self.adapter.wait_video('download-job', self.context)

    def test_video_job_config_errors_pass_through_and_pending_job_can_time_out(self):
        self.context.store.job('config-job', {'status': 'pending', 'model': 'gemini/test'})
        self.sdk.video_status.side_effect = ConfigError('bad status route')
        with self.assertRaisesRegex(ConfigError, 'bad status route'):
            self.adapter.wait_video('config-job', self.context)
        self.sdk.video_status.side_effect = None
        self.sdk.video_status.return_value = {'status': 'completed'}
        self.sdk.video_content.side_effect = ConfigError('bad content route')
        with self.assertRaisesRegex(ConfigError, 'bad content route'):
            self.adapter.wait_video('config-job', self.context)
        self.sdk.video_status.return_value = {'status': 'pending'}
        fake_time = SimpleNamespace(monotonic=Mock(side_effect=[0, 100, 601]), sleep=Mock())
        with patch('aiython.capability_providers.time', fake_time):
            with self.assertRaisesRegex(InvocationError, 'remains pending'):
                self.adapter.wait_video('config-job', self.context)
