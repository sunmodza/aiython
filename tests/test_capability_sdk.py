"""Offline contracts for every LiteLLM capability path."""
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import Mock, patch

from aiython.assets import Audio, Document, Image, Video
from aiython.capabilities import CapabilityRequest, CapabilityRuntime, Embeddings, InvocationError
from aiython.capability_providers import LiteLLMAdapter
from aiython.models import ConfigError, ProfileConfig, ResolvedConfig


class LiteLLMCapabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = ProfileConfig('default', 'litellm', 'gemini/test')
        self.config = ResolvedConfig(None, self.root, 'default', {'default': self.profile},
                                     providers={'sdk': {'model': 'gemini/test'}})
        self.context = CapabilityRuntime(self.config)
        self.adapter = LiteLLMAdapter(self.config, self.profile, 'sdk')
        self.sdk = Mock()
        patcher = patch('aiython.capability_providers.sdk', return_value=self.sdk)
        patcher.start()
        self.addCleanup(patcher.stop)

    def invoke(self, capability, params, model='gemini/test'):
        return self.adapter.invoke(CapabilityRequest(capability, model, params, provider='sdk'), self.context).value

    def asset(self, name, data=b'data', kind=Document):
        path = self.root / name
        path.write_bytes(data)
        return kind(path)

    def test_text_vision_documents_and_video_understanding(self):
        self.sdk.completion.return_value = {'choices': [{'message': {'content': 'answer'}}]}
        self.assertEqual(self.invoke('reasoning', {'prompt': 'question', 'context': [{'text': 'fact', 'object': object()}]}), 'answer')
        self.assertNotIn('object', str(self.sdk.completion.call_args.kwargs['messages']))
        image = self.asset('photo.png', kind=Image)
        self.assertEqual(self.invoke('vision', {'prompt': 'describe', 'assets': [image]}), 'answer')
        self.assertIn('data:image/png;base64,', str(self.sdk.completion.call_args.kwargs['messages']))
        pdf = self.asset('paper.pdf')
        self.sdk.completion.return_value = {'choices': [{'message': {'content': '[{"text":"30 days","asset_index":0,"page":2}]'}}]}
        self.assertEqual(self.invoke('document_understanding', {'assets': [pdf]}),
                         [{'text': '30 days', 'source': str(pdf.path) + ':page 2'}])
        self.sdk.completion.return_value = {'choices': [{'message': {'content': 'video summary'}}]}
        self.assertEqual(self.invoke('video', {'mode': 'understand', 'prompt': 'summarize',
                                               'assets': [self.asset('clip.mp4')]}), 'video summary')

    def test_embedding_and_reranking_preserve_records(self):
        self.sdk.embedding.return_value = {'data': [
            {'index': 1, 'embedding': [0., 1.]}, {'index': 0, 'embedding': [1., 0.]}]}
        vectors = self.invoke('embedding', {'inputs': ['one', 'two']})
        self.assertIsInstance(vectors, Embeddings)
        self.assertEqual(vectors.vectors, [[1., 0.], [0., 1.]])
        first, second = {'text': 'one', 'object': object()}, {'text': 'two', 'object': object()}
        self.sdk.rerank.return_value = {'results': [{'index': 1, 'relevance_score': .9}]}
        result = self.invoke('reranking', {'query': 'two', 'documents': [first, second]})
        self.assertIs(result[0], second)
        self.assertEqual(self.sdk.rerank.call_args.kwargs['documents'], ['one', 'two'])

    def test_audio_image_generation_and_editing(self):
        audio = self.asset('speech.mp3', b'ID3abc', Audio)
        self.sdk.transcription.return_value = {'text': 'spoken words'}
        self.assertEqual(self.invoke('speech_to_text', {'assets': [audio]}), 'spoken words')
        self.sdk.speech.return_value = b'ID3audio'
        speech = self.invoke('text_to_speech', {'text': 'hello'})
        self.assertIsInstance(speech, Audio)
        self.assertEqual(speech.path.read_bytes(), b'ID3audio')
        self.sdk.image_generation.return_value = {'data': [{'b64_json': base64.b64encode(b'PNG').decode()}]}
        generated = self.invoke('image_generation', {'prompt': 'tree'})
        self.assertIsInstance(generated, Image)
        self.assertEqual(generated.path.read_bytes(), b'PNG')
        self.assertNotIn('response_format', self.sdk.image_generation.call_args.kwargs)
        self.sdk.image_edit.return_value = {'data': [{'b64_json': base64.b64encode(b'edited').decode()}]}
        edited = self.invoke('image_editing', {'prompt': 'blue', 'assets': [self.asset('photo.png', kind=Image)]})
        self.assertEqual(edited.path.read_bytes(), b'edited')
        self.assertNotIn('response_format', self.sdk.image_edit.call_args.kwargs)

    def test_openrouter_audio_uses_litellm_openai_compatible_endpoints(self):
        from aiython.providers import sdk as load_sdk
        litellm = load_sdk()

        requests = []
        class AudioHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                requests.append((self.path, body))
                if self.path == '/v1/audio/transcriptions':
                    response, mime = b'{"text":"heard words"}', 'application/json'
                else:
                    response, mime = b'ID3spoken', 'audio/mpeg'
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), AudioHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config.providers['sdk']['api_base'] = f'http://127.0.0.1:{server.server_port}/v1'
        self.config.providers['sdk']['api_key_env'] = 'TEST_AUDIO_KEY'
        self.config.secrets['TEST_AUDIO_KEY'] = 'test-key'
        try:
            with patch('aiython.capability_providers.sdk', return_value=litellm):
                audio = self.asset('speech.mp3', b'ID3abc', Audio)
                self.assertEqual(self.invoke('speech_to_text', {'assets': [audio]},
                                             model='openrouter/openai/gpt-transcribe'), 'heard words')
                speech = self.invoke('text_to_speech', {'text': 'hello', 'voice': 'alloy'},
                                     model='openrouter/elevenlabs/eleven-turbo-v2')
            self.assertEqual(speech.path.read_bytes(), b'ID3spoken')
            self.assertEqual([path for path, _ in requests],
                             ['/v1/audio/transcriptions', '/v1/audio/speech'])
            self.assertIn(b'openai/gpt-transcribe', requests[0][1])
            self.assertIn(b'name="response_format"\r\n\r\njson', requests[0][1])
            self.assertNotIn(b'verbose_json', requests[0][1])
            self.assertIn(b'elevenlabs/eleven-turbo-v2', requests[1][1])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_openrouter_image_generation_uses_litellm_without_response_format(self):
        from aiython.providers import sdk as load_sdk
        litellm = load_sdk()

        requests = []
        encoded = base64.b64encode(b'PNG image').decode()
        class ImageHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
                response = ('{"choices":[{"message":{"role":"assistant","content":"",'
                            '"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,'
                            + encoded + '"}}]}}],"usage":{}}').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), ImageHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config.providers['sdk']['api_base'] = f'http://127.0.0.1:{server.server_port}/v1'
        self.config.providers['sdk']['api_key_env'] = 'TEST_IMAGE_KEY'
        self.config.secrets['TEST_IMAGE_KEY'] = 'test-key'
        try:
            with patch('aiython.capability_providers.sdk', return_value=litellm):
                image = self.invoke('image_generation', {'prompt': 'a blue circle'},
                                    model='openrouter/google/gemini-3.1-flash-image')
            self.assertEqual(image.path.read_bytes(), b'PNG image')
            self.assertEqual(requests[0][0], '/v1/chat/completions')
            self.assertNotIn(b'response_format', requests[0][1])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_video_submit_status_content_and_resume(self):
        self.sdk.video_generation.return_value = {'id': 'operations/test'}
        self.sdk.video_status.side_effect = [{'status': 'in_progress'}, {'status': 'completed'}]
        self.sdk.video_content.return_value = b'video-content'
        with patch('aiython.capability_providers.time.sleep'):
            video = self.invoke('video', {'mode': 'generate', 'prompt': 'tree'})
        self.assertIsInstance(video, Video)
        self.assertEqual(video.path.read_bytes(), b'video-content')
        self.assertEqual(self.context.store.job('operations/test')['status'], 'complete')
        self.assertEqual(self.adapter.wait_video('operations/test', self.context).path, video.path)
        self.assertEqual(self.sdk.video_generation.call_count, 1)
        self.assertEqual(self.sdk.video_status.call_count, 2)
        self.assertEqual(self.sdk.video_status.call_args.kwargs['custom_llm_provider'], 'gemini')
        self.assertEqual(self.sdk.video_content.call_args.kwargs['custom_llm_provider'], 'gemini')

    def test_openrouter_video_submit_status_content_and_resume_with_sdk(self):
        requests = []
        status = {'value': 'pending'}

        class VideoHandler(BaseHTTPRequestHandler):
            def respond(self, code, body, mime='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(('POST', self.path, body, self.headers.get('Authorization')))
                self.respond(202, json.dumps({'id': 'job-123', 'polling_url': '/api/v1/videos/job-123',
                                              'status': 'pending'}).encode())

            def do_GET(self):
                requests.append(('GET', self.path, None, self.headers.get('Authorization')))
                if self.path == '/api/v1/videos/job-123/content?index=0':
                    self.respond(200, b'\x00\x00\x00\x18ftypmp42video', 'video/mp4')
                elif status['value'] == 'error':
                    self.respond(503, b'{"error":"temporarily unavailable"}')
                else:
                    self.respond(200, json.dumps({'id': 'job-123',
                                                  'polling_url': '/api/v1/videos/job-123',
                                                  'status': status['value']}).encode())

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), VideoHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config.providers['sdk']['api_base'] = f'http://127.0.0.1:{server.server_port}/api/v1'
        self.config.providers['sdk']['api_key_env'] = 'TEST_VIDEO_KEY'
        self.config.secrets['TEST_VIDEO_KEY'] = 'test-key'
        try:
            status['value'] = 'error'
            with self.assertRaisesRegex(InvocationError, 'resume operation job-123'):
                self.invoke('video', {'mode': 'generate', 'prompt': 'sunset',
                                      'assets': [self.asset('start.png', b'PNG', Image)]},
                            model='openrouter/minimax/hailuo-3')
            self.assertEqual(self.context.store.job('job-123')['status'], 'pending')
            status['value'] = 'completed'
            video = self.adapter.wait_video('job-123', self.context)
            self.assertEqual(video.path.read_bytes(), b'\x00\x00\x00\x18ftypmp42video')
            self.assertEqual(self.adapter.wait_video('job-123', self.context).path, video.path)
            self.assertEqual([request[0] for request in requests].count('POST'), 1)
            self.assertEqual(requests[0][1], '/api/v1/videos')
            self.assertEqual(requests[0][2]['model'], 'minimax/hailuo-3')
            self.assertEqual(requests[0][2]['frame_images'][0]['frame_type'], 'first_frame')
            self.assertTrue(requests[0][2]['frame_images'][0]['image_url']['url'].startswith('data:image/png;base64,'))
            self.assertTrue(all(request[3] == 'Bearer test-key' for request in requests))
            self.sdk.video_generation.assert_not_called()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_openrouter_video_submission_failures_are_not_retried(self):
        submissions = []
        status = {'code': 400}

        class RejectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                submissions.append(self.path)
                error = {'message': 'provider error', 'code': status['code']}
                if status['code'] == 404:
                    error['metadata'] = {'ineligibility_reasons': [
                        {'reason': 'zdr-violation-by-account', 'endpoint_count': 1}]}
                body = json.dumps({'error': error}).encode()
                self.send_response(status['code'])
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), RejectHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config.providers['sdk']['api_base'] = f'http://127.0.0.1:{server.server_port}/api/v1'
        try:
            for code, accepted in ((400, False), (429, False), (503, True)):
                with self.subTest(code=code):
                    status['code'] = code
                    with self.assertRaisesRegex(InvocationError, f'HTTP {code}') as caught:
                        self.invoke('video', {'mode': 'generate', 'prompt': 'sunset'},
                                    model='openrouter/minimax/hailuo-3')
                    self.assertEqual(caught.exception.accepted, accepted)
                    self.assertEqual(len(submissions), (400, 429, 503).index(code) + 1)
            status['code'] = 404
            with self.assertRaisesRegex(ConfigError, 'Zero Data Retention.*privacy setting'):
                self.invoke('video', {'mode': 'generate', 'prompt': 'sunset'},
                            model='openrouter/minimax/hailuo-3')
            self.assertEqual(len(submissions), 4)
            self.assertEqual(self.context.store.jobs(), [])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_video_pending_can_resume_without_resubmission(self):
        self.sdk.video_generation.return_value = {'id': 'operations/pending'}
        self.sdk.video_status.side_effect = TimeoutError('private')
        with self.assertRaisesRegex(InvocationError, 'resume operation'):
            self.invoke('video', {'mode': 'generate', 'prompt': 'tree'})
        self.assertEqual(self.context.store.job('operations/pending')['status'], 'pending')
        self.sdk.video_status.side_effect = None
        self.sdk.video_status.return_value = {'status': 'completed'}
        self.sdk.video_content.return_value = b'finished'
        resumed = self.adapter.wait_video('operations/pending', self.context)
        self.assertEqual(resumed.path.read_bytes(), b'finished')
        self.assertEqual(self.sdk.video_generation.call_count, 1)

    def test_runtime_v3_state_preserves_legacy_database(self):
        folder = self.root / '.aiython'
        folder.mkdir()
        legacy = folder / 'runtime.sqlite'
        legacy.write_bytes(b'legacy state')
        self.context.store.put('key', 'value')
        self.assertEqual(legacy.read_bytes(), b'legacy state')
        self.assertTrue((folder / 'runtime-v3.sqlite').is_file())

    def test_gemini_upload_and_processing(self):
        large = self.asset('long.mp3', b'x' * (10 * 1024 * 1024 + 1), Audio)
        self.sdk.create_file.return_value = {'id': 'files/123', 'status': 'uploaded'}
        self.sdk.file_retrieve.return_value = {'id': 'files/123', 'status': 'processed'}
        with patch('aiython.capability_providers.time.sleep'):
            part = self.adapter._part(large, 'gemini/test')
        self.assertEqual(part, {'type': 'file', 'file': {'file_id': 'files/123'}})
        self.assertEqual(self.sdk.create_file.call_count, 1)
        self.assertEqual(self.sdk.file_retrieve.call_count, 1)

    def test_rejected_and_ambiguous_failures(self):
        for status, retryable, accepted in ((400, False, False), (429, True, False), (503, False, True)):
            exc = Exception('secret')
            exc.status_code = status
            self.sdk.embedding.side_effect = exc
            with self.assertRaises(InvocationError) as caught:
                self.invoke('embedding', {'inputs': ['text']})
            self.assertEqual((caught.exception.retryable, caught.exception.accepted), (retryable, accepted))
            self.assertNotIn('secret', str(caught.exception))
            self.sdk.embedding.reset_mock()
        self.sdk.embedding.side_effect = ValueError('Unmapped provider passed in. Unable to get the response.')
        with self.assertRaisesRegex(InvocationError, 'LiteLLM does not support embedding') as caught:
            self.invoke('embedding', {'inputs': ['text']})
        self.assertFalse(caught.exception.accepted)
        self.sdk.embedding.side_effect = None
        self.sdk.embedding.return_value = {'data': []}
        with self.assertRaisesRegex(InvocationError, 'Malformed'):
            self.invoke('embedding', {'inputs': ['text']})

    def test_unsupported_sdk_parameter_stops_before_model_retry(self):
        class UnsupportedParamsError(Exception):
            pass

        self.sdk.image_generation.side_effect = UnsupportedParamsError(
            'Setting `response_format` is not supported by this model')
        with self.assertRaisesRegex(ConfigError, 'does not support response_format for image_generation'):
            self.invoke('image_generation', {'prompt': 'a blue circle'})

    def test_null_and_empty_provider_results_are_errors(self):
        self.sdk.completion.return_value = {'choices': [{'message': {'content': None}}]}
        with self.assertRaisesRegex(InvocationError, 'no text'):
            self.invoke('reasoning', {'prompt': 'question'})
        self.sdk.image_generation.return_value = {'data': [{'b64_json': ''}]}
        with self.assertRaises(InvocationError):
            self.invoke('image_generation', {'prompt': 'question'})
