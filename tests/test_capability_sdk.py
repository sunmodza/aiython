"""Offline contracts for every LiteLLM capability path."""
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aithon.assets import Audio, Document, Image, Video
from aithon.capabilities import CapabilityRequest, CapabilityRuntime, Embeddings, InvocationError
from aithon.capability_providers import LiteLLMAdapter
from aithon.models import ProfileConfig, ResolvedConfig


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
        patcher = patch('aithon.capability_providers.sdk', return_value=self.sdk)
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
        self.sdk.image_edit.return_value = {'data': [{'b64_json': base64.b64encode(b'edited').decode()}]}
        edited = self.invoke('image_editing', {'prompt': 'blue', 'assets': [self.asset('photo.png', kind=Image)]})
        self.assertEqual(edited.path.read_bytes(), b'edited')

    def test_video_submit_status_content_and_resume(self):
        self.sdk.video_generation.return_value = {'id': 'operations/test'}
        self.sdk.video_status.side_effect = [{'status': 'in_progress'}, {'status': 'completed'}]
        self.sdk.video_content.return_value = b'video-content'
        with patch('aithon.capability_providers.time.sleep'):
            video = self.invoke('video', {'mode': 'generate', 'prompt': 'tree'})
        self.assertIsInstance(video, Video)
        self.assertEqual(video.path.read_bytes(), b'video-content')
        self.assertEqual(self.context.store.job('operations/test')['status'], 'complete')
        self.assertEqual(self.adapter.wait_video('operations/test', self.context).path, video.path)
        self.assertEqual(self.sdk.video_generation.call_count, 1)
        self.assertEqual(self.sdk.video_status.call_count, 2)
        self.assertEqual(self.sdk.video_status.call_args.kwargs['custom_llm_provider'], 'gemini')
        self.assertEqual(self.sdk.video_content.call_args.kwargs['custom_llm_provider'], 'gemini')

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
        folder = self.root / '.aithon'
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
        with patch('aithon.capability_providers.time.sleep'):
            part = self.adapter._part(large, 'gemini/test')
        self.assertEqual(part, {'type': 'file', 'file': {'file_id': 'files/123'}})
        self.assertEqual(self.sdk.create_file.call_count, 1)
        self.assertEqual(self.sdk.file_retrieve.call_count, 1)

    def test_rejected_and_ambiguous_failures(self):
        for status, retryable, accepted in ((429, True, False), (503, False, True)):
            exc = Exception('secret')
            exc.status_code = status
            self.sdk.embedding.side_effect = exc
            with self.assertRaises(InvocationError) as caught:
                self.invoke('embedding', {'inputs': ['text']})
            self.assertEqual((caught.exception.retryable, caught.exception.accepted), (retryable, accepted))
            self.assertNotIn('secret', str(caught.exception))
            self.sdk.embedding.reset_mock()
        self.sdk.embedding.side_effect = None
        self.sdk.embedding.return_value = {'data': []}
        with self.assertRaisesRegex(InvocationError, 'Malformed'):
            self.invoke('embedding', {'inputs': ['text']})

    def test_null_and_empty_provider_results_are_errors(self):
        self.sdk.completion.return_value = {'choices': [{'message': {'content': None}}]}
        with self.assertRaisesRegex(InvocationError, 'no text'):
            self.invoke('reasoning', {'prompt': 'question'})
        self.sdk.image_generation.return_value = {'data': [{'b64_json': ''}]}
        with self.assertRaises(InvocationError):
            self.invoke('image_generation', {'prompt': 'question'})
