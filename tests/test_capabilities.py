import contextlib
from dataclasses import dataclass, replace
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aithon.agent import ToolAgent
from aithon.assets import Image, Document, Video
from aithon.capabilities import (CapabilityRuntime, CapabilityResult, Embeddings,
    CapabilityError, CapabilityPermissionError, InvocationError, LocalProvider)
from aithon.cli import run_script, main
from aithon.config import resolve, describe
from aithon.models import ProfileConfig, ResolvedConfig, AgentRequest, SourceSpan, ConfigError
from aithon.runtime import Runtime, RuntimeBridge
from aithon.type_constraints import validate_output


def call(id, name, **args):
    return {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def reply(*calls):
    return {'role': 'assistant', 'content': None, 'tool_calls': list(calls)}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = ProfileConfig('default', 'fake', 'model', routes={
            c: [{'provider': 'fake', 'model': 'model'}] for c in ('reasoning', 'embedding', 'reranking', 'vision', 'image_generation', 'speech_to_text')})
        self.config = ResolvedConfig(None, self.root, 'default', {'default': self.profile})
        self.runtime = Runtime(self.config)
        self.caps = self.runtime.capabilities
        self.adapter = Mock(version='test', api_base=None)
        self.adapter.capabilities.return_value = set(self.profile.routes)
        self.caps.adapters[('default', 'fake')] = self.adapter

    def bridge(self):
        return RuntimeBridge(inspect.currentframe().f_back, self.runtime)

    def test_same_reasoning_model_is_not_advertised_as_a_nested_call(self):
        legacy = ProfileConfig('legacy', 'fake', 'model')
        names = {spec['name'] for spec in self.caps.describe(legacy)}
        self.assertNotIn('reasoning', names)
        names_with_route = {spec['name'] for spec in self.caps.describe(self.profile)}
        self.assertNotIn('reasoning', names_with_route)
        self.profile.routes['reasoning'] = [{'provider': 'other', 'model': 'different-model'}]
        names_with_override = {spec['name'] for spec in self.caps.describe(self.profile)}
        self.assertIn('reasoning', names_with_override)

    def test_plan_one_response_and_original_product_identity(self):
        @dataclass
        class Product:
            name: str
        product = Product('red shoe')
        other = Product('blue boot')
        records = [{'text': 'red shoe', 'source': 'catalog:1', 'object': product},  # noqa: F841 - read through the live frame
                   {'text': 'blue boot', 'source': 'catalog:2', 'object': other}]
        def invoke(request, context):
            if request.capability == 'embedding':
                vectors = [[1.,0.], [0.,1.]] if len(request.params['inputs']) == 2 else [[1.,0.]]
                return CapabilityResult(Embeddings(vectors, 'same-space'))
            return CapabilityResult(request.params['documents'])
        self.adapter.invoke.side_effect = invoke
        steps = [
            {'id':'docs', 'capability':'embedding', 'params': {'inputs': {'$binding':'records'}}},
            {'id':'query', 'capability':'embedding', 'params': {'inputs':['red shoe']}},
            {'id':'index', 'capability':'indexing', 'params': {'embeddings': {'$ref':'docs'}, 'documents': {'$binding':'records'}}},
            {'id':'search', 'capability':'semantic_search', 'params': {'index': {'$ref':'index'}, 'query': {'$ref':'query'}, 'limit':1}},
            {'id':'rerank', 'capability':'reranking', 'params': {'query':'red shoe', 'documents': {'$ref':'search'}}}]
        provider = Mock()
        provider.complete.return_value = reply(call('plan', 'run_plan', steps=steps, output={'$ref':'rerank','path':['*','object']}, result_id='answer'),
                                               call('done','finish',result_from='answer'))
        request = AgentRequest('find a shoe', '', {}, {}, SourceSpan('test.py',1,0,1,1), self.profile, (), output_type='list[Product]')
        result = ToolAgent(provider).execute(request, self.bridge())
        self.assertIs(result[0], product)
        self.assertEqual(provider.complete.call_count, 1)

    def test_document_pipeline_citations_without_extra_agent_round(self):
        doc = self.root / 'policy.md'
        doc.write_text('Refunds are available for 30 days.')
        self.profile.routes['document_understanding'] = [{'provider':'local', 'model':'text'}]
        self.caps.adapters[('default','local')] = LocalProvider()
        def invoke(request, context):
            if request.capability == 'embedding':
                return CapabilityResult(Embeddings([[1.,0.] for _ in request.params['inputs']], 'docs'))
            if request.capability == 'reranking': return CapabilityResult(request.params['documents'])
            return CapabilityResult('30 days [' + request.params['context'][0]['source'] + ']')
        self.adapter.invoke.side_effect = invoke
        steps = [
            {'id':'read','capability':'document_understanding','params':{'assets':['policy.md']}},
            {'id':'emb','capability':'embedding','params':{'inputs':{'$ref':'read'}}},
            {'id':'idx','capability':'indexing','params':{'documents':{'$ref':'read'},'embeddings':{'$ref':'emb'}}},
            {'id':'q','capability':'embedding','params':{'inputs':['Refund period?']}},
            {'id':'s','capability':'semantic_search','params':{'index':{'$ref':'idx'},'query':{'$ref':'q'}}},
            {'id':'r','capability':'reranking','params':{'query':'Refund period?','documents':{'$ref':'s'}}},
            {'id':'a','capability':'reasoning','params':{'prompt':'Answer with sources','context':{'$ref':'r'}}}]
        result = self.caps.run_plan(self.profile, steps, {'$ref':'a'}, self.bridge())
        self.assertIn('policy.md:1', result)
        self.assertIn('30 days', result)

    def test_plain_path_script_uses_document_capability_without_import(self):
        (self.root / 'policy.md').write_text('Refunds are available for 30 days.')
        self.profile.routes['document_understanding'] = [{'provider': '$local', 'model': 'text'}]
        script = self.root / 'main.py'
        script.write_text('document_path = "policy.md"\nchunks = extract text from document_path\n')
        provider = Mock()
        provider.complete.return_value = reply(
            call('plan', 'run_plan', steps=[{
                'id': 'read', 'capability': 'document_understanding',
                'params': {'assets': [{'$binding': 'document_path'}]},
            }], output={'$ref': 'read'}, result_id='chunks'),
            call('done', 'finish', result_from='chunks'),
        )

        result = run_script(script, config=self.config, agent_factory=lambda _: ToolAgent(provider))

        self.assertIn('Refunds are available for 30 days.', result['chunks'][0]['text'])
        self.assertEqual(provider.complete.call_count, 1)

    def test_embedding_can_use_a_plain_path_binding_as_an_asset(self):
        query_image_path = 'shoe.jpg'
        (self.root / query_image_path).write_bytes(b'image')
        self.adapter.invoke.return_value = CapabilityResult(Embeddings([[1., 0.]], 'images'))
        steps = [{'id': 'embed', 'capability': 'embedding',
                  'params': {'inputs': [{'$asset': {'$binding': 'query_image_path'}}]}}]

        result = self.caps.run_plan(self.profile, steps, {'$ref': 'embed'}, self.bridge())

        self.assertEqual(result.vectors, [[1., 0.]])
        image = self.adapter.invoke.call_args.args[0].params['inputs'][0]
        self.assertEqual(image.path, self.root / query_image_path)
        self.assertEqual(image.mime_type, 'image/jpeg')

    def test_cycle_missing_route_and_invalid_later_input_have_no_effects(self):
        for steps in (
            [{'id':'a','capability':'reasoning','params':{'prompt': {'$ref':'a'}}}],
            [{'id':'a','capability':'image_generation','params':{'prompt':'create'}},
             {'id':'b','capability':'reasoning','params':{'prompt':42}}],
            [{'id':'a','capability':'image_generation','params':{'prompt':'create'}},
             {'id':'b','capability':'text_to_speech','params':{'text':'hello'}}],
        ):
            with self.assertRaises((CapabilityError, ConfigError)):
                self.caps.run_plan(self.profile, steps, {'$ref':'a'}, self.bridge())
        self.adapter.invoke.assert_not_called()

    def test_completed_steps_reused_after_failure(self):
        bridge = self.bridge()
        image = self.root / 'created.png'; image.write_bytes(b'png')
        count = [0]
        def invoke(request, context):
            if request.capability == 'image_generation':
                count[0] += 1
                return CapabilityResult(Image(image))
            raise InvocationError('rate limited', retryable=True)
        self.adapter.invoke.side_effect = invoke
        steps = [{'id':'image','capability':'image_generation','params':{'prompt':'a tree'}},
                 {'id':'fail','capability':'vision','params':{'prompt':'describe','assets':[{'$ref':'image'}]}}]
        with self.assertRaises(InvocationError): self.caps.run_plan(self.profile, steps, {'$ref':'fail'}, bridge)
        self.adapter.invoke.side_effect = lambda r,c: CapabilityResult('tree')
        self.assertEqual(self.caps.run_plan(self.profile, steps, {'$ref':'fail'}, bridge), 'tree')
        self.assertEqual(count[0], 1)

    def test_unknown_acceptance_prevents_resubmission(self):
        self.adapter.invoke.side_effect = InvocationError('unknown', accepted=True)
        bridge = self.bridge()
        steps = [{'id':'image','capability':'image_generation','params':{'prompt':'tree'}}]
        with self.assertRaises(InvocationError): self.caps.run_plan(self.profile, steps, {'$ref':'image'}, bridge)
        with self.assertRaisesRegex(CapabilityError, 'acceptance is unknown'):
            self.caps.run_plan(self.profile, steps, {'$ref':'image'}, bridge)
        self.assertEqual(self.adapter.invoke.call_count, 1)

    def test_cache_invalidation_and_permission_before_cache(self):
        path = self.root / 'photo.png'; path.write_bytes(b'one')
        self.adapter.invoke.return_value = CapabilityResult('image analysis')
        params = {'assets':[str(path)],'prompt':'describe'}
        self.caps.invoke(self.profile,'vision',params)
        self.caps.invoke(self.profile,'vision',params)
        self.assertEqual(self.adapter.invoke.call_count,1)
        path.write_bytes(b'two')
        self.caps.invoke(self.profile,'vision',params)
        self.assertEqual(self.adapter.invoke.call_count,2)
        restricted = replace(self.profile, permissions=('network',))
        with patch.object(self.caps.store,'cached') as cached:
            with self.assertRaises(CapabilityPermissionError): self.caps.invoke(restricted,'vision',params)
            cached.assert_not_called()
        self.caps.invoke(self.profile,'vision',params,cache_salt={'output_type':'str'})
        self.assertEqual(self.adapter.invoke.call_count,3)

    def test_route_fallback_only_explicit_rejection(self):
        other = Mock(version='test')
        other.capabilities.return_value = {'reasoning'}
        other.invoke.return_value = CapabilityResult('fallback')
        self.caps.adapters[('default','second')] = other
        self.profile.routes['reasoning'].append({'provider':'second','model':'other'})
        self.adapter.invoke.side_effect = InvocationError('429',retryable=True)
        self.assertEqual(self.caps.invoke(self.profile,'reasoning',{'prompt':'hi'}),'fallback')
        other.reset_mock()
        self.adapter.invoke.side_effect = InvocationError('timeout',retryable=True,accepted=True)
        with self.assertRaises(InvocationError): self.caps.invoke(self.profile,'reasoning',{'prompt':'hi'})
        other.invoke.assert_not_called()

    def test_embedding_space_mismatch(self):
        index = self.caps.invoke(self.profile,'indexing',{'embeddings':Embeddings([[1.,0.]],'old'), 'documents':['one']})
        with self.assertRaisesRegex(CapabilityError,'space mismatch'):
            self.caps.invoke(self.profile,'semantic_search',{'index':index,'query':Embeddings([[1.,0.]],'new')})

    def test_v3_routes_and_secret_redaction(self):
        path = self.root / 'aithon.toml'
        path.write_text('''version = 3
model = "openrouter/chosen"
api_key_env = "SECRET_SENTINEL"
[capabilities]
embedding = "openai/text-embedding-3-small"
''')
        config = resolve(self.root / 'main.py')
        self.assertEqual(config.profiles['default'].model,'openrouter/chosen')
        self.assertNotIn('SECRET_SENTINEL',json.dumps(describe(config)['profiles']['default']['routes']))

    def test_typed_assignment_and_directive_reaches_request(self):
        script = self.root / 'main.py'
        script.write_text('# aithon: capability="semantic-search" provider="local"\nresult: list[int] = find these results\n')
        requests = []
        class FakeAgent:
            def execute(self, request, runtime):
                requests.append(request)
                return [1,2]
        result = run_script(script, config=self.config, agent_factory=lambda p:FakeAgent())
        self.assertEqual(result['result'],[1,2])
        self.assertEqual((requests[0].capability,requests[0].provider,requests[0].output_type),('semantic_search','local','list[int]'))

    def test_output_types_do_not_execute_annotation(self):
        for value, annotation in [(True,'int'),(['oops'],'list[int]'),(1,'__import__("os").system("echo BAD")')]:
            with self.assertRaises(CapabilityError): validate_output(value,annotation,{})
        validate_output([1,2],'list[int]',{})
        validate_output(None,'int | None',{})
        validate_output('yes','Literal["yes", "no"]',{})

    def test_explain_is_static_and_unknown(self):
        script = self.root / 'main.py'; script.write_text('answer = find the answer\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('aithon.providers.sdk') as remote:
            main(['--explain',str(script)])
        remote.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['blocks'][0]['cache'],'unknown')


class LiveCapabilityTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('AITHON_LIVE_CAPABILITY_CONFIG') and
                         __import__('os').environ.get('AITHON_LIVE_CAPABILITY_CASES'),
                         'Paid provider tests are opt-in')
    def test_configured_live_routes(self):
        import os
        path = Path(os.environ['AITHON_LIVE_CAPABILITY_CONFIG']).resolve()
        config = resolve(path.parent / 'main.py',config_path=str(path))
        runtime = CapabilityRuntime(config)
        profile = config.profiles[config.default_profile]
        cases = json.loads(os.environ['AITHON_LIVE_CAPABILITY_CASES'])
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(capability=case['capability']):
                result = runtime.invoke(profile,case['capability'],case['params'],cache=False)
                self.assertIsNotNone(result)


class CapabilityEdgeTests(unittest.TestCase):
    setUp = CapabilityTests.setUp
    bridge = CapabilityTests.bridge
    def test_media_path_reference_reaches_embedding_as_asset(self):
        path = self.root / 'shoe.png'; path.write_bytes(b'image')
        def invoke(request, context):
            self.assertIsInstance(request.params['inputs'][0],Document)
            self.assertEqual(request.params['inputs'][0].path,path)
            return CapabilityResult(Embeddings([[1.,0.]],'visual'))
        self.adapter.invoke.side_effect = invoke
        steps = [{'id':'visual','capability':'embedding','params':{'inputs':[{'$asset':'shoe.png'}]}}]
        value = self.caps.run_plan(self.profile,steps,{'$ref':'visual'},self.bridge())
        self.assertEqual(value.space,'visual')

    def test_job_resume_uses_saved_provider_and_model(self):
        self.profile.routes['video'] = [{'provider':'fake','model':'veo','modes':['generate']}]
        self.adapter.api_base = 'https://example.test/v1'
        self.adapter.wait_video.return_value = Video(self.root/'done.mp4')
        self.caps.store.job('operations/test',{'provider_url':'https://example.test/v1','model':'veo','status':'pending'})
        self.assertIsInstance(self.caps.resume_job(self.profile,'operations/test'),Video)
        self.adapter.wait_video.assert_called_once_with('operations/test',self.caps)
        self.profile.routes['video'][0]['model'] = 'changed'
        with self.assertRaises(ConfigError): self.caps.resume_job(self.profile,'operations/test')

    def test_permission_denial_stops_agent_without_another_model_call(self):
        profile = replace(self.profile,permissions=('network',))
        provider = Mock()
        provider.complete.return_value = reply(call('exec','execute',code='1+1'),call('done','finish'))
        request = AgentRequest('calculate', '', {}, {}, SourceSpan('test.py',1,0,1,1),profile,())
        with self.assertRaises(CapabilityPermissionError): ToolAgent(provider).execute(request,self.bridge())
        self.assertEqual(provider.complete.call_count,1)

    def test_indexes_share_storage_but_not_live_objects(self):
        first, second = object(), object()
        def build(obj):
            return self.caps.invoke(self.profile,'indexing',{'embeddings':Embeddings([[1.,0.]],'space'),
                'documents':[{'text':'same text','object':obj}]})
        a, b = build(first), build(second)
        self.assertEqual(a.name,b.name)
        for index, obj in [(a,first),(b,second)]:
            hits = self.caps.invoke(self.profile,'semantic_search',{'index':index,'query':Embeddings([[1.,0.]],'space')})
            self.assertIs(hits[0]['object'],obj)

    def test_generated_media_cache_is_opt_in_and_checks_artifact(self):
        image = self.root / 'out.png'; image.write_bytes(b'image')
        self.adapter.invoke.return_value = CapabilityResult(Image(image))
        for _ in range(2): self.caps.invoke(self.profile,'image_generation',{'prompt':'tree'})
        self.assertEqual(self.adapter.invoke.call_count,2)
        for _ in range(2): self.caps.invoke(self.profile,'image_generation',{'prompt':'tree'},cache=True)
        self.assertEqual(self.adapter.invoke.call_count,3)
        image.write_bytes(b'changed')
        self.caps.invoke(self.profile,'image_generation',{'prompt':'tree'},cache=True)
        self.assertEqual(self.adapter.invoke.call_count,4)

    def test_permissions_reject_whole_plan_before_generation(self):
        profile = replace(self.profile,permissions=('network','generate_file','write_filesystem'))
        steps = [{'id':'make','capability':'image_generation','params':{'prompt':'tree'}},
                 {'id':'read','capability':'vision','params':{'assets':['private.png'],'prompt':'describe'}}]
        with self.assertRaises(CapabilityPermissionError):
            self.caps.run_plan(profile,steps,{'$ref':'read'},self.bridge())
        self.adapter.invoke.assert_not_called()

    def test_provider_directive_only_restricts_target_step(self):
        self.adapter.invoke.return_value = CapabilityResult(Embeddings([[1.,0.]],'same'))
        steps = [{'id':'e','capability':'embedding','params':{'inputs':['doc']}},
                 {'id':'idx','capability':'indexing','params':{'embeddings':{'$ref':'e'},'documents':['doc']}},
                 {'id':'s','capability':'semantic_search','params':{'index':{'$ref':'idx'},'query':{'$ref':'e'}}}]
        result = self.caps.run_plan(self.profile,steps,{'$ref':'s'},self.bridge(),provider='$local')
        self.assertEqual(result[0]['text'],'doc')

    def test_independent_steps_run_concurrently(self):
        import threading
        barrier = threading.Barrier(2)
        def invoke(request, context):
            barrier.wait(timeout=2)
            return CapabilityResult(request.params['prompt'])
        self.adapter.invoke.side_effect = invoke
        steps = [{'id':name,'capability':'reasoning','params':{'prompt':name}} for name in ('a','b')]
        self.assertEqual(self.caps.run_plan(self.profile,steps,{'$ref':'b'},self.bridge()),'b')

    def test_type_repair_keeps_generated_artifact(self):
        image = self.root / 'out.png'; image.write_bytes(b'image')
        self.adapter.invoke.return_value = CapabilityResult(Image(image))
        provider = Mock()
        provider.complete.side_effect = [reply(
            call('p','run_plan',steps=[{'id':'make','capability':'image_generation','params':{'prompt':'tree'}}],
                 output={'$ref':'make'},result_id='image'),call('f','finish',result_from='image')),
            reply(call('e','evaluate',code='"saved image"',result_id='text'),call('f2','finish',result_from='text'))]
        request = AgentRequest('generate and report', '', {}, {}, SourceSpan('test.py',1,0,1,1),self.profile,(),output_type='str')
        self.assertEqual(ToolAgent(provider).execute(request,self.bridge()),'saved image')
        self.assertEqual(self.adapter.invoke.call_count,1)

    def test_dataclass_slots_and_typeddict_constraints(self):
        from typing import TypedDict, NotRequired
        @dataclass(slots=True)
        class Product:
            name: str
        class Record(TypedDict):
            count: int
            note: NotRequired[str]
        validate_output(Product('shoe'),'Product',{'Product':Product})
        validate_output({'count':2},'Record',{'Record':Record})
        with self.assertRaises(CapabilityError): validate_output({'count':'two'},'Record',{'Record':Record})
