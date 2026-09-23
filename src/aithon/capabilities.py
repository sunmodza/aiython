"""Capability contracts, deterministic plans, project cache and local vector search.

This is a trusted in-process runtime, not a Python sandbox.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Protocol

from .assets import Asset, ASSET_TYPES, Document, VectorIndex
from .models import AithonError, ConfigError, ProviderError


class CapabilityError(AithonError):
    pass


class CapabilityPermissionError(CapabilityError):
    pass


class InvocationError(ProviderError):
    def __init__(self, message, *, retryable=False, accepted=False):
        super().__init__(message)
        self.retryable = retryable
        self.accepted = accepted


@dataclass
class Embeddings:
    vectors: list[list[float]]
    space: str

    def __post_init__(self):
        if (not self.vectors or not self.vectors[0] or
            any(len(v) != len(self.vectors[0]) or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                for v in self.vectors)):
            raise CapabilityError('Embedding vectors must be finite and have consistent dimensions')


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ('network',)
    cache: bool = False
    pure: bool = True
    output: str = 'str'

    def validate_inputs(self, params, *, references=False):
        self.validate(params)
        if self.name == 'video' and params.get('mode') == 'understand' and not params.get('assets'):
            raise CapabilityError('Video understanding requires assets')
        def ref(value):
            return references and type(value) is dict and bool({'$ref', '$binding', '$handle', '$asset'} & value.keys())
        for key, value in params.items():
            if ref(value): continue
            if key in ('prompt', 'text', 'voice', 'task_type', 'mode') and not isinstance(value, str):
                raise CapabilityError(f'{self.name}: {key} must be text')
            if key == 'query' and self.name == 'reranking' and not isinstance(value, str):
                raise CapabilityError('reranking query must be text')
            if key in ('dimensions', 'limit') and (type(value) is not int or value < 1):
                raise CapabilityError(f'{key} must be a positive integer')
            if key in ('assets', 'inputs', 'documents'):
                if not isinstance(value, list) or not value:
                    raise CapabilityError(f'{key} must be a nonempty list')
                if key == 'assets' and any(not ref(v) and not isinstance(v, (str, Path, Asset)) for v in value):
                    raise CapabilityError('assets must contain paths or Asset objects')
                if key == 'inputs' and any(not ref(v) and not isinstance(v, (str, Asset)) and
                    not (type(v) is dict and isinstance(v.get('text'), str)) for v in value):
                    raise CapabilityError('embedding inputs must be text, text records or assets')
            if not references and key in ('embeddings', 'query') and self.name in ('indexing', 'semantic_search') and not isinstance(value, Embeddings):
                raise CapabilityError(f'{key} must be Embeddings')
            if not references and key == 'index' and not isinstance(value, VectorIndex):
                raise CapabilityError('index must be VectorIndex')

    @property
    def input_schema(self):
        properties = {}
        for key in self.required + self.optional:
            if key in ('prompt', 'text', 'voice', 'mode', 'task_type') or key == 'query' and self.name == 'reranking':
                schema = {'type': 'string'}
            elif key in ('limit', 'dimensions'):
                schema = {'type': 'integer', 'minimum': 1}
            elif key in ('assets', 'inputs', 'documents'):
                schema = {'type': 'array', 'minItems': 1}
            elif key in ('index', 'embeddings', 'query'):
                schema = {'type': 'object', 'description': 'Live VectorIndex or Embeddings reference'}
            else:
                schema = {}
            properties[key] = schema
        return {'type': 'object', 'properties': properties, 'required': list(self.required), 'additionalProperties': False}

    @property
    def output_schema(self):
        if ' | ' in self.output:
            return {'anyOf': [{'type': 'string' if name == 'str' else 'array' if name.startswith('list') else 'object',
                               'x-python-type': name} for name in self.output.split(' | ')]}
        kind = 'string' if self.output == 'str' else 'array' if self.output.startswith('list') else 'object'
        return {'type': kind, 'x-python-type': self.output}

    def describe(self):
        return {'name': self.name, 'description': self.description, 'input_schema': self.input_schema,
                'output_schema': self.output_schema}


    def validate(self, params):
        if not isinstance(params, dict) or not set(self.required) <= params.keys():
            raise CapabilityError(f'{self.name}: missing required inputs {self.required}')
        if params.keys() - set(self.required + self.optional):
            raise CapabilityError(f'{self.name}: unknown input fields')


SPECS = [
    CapabilitySpec('reasoning', 'Answer prompt using optional context; cite source locations from context.', ('prompt',), ('context',)),
    CapabilitySpec('vision', 'Analyze images with prompt.', ('assets', 'prompt'), cache=True),
    CapabilitySpec('document_understanding', 'Extract/answer documents. Local adapter reads txt/md; Gemini handles PDF.', ('assets',), ('prompt',), permissions=('read_asset',), cache=True, output='str | list[dict]'),
    CapabilitySpec('embedding', 'Embed inputs (text or assets); returns vectors plus their embedding space.', ('inputs',), ('dimensions', 'task_type'), cache=True, output='Embeddings'),
    CapabilitySpec('indexing', 'Build index from embeddings and documents (text or {text,source,object}); preserves live object identity.', ('embeddings', 'documents'), permissions=('write_filesystem',), cache=False, pure=False, output='VectorIndex'),
    CapabilitySpec('semantic_search', 'Search index with a single query Embeddings in the SAME space; returns hits with text/source/score/object.', ('index', 'query'), ('limit',), permissions=('read_asset',), output='list[dict]'),
    CapabilitySpec('reranking', 'Reorder documents/search hits by relevance; retains source objects.', ('query', 'documents'), ('limit',), output='list'),
    CapabilitySpec('speech_to_text', 'Transcribe audio assets verbatim.', ('assets',), ('prompt',), cache=True),
    CapabilitySpec('text_to_speech', 'Generate speech from text.', ('text',), ('voice',), permissions=('network', 'generate_file', 'write_filesystem'), pure=False, output='Audio'),
    CapabilitySpec('image_generation', 'Generate image from prompt.', ('prompt',), (), ('network', 'generate_file', 'write_filesystem'), pure=False, output='Image'),
    CapabilitySpec('image_editing', 'Edit input images following prompt.', ('assets', 'prompt'), (), ('network', 'generate_file', 'write_filesystem'), pure=False, output='Image'),
    CapabilitySpec('video', 'Analyze video assets (mode=understand) or generate video (mode=generate).', ('mode', 'prompt'), ('assets',), pure=False, output='str | Video'),
]


class Registry:
    def __init__(self):
        self.specs = {s.name: s for s in SPECS}
        for entry in importlib.metadata.entry_points(group='aithon.capabilities'):
            self.register(entry.load()())

    def register(self, spec):
        if not isinstance(spec, CapabilitySpec) or spec.name in self.specs:
            raise ConfigError('Capability registration requires a unique CapabilitySpec')
        self.specs[spec.name] = spec


@dataclass
class CapabilityRequest:
    capability: str
    model: str
    params: dict
    revision: str = ''
    provider: str = ''


@dataclass
class CapabilityResult:
    value: object
    usage: dict = field(default_factory=dict)


class CapabilityProvider(Protocol):
    version: str
    def capabilities(self) -> set[str]: ...
    def invoke(self, request: CapabilityRequest, context) -> CapabilityResult: ...


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def encode(value):
    if isinstance(value, Asset):
        return {'$asset': type(value).__name__, 'path': str(value.path), 'mime': value.mime_type,
                'hash': value.content_hash()}
    if isinstance(value, Embeddings):
        return {'$embeddings': value.vectors, 'space': value.space}
    if value is None or type(value) in (str, int, bool, float):
        return value
    if type(value) in (list, tuple):
        return [encode(v) for v in value]
    if type(value) is dict and all(type(k) is str and not k.startswith('$') for k in value):
        return {k: encode(v) for k, v in value.items()}
    raise TypeError('Live mutable/custom objects are not cacheable')


def decode(value):
    if type(value) is list:
        return [decode(v) for v in value]
    if type(value) is dict:
        if '$asset' in value:
            asset = ASSET_TYPES[value['$asset']](Path(value['path']), value['mime'])
            if not asset.path.is_file() or asset.content_hash() != value['hash']:
                raise ValueError('Cached artifact missing or changed')
            return asset
        if '$embeddings' in value:
            return Embeddings(value['$embeddings'], value['space'])
        return {k: decode(v) for k, v in value.items()}
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root) / '.aithon'
        self._lock = threading.Lock()

    @contextmanager
    def connect(self, *, readonly=False):
        if readonly:
            connection = sqlite3.connect((self.root / 'runtime-v3.sqlite').as_uri() + '?mode=ro', uri=True)
            try:
                yield connection
            finally:
                connection.close()
            return
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / 'runtime-v3.sqlite', timeout=30)
        connection.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, created REAL, value TEXT)')
        connection.execute('CREATE TABLE IF NOT EXISTS indexes (name TEXT PRIMARY KEY, space TEXT, dimensions INTEGER, data TEXT)')
        connection.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, value TEXT)')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def cached(self, key):
        with self._lock, self.connect() as db:
            row = db.execute('SELECT created,value FROM cache WHERE key=?', (key,)).fetchone()
        if row and time.time() - row[0] < 86400:
            try:
                return True, decode(json.loads(row[1]))
            except (ValueError, KeyError, OSError):
                pass
        return False, None

    def put(self, key, value):
        try:
            encoded = canonical(encode(value))
        except (TypeError, ValueError):
            return
        with self._lock, self.connect() as db:
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (key, time.time(), encoded))

    def jobs(self):
        path = self.root / 'runtime-v3.sqlite'
        if not path.exists(): return []
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            return [{'id': row[0], **json.loads(row[1])} for row in connection.execute('SELECT id,value FROM jobs')]
        finally:
            connection.close()

    def job(self, identifier, record=None):
        with self._lock, self.connect() as db:
            if record is not None:
                db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', (identifier, canonical(record)))
            else:
                row = db.execute('SELECT value FROM jobs WHERE id=?', (identifier,)).fetchone()
                return json.loads(row[0]) if row else None


class LocalProvider:
    version = '1'

    def capabilities(self):
        return {'indexing', 'semantic_search', 'document_understanding'}

    def invoke(self, request, context):
        p = request.params
        if request.capability == 'document_understanding':
            if p.get('prompt'):
                raise InvocationError('Local documents support extraction; route prompted understanding to Gemini', retryable=True)
            chunks = []
            for asset in p['assets']:
                if asset.path.suffix.lower() not in ('.txt', '.md'):
                    raise InvocationError('Local document reader accepts .txt/.md; route PDFs to Gemini', retryable=True)
                lines = asset.path.read_text(encoding='utf-8').splitlines()
                for start in range(0, len(lines), 40):
                    chunks.append({'text': '\n'.join(lines[start:start + 40]),
                                   'source': f'{asset.path}:{start + 1}'})
            return CapabilityResult(chunks)
        if request.capability == 'indexing':
            embeddings, documents = p['embeddings'], p['documents']
            if not isinstance(embeddings, Embeddings) or len(embeddings.vectors) != len(documents):
                raise CapabilityError('One embedding per document is required')
            rows = []
            objects = []
            for document, vector in zip(documents, embeddings.vectors):
                record = {'text': document} if isinstance(document, str) else document
                if not isinstance(record, dict) or not isinstance(record.get('text'), str):
                    raise CapabilityError('Documents must be text or records containing text')
                if not isinstance(record.get('source', ''), str):
                    raise CapabilityError('Document source must be a string')
                rows.append({'text': record['text'], 'source': record.get('source', ''), 'vector': vector})
                objects.append(record.get('object', document))
            name = hashlib.sha256(canonical({'space': embeddings.space, 'rows': rows}).encode()).hexdigest()
            with context.store.connect() as db:
                db.execute('INSERT OR IGNORE INTO indexes VALUES (?,?,?,?)',
                           (name, embeddings.space, len(embeddings.vectors[0]), canonical(rows)))
            index = VectorIndex(context.store.root / 'runtime-v3.sqlite', name, embeddings.space, len(embeddings.vectors[0]), tuple(objects))
            return CapabilityResult(index)
        index, query = p['index'], p['query']
        if not isinstance(index, VectorIndex) or not isinstance(query, Embeddings) or len(query.vectors) != 1:
            raise CapabilityError('Search requires VectorIndex and one query embedding')
        if index.path.resolve() != (context.store.root / 'runtime-v3.sqlite').resolve():
            raise CapabilityError('Index belongs to a different project')
        if query.space != index.space or len(query.vectors[0]) != index.dimensions:
            raise CapabilityError('Embedding space mismatch; rebuild index explicitly before switching models')
        with context.store.connect(readonly=True) as db:
            row = db.execute('SELECT space,dimensions,data FROM indexes WHERE name=?', (index.name,)).fetchone()
        if not row or row[:2] != (index.space, index.dimensions):
            raise CapabilityError('Index missing or metadata mismatch')
        vector = query.vectors[0]
        def cosine(other):
            denominator = math.sqrt(sum(x*x for x in vector)) * math.sqrt(sum(x*x for x in other))
            return sum(a*b for a,b in zip(vector, other)) / denominator if denominator else 0.0
        hits = []
        originals = index._objects
        for i, record in enumerate(json.loads(row[2])):
            hit = {k: record[k] for k in ('text', 'source')}
            hit.update(score=cosine(record['vector']), index=i)
            hit['object'] = originals[i] if originals else {'text': hit['text'], 'source': hit['source']}
            hits.append(hit)
        return CapabilityResult(sorted(hits, key=lambda h: h['score'], reverse=True)[:p.get('limit', 5)])


class CapabilityRuntime:
    def __init__(self, config, *, trace=False, adapters=None):
        self.config = config
        self.registry = Registry()
        for profile in config.profiles.values():
            if profile.routes.keys() - self.registry.specs.keys():
                raise ConfigError('Profile routes reference an unregistered capability')
        self.store = Store(config.project_root)
        self.trace = trace
        self.events = []
        self.adapters = adapters or {}
        self._lock = threading.Lock()

    def require(self, profile, *permissions):
        missing = set(permissions) - set(profile.permissions)
        if missing:
            raise CapabilityPermissionError('Capability permission denied: ' + ', '.join(sorted(missing)))

    def suggest_missing_route(self, profile, capability):
        if (capability not in self.registry.specs or profile.routes.get(capability)
                or capability in ('reasoning', 'indexing', 'semantic_search')):
            return None
        from .config_hints import append_missing_route_example
        with self._lock:
            return append_missing_route_example(self.config, profile, capability)

    def routes(self, profile, capability, provider=None, mode=None):
        entries = profile.routes.get(capability, [])
        if not entries and capability in ('indexing', 'semantic_search'):
            entries = [{'provider': '$local', 'model': 'exact-cosine'}]
        if not entries and capability == 'reasoning':
            entries = [{'provider': '$legacy', 'model': profile.model}]
        if mode:
            entries = [r for r in entries if mode in r.get('modes', [mode])]
        if provider:
            entries = [r for r in entries if r['provider'] == provider]
        if not entries:
            hint = self.suggest_missing_route(profile, capability)
            suffix = f'; a commented setup example is in {hint}' if hint else ''
            raise ConfigError(f'No route for capability {capability!r} in profile {profile.name!r}{suffix}')
        return entries

    def adapter(self, route, profile):
        name = route['provider']
        key = (profile.name, name)
        if key not in self.adapters:
            if name == '$local':
                adapter = LocalProvider()
            else:
                from .capability_providers import make_adapter
                adapter = make_adapter(self.config, profile, name)
            self.adapters[key] = adapter
        return self.adapters[key]

    def resume_job(self, profile, identifier):
        self.require(profile, 'read_asset', 'network', 'generate_file', 'write_filesystem')
        record = self.store.job(identifier)
        if not record:
            raise CapabilityError('Unknown job operation ID')
        for route in self.routes(profile, 'video', mode='generate'):
            adapter = self.adapter(route, profile)
            if (getattr(adapter, 'api_base', None) == record.get('provider_url')
                    and (not record.get('provider') or route['provider'] == record['provider'])
                    and route['model'] == record.get('model') and callable(getattr(adapter, 'wait_video', None))):
                return adapter.wait_video(identifier, self)
        raise ConfigError('Job provider/model must remain configured in this profile to resume')

    def describe(self, profile):
        names = set(profile.routes) | {'indexing', 'semantic_search', 'document_understanding'}
        # The agent already uses its reasoning model. Advertising the same model
        # as a nested capability adds an unnecessary provider round trip.
        if all(route['model'] == profile.model for route in profile.routes.get('reasoning', [])):
            names.discard('reasoning')
        return [s.describe() for n, s in self.registry.specs.items() if n in names]

    def event(self, **event):
        with self._lock:
            from .stats import CURRENT_STATS
            stats = CURRENT_STATS.get()
            if stats is not None:
                stats.capability_events.append(event)
            self.events.append(event)
            if self.trace:
                print('aithon plan: ' + canonical(event), file=sys.stderr)

    def assets(self, params, profile):
        params = dict(params)
        if 'assets' in params:
            self.require(profile, 'read_asset')
            if not isinstance(params['assets'], list) or not params['assets']:
                raise CapabilityError('assets must be a nonempty list of paths or Asset objects')
            params['assets'] = [self.asset(a) for a in params['assets']]
        seen = set()
        def check(value):
            if id(value) in seen: return
            seen.add(id(value))
            if isinstance(value, Asset):
                self.require(profile, 'read_asset')
            elif type(value) in (list, tuple):
                for item in value: check(item)
            elif type(value) is dict:
                for item in value.values(): check(item)
        check(params)
        if 'inputs' in params:
            params['inputs'] = [self.asset(v) if isinstance(v, Asset) else v for v in params['inputs']]
        return params

    def asset(self, value):
        if isinstance(value, Asset):
            path = value.path
            return type(value)(path if path.is_absolute() else self.config.project_root / path, value.mime_type)
        if isinstance(value, (str, Path)):
            path = Path(value)
            return Document(path if path.is_absolute() else self.config.project_root / path)
        raise CapabilityError('Asset must be a path or Asset object')

    def invoke(self, profile, capability, params, *, provider=None, cache=None, cache_salt=None):
        if capability not in self.registry.specs:
            raise ConfigError(f'Unregistered capability: {capability}')
        spec = self.registry.specs[capability]
        spec.validate_inputs(params)
        self.require(profile, *spec.permissions)
        if capability == 'video' and params['mode'] == 'generate':
            self.require(profile, 'generate_file', 'write_filesystem')
        params = self.assets(params, profile)
        if capability == 'embedding':
            params['inputs'] = [v['text'] if type(v) is dict else v for v in params['inputs']]
        if capability in ('vision', 'image_editing') and any(not a.mime_type.startswith('image/') for a in params['assets']):
            raise CapabilityError('Image capability requires image assets')
        if capability == 'speech_to_text' and any(not a.mime_type.startswith(('audio/', 'video/')) for a in params['assets']):
            raise CapabilityError('Transcription requires audio or video assets')
        if 'limit' in params and (type(params['limit']) is not int or params['limit'] < 1):
            raise CapabilityError('limit must be a positive integer')
        if capability == 'video' and params['mode'] not in ('understand', 'generate'):
            raise CapabilityError('video mode must be understand or generate')
        if (capability == 'document_understanding' and provider is None and not params.get('prompt') and
                all(asset.path.suffix.lower() in ('.txt', '.md') for asset in params['assets'])):
            result = LocalProvider().invoke(CapabilityRequest(capability, 'local-text', params), self)
            self.validate_result(capability, params, result.value)
            return result.value
        entries = self.routes(profile, capability, provider, params.get("mode"))
        for number, route in enumerate(entries):
            adapter = self.adapter(route, profile)
            if not isinstance(adapter, LocalProvider): self.require(profile, 'network')
            if capability not in adapter.capabilities():
                raise ConfigError(f'Configured adapter does not support {capability}')
            key = None
            enabled = spec.cache if cache is None else cache
            # Cache read/write is an explicit filesystem capability, even on a hit.
            if enabled and {'read_asset', 'write_filesystem'} <= set(profile.permissions):
                try:
                    key = hashlib.sha256(canonical({'capability': capability, 'params': encode(params),
                        'route': route, 'endpoint': getattr(adapter, 'api_base', None),
                        'adapter': adapter.version, 'context': cache_salt}).encode()).hexdigest()
                except (TypeError, ValueError, RecursionError):
                    pass
            started = time.monotonic()
            if key:
                found, value = self.store.cached(key)
                if found:
                    self.validate_result(capability, params, value)
                    self.event(capability=capability, provider=route['provider'], model=route['model'], cache='hit', seconds=0, cost=None)
                    return value
            try:
                result = adapter.invoke(CapabilityRequest(capability, route['model'], params, route.get('revision', ''), route['provider']), self)
            except InvocationError as exc:
                self.event(capability=capability, provider=route['provider'], status='failed', accepted=exc.accepted)
                if exc.retryable and not exc.accepted and number + 1 < len(entries):
                    continue
                raise
            if not isinstance(result, CapabilityResult):
                raise CapabilityError('Adapter must return CapabilityResult')
            self.validate_result(capability, params, result.value)
            if key:
                self.store.put(key, result.value)
            usage = {k:v for k,v in result.usage.items() if type(v) in (int, float) and math.isfinite(v)} if isinstance(result.usage, dict) else {}
            self.event(capability=capability, provider=route['provider'], model=route['model'], cache='miss' if key else 'disabled',
                       seconds=time.monotonic()-started, usage=usage, cost=usage.get('cost'))
            return result.value

    @staticmethod
    def validate_result(capability, params, value):
        from .assets import Image, Audio, Video
        targets = {'embedding': Embeddings, 'indexing': VectorIndex, 'semantic_search': list,
                   'reranking': list, 'image_generation': Image, 'image_editing': Image,
                   'text_to_speech': Audio, 'speech_to_text': str, 'vision': str, 'reasoning': str}
        if capability == 'video':
            targets['video'] = Video if params['mode'] == 'generate' else str
        if capability in targets and not isinstance(value, targets[capability]):
            raise CapabilityError(f'{capability}: invalid adapter result type')
        if capability == 'embedding' and len(value.vectors) != len(params['inputs']):
            raise CapabilityError('Embedding result count does not match inputs')

    def run_plan(self, profile, steps, output, bridge, *, capability=None, provider=None):
        if not isinstance(steps, list) or not steps or len(steps) > 64:
            raise CapabilityError('Plan must contain 1..64 steps')
        by_id, dependencies = {}, {}
        target_capability = capability
        if not target_capability and provider and isinstance(output, dict):
            target_capability = next((s.get('capability') for s in steps if isinstance(s, dict) and s.get('id') == output.get('$ref')), None)
            if target_capability is None:
                raise CapabilityError('provider directive requires a target capability or step output')
        def refs(value):
            found = set()
            if isinstance(value, dict):
                if '$ref' in value:
                    if value.keys() - {'$ref', 'path'} or not isinstance(value['$ref'], str):
                        raise CapabilityError('Invalid step reference')
                    path = value.get('path', [])
                    if not isinstance(path, list) or any(type(k) not in (str, int) for k in path):
                        raise CapabilityError('Reference path must contain string keys or integer indices')
                    found.add(value['$ref'])
                elif '$asset' in value:
                    asset_ref = value['$asset']
                    if len(value) != 1 or not (
                        isinstance(asset_ref, str)
                        or (isinstance(asset_ref, dict) and len(asset_ref) == 1
                            and ('$binding' in asset_ref or '$handle' in asset_ref))
                        or (isinstance(asset_ref, dict) and '$ref' in asset_ref)
                    ):
                        raise CapabilityError('Asset reference must contain a path or a runtime/step reference')
                    self.require(profile, 'read_asset')
                    if isinstance(asset_ref, dict):
                        found |= refs(asset_ref)
                elif '$handle' in value or '$binding' in value:
                    if len(value) != 1:
                        raise CapabilityError('Invalid runtime reference')
                    if '$handle' in value:
                        item = bridge.dereference(value['$handle'])
                    else:
                        name = value['$binding']
                        if not isinstance(name, str) or not name.isidentifier():
                            raise CapabilityError('Binding references must be identifiers')
                        namespace = bridge.namespace()
                        if name not in namespace and name not in bridge.frame.f_globals:
                            raise CapabilityError('Binding reference does not exist')
                        item = namespace[name] if name in namespace else bridge.frame.f_globals[name]
                    self.assets({'value': item}, profile)
                else:
                    for item in value.values(): found |= refs(item)
            elif isinstance(value, list):
                for item in value: found |= refs(item)
            return found
        for step in steps:
            if not isinstance(step, dict) or step.keys() - {'id', 'capability', 'params', 'provider', 'cache'}:
                raise CapabilityError('Invalid plan step')
            name, cap = step.get('id'), step.get('capability')
            if not isinstance(name, str) or not name or name in by_id:
                raise CapabilityError('Plan step IDs must be nonempty and unique')
            if not isinstance(cap, str) or cap not in self.registry.specs:
                available = ', '.join(spec['name'] for spec in self.describe(profile))
                raise CapabilityError(f'Unknown capability {cap!r}; available in this profile: {available}')
            spec = self.registry.specs[cap]
            spec.validate_inputs(step.get('params'), references=True)
            self.require(profile, *spec.permissions)
            if 'assets' in step['params']:
                self.require(profile, 'read_asset')
            if cap == 'video':
                mode = step['params']['mode']
                if mode not in ('understand', 'generate'):
                    raise CapabilityError('Video mode must be a literal understand or generate')
                if mode == 'generate': self.require(profile, 'generate_file', 'write_filesystem')
            if 'cache' in step and type(step['cache']) is not bool:
                raise CapabilityError('cache must be boolean')
            selected = provider if (cap == target_capability) and provider else step.get('provider')
            for route in self.routes(profile, cap, selected, step["params"].get("mode")):
                adapter = self.adapter(route, profile)
                if not isinstance(adapter, LocalProvider): self.require(profile, 'network')
                if cap not in adapter.capabilities():
                    raise ConfigError(f'Route adapter does not support {cap}')
            by_id[name] = step
            dependencies[name] = refs(step['params'])
        prior = getattr(bridge, 'completed_steps', {})
        output_refs = refs(output)
        if not output_refs <= by_id.keys() | prior.keys():
            raise CapabilityError('Output references an unknown step')
        if capability and not any(s['capability'] == capability for s in steps):
            raise CapabilityError('Plan must include the directive capability')
        # Validate the entire DAG before any execution.
        checked = set(prior) - by_id.keys()
        while not by_id.keys() <= checked:
            ready = {n for n, deps in dependencies.items() if n not in checked and deps <= checked}
            if not ready:
                raise CapabilityError('Plan contains unknown dependencies or a cycle')
            checked |= ready
        results = {n: saved[1] for n, saved in prior.items() if n not in by_id}
        if not hasattr(bridge, 'completed_steps'):
            bridge.completed_steps = {}
        completed = bridge.completed_steps
        if not hasattr(bridge, 'uncertain_steps'):
            bridge.uncertain_steps = set()
        def resolve(value):
            if isinstance(value, dict):
                if '$ref' in value:
                    result = results[value['$ref']]
                    def project(item, path):
                        if not path: return item
                        if type(item) not in (dict, list, tuple):
                            raise CapabilityError('Reference path can only traverse dict/list values')
                        if path[0] == '*' and type(item) in (list, tuple):
                            return [project(v, path[1:]) for v in item]
                        return project(item[path[0]], path[1:])
                    return project(result, value.get('path', []))
                if '$asset' in value: return self.asset(resolve(value['$asset']))
                if '$handle' in value: return bridge.dereference(value['$handle'])
                if '$binding' in value: return bridge.namespace()[value['$binding']] if value['$binding'] in bridge.namespace() else bridge.frame.f_globals[value['$binding']]
                return {k: resolve(v) for k,v in value.items()}
            if isinstance(value, list): return [resolve(v) for v in value]
            return value
        def execute(name):
            step = by_id[name]
            fingerprint = canonical(step)
            saved = completed.get(name)
            if saved:
                if saved[0] != fingerprint:
                    raise CapabilityError('Completed step ID cannot be reused with changed inputs; use a new ID')
                return saved[1]
            uncertain = getattr(bridge, 'uncertain_steps', set())
            submission = canonical({k:v for k,v in step.items() if k != 'id'})
            if submission in uncertain:
                raise CapabilityError('Previous submission acceptance is unknown; inspect/resume the provider job instead of resubmitting')
            try:
                value = self.invoke(profile, step['capability'], resolve(step['params']),
                    provider=provider if provider and (step['capability'] == target_capability) else step.get('provider'), cache=step.get('cache'),
                    cache_salt={'hints': getattr(getattr(bridge, 'ai_request', None), 'prompts', ()),
                            'output_type': getattr(getattr(bridge, 'ai_request', None), 'output_type', None)})
            except InvocationError as exc:
                if exc.accepted:
                    uncertain.add(submission)
                    bridge.uncertain_steps = uncertain
                raise
            completed[name] = (fingerprint, value)
            return value
        while not by_id.keys() <= results.keys():
            ready = [n for n, deps in dependencies.items() if n not in results and deps <= results.keys()]
            pure = [n for n in ready if self.registry.specs[by_id[n]['capability']].pure]
            if len(pure) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(pure))) as pool:
                    pending = {name: pool.submit(copy_context().run, execute, name) for name in pure}
                    errors = []
                    for name, future in pending.items():
                        try: results[name] = future.result()
                        except Exception as exc: errors.append(exc)
                    if errors: raise errors[0]
            else:
                name = ready[0]
                results[name] = execute(name)
        return resolve(output)
