# Capability runtime

Aithon exposes capabilities to the model at each suspended statement. The model chooses a capability and its inputs; ordinary user scripts can pass paths or live objects without importing `aithon`. The runtime validates the plan, checks permissions, resolves references, and executes each step against the configured route. Python still controls the outer statement and loop order.

## Routes

Use LiteLLM model IDs in `aithon.toml` version 3:

```toml
version = 3
model = "openrouter/vendor/tool-model"
api_key_env = "OPENROUTER_API_KEY"

[capabilities]
vision = "gemini/gemini-2.5-flash"
document_understanding = "gemini/gemini-2.5-flash"
embedding = "openai/text-embedding-3-small"
reranking = "cohere/rerank-v3.5"
speech_to_text = "openai/whisper-1"
text_to_speech = "openai/tts-1"
image_generation = "openai/gpt-image-1"
image_editing = "openai/gpt-image-1"
video = { understand = "gemini/gemini-2.5-flash", generate = "gemini/veo-3.0-generate-preview" }
```

Replace the sample models with ones available to your account and set their credentials in environment variables. Each route can be an inline table with `model`, `api_base`, `api_key_env`, and optional `revision`; a list of routes enables fallback only after a definite 429 rejection. Use `[profiles.NAME.capabilities]` for profile-specific overrides. The top-level model is the reasoning route unless overridden.

`aithon setup` creates a minimal config. If an AI statement asks for a capability with no route, Aithon appends a commented example under the appropriate `[capabilities]` table. It never overwrites a version 1/2 config or inserts a live route without a real model ID.

## Inputs and results

- `reasoning(prompt, context?)` returns text; source fields in context are retained, while live `object` fields are omitted from provider input.
- `vision(assets, prompt)` and `document_understanding(assets, prompt?)` accept project file paths. Local `.txt` and `.md` extraction can run without a model when no prompt is supplied. Document extraction reports source paths and PDF page numbers when the provider returns them.
- `embedding(inputs, dimensions?, task_type?)` returns vectors plus an embedding-space identifier. `reranking(query, documents, limit?)` returns the original records in provider rank order, preserving any live objects.
- `speech_to_text(assets, prompt?)` returns a transcript. `text_to_speech(text, voice?)`, `image_generation(prompt)`, and `image_editing(assets, prompt)` return project-local artifacts.
- `video(mode="understand", assets, prompt)` returns text. `video(mode="generate", prompt, assets?)` creates a persistent job and eventually returns a video artifact. `aithon jobs list` and `aithon jobs resume OPERATION_ID` resume polling/download without another submission.
- `indexing(embeddings, documents)` and `semantic_search(index, query, limit?)` run locally in SQLite and keep live object references within the process.

All remote calls use the [LiteLLM Python SDK](https://docs.litellm.ai/docs/) directly. Large Gemini media uses the SDK file-upload path and waits for processing before referencing the file. Capability calls do not retry requests with ambiguous acceptance. Video jobs have a separate 600-second polling deadline and remain resumable afterward.

The plan tool accepts a DAG of steps with IDs, dependencies such as `{"$ref":"step-id"}`, and one output reference. The runtime validates the full plan and permissions before executing any step. Independent pure steps may run concurrently. A failed plan does not roll back completed effects; the model can reuse completed step IDs to avoid resubmission.

Project caches, vector indexes, and job manifests live in `.aithon/runtime-v3.sqlite`. The previous database is left untouched. Read-only extraction and analysis can cache by content; generated media is fresh unless a plan opts into caching. Caches do not serialize Python live objects. `--trace-plan` shows actual step events; `--explain` is static and does not run capabilities.

See [sample programs and assets](../examples/capabilities/README.md).
