# Capability runtime

Aiython exposes capabilities to the model at each suspended statement. The model chooses a capability and its inputs; ordinary user scripts can pass paths or live objects without importing `aiython`. The runtime validates the plan, checks permissions, resolves references, and executes each step against the configured route. Python still controls the outer statement and loop order.

## Routes

Use LiteLLM model IDs in `aiython.toml` version 3:

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

`aiython setup` creates the main model configuration and can add any supported remote capability route through **Configure capabilities**. The menu lets you choose a model and save a route-specific API key; it can configure several capabilities in one session. Search matches words in model IDs and names. OpenRouter suggestions are filtered by advertised input/output modalities; image suggestions require text output too because the pinned LiteLLM adapter uses chat completions rather than OpenRouter's dedicated image endpoint. Gemini reasoning and embedding suggestions use advertised methods. Other catalogs may lack capability metadata, and exact IDs remain enterable. For noninteractive use, run `aiython setup --capability speech_to_text --model openai/whisper-1 --non-interactive`. Add `--profile NAME` for a profile-specific route. Video takes `--understand-model` and `--generate-model`; either can be changed later without replacing the other. A capability route includes its own `api_key_env`, even when it uses the same provider as the main model, so project-saved keys reach the correct SDK call. `--check` tests the reasoning model only; use a capability program to verify media or other remote routes without an unexpected generation job.

If an AI statement asks for a capability with no route, Aiython shows the appropriate `aiython setup --capability ...` command and appends a commented example to the project config. It never overwrites a version 1/2 config or inserts a live route without a real model ID.

OpenRouter image-only models use its dedicated image endpoint, which the pinned LiteLLM image adapter does not call; setup offers only image models that also support the chat endpoint. OpenRouter video generation uses the official OpenRouter Python SDK and its dedicated video models catalog.

OpenRouter video jobs require temporary retention and cannot run when the OpenRouter account enforces Zero Data Retention (ZDR). If a submission is blocked by that policy, Aiython reports the account setting directly and stops before creating a job. Change the setting at [OpenRouter privacy settings](https://openrouter.ai/settings/privacy) if video generation is appropriate for the account, or configure a direct video provider instead.

## Inputs and results

- `reasoning(prompt, context?)` returns text; source fields in context are retained, while live `object` fields are omitted from provider input.
- `vision(assets, prompt)` and `document_understanding(assets, prompt?)` accept project file paths. Local `.txt` and `.md` extraction can run without a model when no prompt is supplied. Document extraction reports source paths and PDF page numbers when the provider returns them.
- `embedding(inputs, dimensions?, task_type?)` returns vectors plus an embedding-space identifier. `reranking(query, documents, limit?)` returns the original records in provider rank order, preserving any live objects.
- `speech_to_text(assets, prompt?)` returns a transcript. `text_to_speech(text, voice?)`, `image_generation(prompt)`, and `image_editing(assets, prompt)` return project-local artifacts.
- `video(mode="understand", assets, prompt)` returns text. `video(mode="generate", prompt, assets?)` creates a persistent job and eventually returns a video artifact. `aiython jobs list` and `aiython jobs resume OPERATION_ID` resume polling/download without another submission.
- `indexing(embeddings, documents)` and `semantic_search(index, query, limit?)` run locally in SQLite and keep live object references within the process.

Remote calls use the [LiteLLM Python SDK](https://docs.litellm.ai/docs/), except OpenRouter video generation, which uses the [official OpenRouter Python SDK](https://openrouter.ai/docs/client-sdks/python/overview) because the pinned LiteLLM version cannot submit OpenRouter video jobs. Large Gemini media uses the SDK file-upload path and waits for processing before referencing the file. Capability calls do not retry requests with ambiguous acceptance. Video jobs have a separate 600-second polling deadline and remain resumable afterward.

The plan tool accepts a DAG of steps with IDs, dependencies such as `{"$ref":"step-id"}`, and one output reference. The runtime validates the full plan and permissions before executing any step. Independent pure steps may run concurrently. A failed plan does not roll back completed effects; the model can reuse completed step IDs to avoid resubmission.

Project caches, vector indexes, and job manifests live in `.aiython/runtime-v3.sqlite`. The previous database is left untouched. Read-only extraction and analysis can cache by content; generated media is fresh unless a plan opts into caching. Caches do not serialize Python live objects. `--trace-plan` shows actual step events; `--explain` is static and does not run capabilities.

See [sample programs and assets](../examples/capabilities/README.md).
