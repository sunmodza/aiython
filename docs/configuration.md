# Configuration and setup

Aiython uses the nearest version 3 `aiython.toml` found by searching upward from the script, stopping at the Git root. Run `aiython setup` anywhere in that project to create or edit it. A nested `aiython.toml` starts a separate project configuration and does not inherit the parent's credential file. If you installed Aiython with `uv add aiython`, prefix the commands below with `uv run`.

## Interactive setup

```bash
aiython setup
```

The menu shows the current provider, model, and credential status. You can change the main model, search provider model IDs, add capability routes, or update a key without starting over. Exact LiteLLM model IDs also work when a catalog is unavailable. The main model must support tool calling.

If you enter a key, Aiython stores it in a gitignored `.aiython/credentials.env` with file mode `0600`. It writes only the environment variable name to TOML. Process environment values take precedence over the project file. `config show` and provider errors do not print keys.

The optional setup check makes one small, billable reasoning request with a 25-second timeout. It does not test media routes or save a failed change.

## Noninteractive setup

```bash
aiython setup --provider openrouter --model vendor/tool-model --non-interactive
aiython setup --model openrouter/vendor/another-model --non-interactive
aiython setup --check --non-interactive
```

Changing only `--model` preserves existing comments, capability routes, and profiles. `--provider` changes the service; `--path` targets an exact config file. Noninteractive setup does not accept an API key on the command line. Set the named environment variable instead, or use `--set-key` for a secure terminal prompt.

## Configure a capability

The setup menu's **Configure capabilities** option adds routes for vision, documents, embeddings, reranking, speech, images, or video. It can add several routes in one session. Search matches words in model names and IDs rather than scattered characters. OpenRouter suggestions use advertised modalities; Gemini reasoning and embedding suggestions use advertised methods. Other catalogs may lack comparable metadata. Exact IDs can always be entered manually.

For scripts or CI, select a capability explicitly:

```bash
aiython setup --capability speech_to_text --model openai/whisper-1 --non-interactive
aiython setup --capability video --understand-model openrouter/google/gemini-2.5-flash --generate-model openrouter/minimax/hailuo-3 --non-interactive
aiython setup --capability image_generation --profile fast --model openai/image-model --non-interactive
```

These IDs illustrate the syntax; availability depends on your provider account. Each route may use a separate `api_key_env` and provider. The video understanding and generation routes can be changed independently. `--check` tests only the reasoning model because a media probe could create a billable artifact or job; run a [capability example](examples.md#documents-and-media) to test a media route intentionally.

When a required route is missing, Aiython reports the matching setup command and appends an inert commented example to the TOML. It does not invent a model ID or activate a route on your behalf.

## Version 3 TOML

The common case needs one top-level model. Use `[capabilities]` only when another task needs its own model, and `[profiles.NAME]` for an override:

```toml
version = 3
model = "openrouter/vendor/tool-model"
api_key_env = "OPENROUTER_API_KEY"
timeout = 120

[capabilities]
embedding = "openai/text-embedding-3-small"
vision = "gemini/gemini-2.5-flash"
video = { understand = "gemini/gemini-2.5-flash", generate = "gemini/veo-3.0-generate-preview" }

[profiles.fast]
model = "openai/gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
```

Model IDs use LiteLLM syntax. A route can also be `{ model = "...", api_base = "https://...", api_key_env = "KEY_NAME" }`. For a custom OpenAI-compatible endpoint, set `api_base` and `api_key_env` and use a LiteLLM ID such as `openai/local-model`. A list of routes permits fallback after a definite 429 rejection; Aiython does not retry a request whose acceptance is uncertain. `env_file` points to a literal file inside the project.

Run `aiython config show --script PATH` to inspect resolved settings. `--config` chooses an exact file. `--profile` selects a default named profile; `--force-profile` overrides source directives. CLI options for script execution go before the script path, while later arguments go to the program.

Run an importable module or package with `aiython -m package.module` or `aiython -m package`. Put Aiython options before `-m`; arguments after the module name go to that module. Packages need a `__main__.py`, as with `python -m`. Source files get Aiython's source transformation; modules available only as bytecode run as Python bytecode. Resolving a dotted name for `--explain -m` imports its parent package.

Version 1 and 2 configs are rejected with migration guidance and never overwritten. Move the old reasoning model to top-level `model`, put other routes under `[capabilities]`, and replace direct keys with `api_key_env` plus a private env file. Runtime state starts fresh in `.aiython/runtime-v3.sqlite`; older databases remain untouched.

## Deadlines and jobs

The default AI invocation deadline is 120 seconds, covering model turns and runtime tool work. Aiython passes the remaining time to the SDK and reports progress before provider requests. `--stats` separates context construction, provider, and runtime time; `--trace-plan` shows capability routes and cache behavior.

Video generation is a separate resumable job. Aiython saves the operation ID before polling so `aiython jobs resume OPERATION_ID --script PATH` can continue without resubmitting. OpenRouter video generation uses its Python SDK for submit, status, and content. Accounts enforcing Zero Data Retention cannot submit video jobs because the provider must retain generated content temporarily; Aiython reports that policy block explicitly.
