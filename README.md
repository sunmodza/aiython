# Aithon

Aithon runs Python 3.14+ and asks a model to complete a suspended statement when the source contains an AI expression or an unhandled runtime exception reaches a recovery checkpoint. Python controls statement order, loops, and ordinary side effects. A model invocation operates on the current live frame; it does not replace the surrounding program.

```bash
uv sync
uv run aithon setup
uv run aithon --stats main.py
```

Ordinary Python runs without an API key and does not import LiteLLM or contact a provider. `aithon setup` creates or edits the nearest project `aithon.toml`. In a terminal, it shows the current provider, model and credential status, then lets you search provider model IDs or update the provider or API key. You can always enter an exact model ID if the catalog is unavailable. A key entered interactively is saved in a gitignored project env file with mode `0600`; keys are never written to TOML. Setup can test tool calling with one small, billable provider request when you choose to do so.

For noninteractive setup:

```bash
uv run aithon setup --provider openrouter --model vendor/model --non-interactive
uv run aithon setup --model openrouter/vendor/another-model --non-interactive
uv run aithon setup --check --non-interactive
```

The second command updates only the model and preserves comments, capability routes and profiles. `--provider` changes the service, `--set-key` securely prompts to replace a saved key in a terminal, and `--path` selects an exact config file. Noninteractive setup never asks for or accepts a key on the command line; set its environment variable instead. `--check` sends one tool-call probe with a 25-second timeout and does not save a failed change. Provider model catalogs are used only for the interactive search, with a five-second timeout and manual entry fallback.

The model must support tool calling. Aithon uses the [LiteLLM Python SDK](https://docs.litellm.ai/docs/) in process, with no proxy service. You can choose any model and provider supported by the installed LiteLLM version.

## Write a program

```python
items = []
for number in range(3):
    item: int = choose the next item from number and items
    items.append(item)
print(items)
```

The `for` loop runs in Python. The AI expression is invoked once per iteration with the current frame. AI tools can evaluate expressions, execute Python against the frame, inspect objects, and use configured capabilities. They retain object identity. Python's own calls and effects occur in their normal order. Because the model can use live `eval` and `exec`, it can also inspect or invoke anything allowed by the process; Aithon is not a sandbox.

Valid Python remains Python. A name such as `result = unknown_name` follows the runtime recovery path if it raises `NameError`; a natural-language expression that does not parse as Python is suspended as an AI block. Use `--explain` to see the blocks and checkpoints without running the script or contacting a model. Ambiguous source can be grouped into a larger AI block, so inspect it when boundaries matter.

```bash
uv run aithon --explain main.py
uv run aithon --stats main.py
uv run aithon --trace-plan main.py
uv run aithon config show --script main.py
uv run aithon jobs list --script main.py
uv run aithon jobs resume OPERATION_ID --script main.py
```

Aithon options go before the script; later arguments are passed to the program. `--profile` selects a default profile and `--force-profile` overrides source directives.

## Configure a project

`aithon.toml` version 3 has one top-level model for the common case. Model IDs use LiteLLM syntax directly. Use `[capabilities]` only for work that needs a different model. A named profile can override either setting.

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

A route may also be `{ model = "...", api_base = "https://...", api_key_env = "KEY_NAME" }`, or a list of routes for fallback after a definite 429 rejection. For a custom OpenAI-compatible endpoint, set top-level `api_base` and `api_key_env` and use a LiteLLM model ID such as `openai/local-model`. `env_file` points to a literal project-local dotenv file; process environment values take precedence. Keys never appear in `config show` or in provider errors.

Aithon searches upward from the script for the nearest `aithon.toml`, stopping at the Git root. `--config` chooses an exact path. Version 1 and 2 files are rejected with migration guidance and are never overwritten. Move the old reasoning model to top-level `model`, convert other routes to `[capabilities]`, and replace direct keys with `api_key_env` plus a private env file. Runtime state starts fresh in `.aithon/runtime-v3.sqlite`; older databases are left untouched.
Run `aithon setup` anywhere in the project; it finds and edits the nearest configuration, including the one inherited by bundled examples. Use `--path` to target a different project file. A nested `aithon.toml` is a separate configuration and does not inherit the root credential file.

The default AI invocation deadline is 120 seconds, including model turns and runtime tool work. Remaining time is passed to LiteLLM. A request with ambiguous acceptance is never retried automatically. Video generation is a separate resumable job with its own polling deadline. `--stats` reports progress immediately before provider calls, so a delay before that line is local runtime work; a delay after it includes SDK and network time.

## Model tool protocol

The model returns tool calls, ending with exactly one tagged `finish` outcome. For example:

```text
finish(outcome={"kind":"literal","value":"The answer is 42"})
finish(outcome={"kind":"expression","code":"x + y"})
finish(outcome={"kind":"reference","id":"answer"})
finish(outcome={"kind":"handle","id":"object-1"})
finish(outcome={"kind":"none"})
finish(outcome={"kind":"error","reason":"No route is configured"})
```

`evaluate(code="x + y", result_id="answer")` followed by a reference outcome can return a live object in one response. `execute` performs assignments or mutations. A required expression cannot finish with `none`; a standalone side-effect statement can. `recover(action="complete", outcome=...)` returns a replacement, while `retry` and `reraise` have no outcome.

Every batch is validated before any tool executes. A terminal must be last; malformed batches cause no effects. Two invalid batches or repeated empty responses stop with a clear error. Completed side effects are never rolled back or silently replayed. Output annotations are checked against the same runtime type contracts used for ordinary Python. See [type safety](docs/type-safety.md).

## Capabilities and examples

Configured capabilities include reasoning, vision, document understanding, embeddings, reranking, speech transcription and synthesis, image generation and editing, and video understanding and generation. Local text extraction, vector indexing, and semantic search use project storage. When a required route is missing, Aithon appends a commented version 3 configuration example to the project TOML and stops. See the [capability guide](docs/capabilities.md), [recipe examples](examples/recipes/README.md), and [sample assets](examples/capabilities/README.md).

`--stats` records per-invocation calls, tool counts, context bytes, provider and runtime timing, reported token usage, and redacted errors. Request bytes are an estimate of serialized model messages and tools, not exact wire bytes. See the [performance evaluation guide](docs/performance-evaluation.md) for the correctness and latency gate.

## Verify

```bash
uv run python -m unittest discover -s tests -q
```

The default suite mocks provider calls and does not require credentials. Live capability tests are opt-in through `AITHON_LIVE_CAPABILITY_CONFIG` and `AITHON_LIVE_CAPABILITY_CASES`.
