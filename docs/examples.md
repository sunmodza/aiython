# Examples

The [recipe directory](https://github.com/sunmodza/aiython/tree/main/examples/recipes) contains small programs that isolate one behavior each. You can inspect any recipe without a model or API key:

```bash
uv run aiython --explain examples/recipes/01_typed_result.py
```

Run `aiython setup` at the repository root before executing a recipe. Then, for example:

```bash
uv run aiython --stats examples/recipes/01_typed_result.py
```

| Recipe | What it shows |
| --- | --- |
| [Typed result](https://github.com/sunmodza/aiython/blob/main/examples/recipes/01_typed_result.py) | A `TypedDict` and `Literal` constrain an AI result. |
| [Update state](https://github.com/sunmodza/aiython/blob/main/examples/recipes/02_update_state.py) | An AI statement changes existing objects. |
| [Python loop](https://github.com/sunmodza/aiython/blob/main/examples/recipes/03_loop.py) | Python routes tickets in a loop; AI summarizes once afterward. |
| [Recovery](https://github.com/sunmodza/aiython/blob/main/examples/recipes/04_recovery.py) | A valid Python statement fails and reaches a recovery checkpoint. |
| [Existing object](https://github.com/sunmodza/aiython/blob/main/examples/recipes/05_existing_object.py) | AI selects a live dataclass instance without copying it. |
| [Python first](https://github.com/sunmodza/aiython/blob/main/examples/recipes/06_python_first.py) | Python computes Fibonacci; AI explains the result afterward. |

AI-generated values can vary between runs. The recovery recipe is valid Python, so its AI work starts only when the `KeyError` occurs. The [recipe guide](https://github.com/sunmodza/aiython/blob/main/examples/recipes/README.md) has details about project configuration.

## Documents and media

The [capability examples](https://github.com/sunmodza/aiython/tree/main/examples/capabilities) include sample files and programs for documents, product images, audio, and video. They require the matching routes in your configuration; see the [capability guide](capabilities.md) before running them. You can inspect their AI boundaries without credentials:

```bash
uv run aiython --explain examples/capabilities/documents.py
uv run aiython --explain examples/capabilities/products.py
```

The full [capability example guide](https://github.com/sunmodza/aiython/blob/main/examples/capabilities/README.md) explains the bundled assets and setup. Executing these programs can call paid providers, and the media example requests several generation capabilities.
