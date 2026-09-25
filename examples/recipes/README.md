# Small Aiython examples

Each file isolates one behavior and can be inspected without a model:

```bash
uv run aiython --explain examples/recipes/01_typed_result.py
```

To run an example with a tool-calling model, run `aiython setup` at the repository
root. These scripts inherit that project configuration. The local
`aiython.toml.example` is a template only; copying it here creates a separate
configuration with separate credential resolution. Then run, for example:

```bash
uv run aiython --stats examples/recipes/01_typed_result.py
```

| File | Behavior |
| --- | --- |
| `01_typed_result.py` | Return a `TypedDict` with a checked `Literal` field. |
| `02_update_state.py` | Mutate existing objects through a standalone AI statement. |
| `03_loop.py` | Run one AI statement for each Python loop iteration. |
| `04_recovery.py` | Recover from an intentional `KeyError` at a checkpoint. |
| `05_existing_object.py` | Select an existing dataclass instance without copying it. |
| `06_python_first.py` | Compute Fibonacci in Python, then ask AI for an explanation. |

AI-generated values can vary between runs. `--explain` shows syntax blocks and
checkpoints without executing the program or contacting a provider. The recovery
example is valid Python, so its AI work begins only when the `KeyError` occurs.
