# Examples

Each recipe shows one Aiython behavior. Copy a program into a `.py` file, run `aiython setup`, then inspect and run it:

```bash
aiython --explain example.py
aiython --stats example.py
```

`--explain` needs no model or API key. Running the program can call your configured provider and incur charges. If you installed Aiython in a uv project, prefix these commands with `uv run`.

## Typed result

A `TypedDict` and `Literal` constrain an AI result. Aiython checks the value before assigning it to `draft`.

```python
--8<-- "examples/recipes/01_typed_result.py"
```

## Update state

An AI statement changes existing Python objects.

```python
--8<-- "examples/recipes/02_update_state.py"
```

## Python loop

Python routes tickets in a loop; AI summarizes once afterward.

```python
--8<-- "examples/recipes/03_loop.py"
```

## Recovery

A valid Python statement fails and reaches a recovery checkpoint. Its AI work begins only when the `KeyError` occurs.

```python
--8<-- "examples/recipes/04_recovery.py"
```

## Existing object

AI selects a live dataclass instance. The identity check confirms that the returned object was not copied.

```python
--8<-- "examples/recipes/05_existing_object.py"
```

## Python first

Python computes Fibonacci; AI explains the result afterward.

```python
--8<-- "examples/recipes/06_python_first.py"
```

AI-generated values can vary between runs. Read [how the runtime works](runtime.md) for the execution boundary and [type safety](type-safety.md) for result checks.

## Documents and media

Capability programs need matching routes in your configuration. See the [capability guide](capabilities.md) before running them. Save each script and its assets in the same directory.

### Document understanding

Save this program as `documents.py`:

```python
--8<-- "examples/capabilities/documents.py"
```

Save this sample policy as `policy.md` beside it:

```markdown
--8<-- "examples/capabilities/policy.md"
```

### Image matching

Save this program as `products.py`:

```python
--8<-- "examples/capabilities/products.py"
```

Download [red-shoe.jpg](assets/examples/red-shoe.jpg), [blue-boot.jpg](assets/examples/blue-boot.jpg), and [shoe.jpg](assets/examples/shoe.jpg) beside it. These are fictional, unbranded product images.

### Media and video

The larger media program combines speech, vision, image, and video capabilities:

```python
--8<-- "examples/capabilities/media.py"
```

For this program, also download [meeting.mp3](assets/examples/meeting.mp3), [clip.mp4](assets/examples/clip.mp4), and the three product images above. The [sample transcript](assets/examples/meeting-transcript.md) lets you check the speech result. The smaller video-only program shows a resumable generation job:

```python
--8<-- "examples/capabilities/video_generation.py"
```

Media generation can take longer and incur charges. Run `aiython --explain PATH` first to inspect the boundary without making a provider call.
