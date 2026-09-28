# Contributing to Aiython

Bug reports, documentation fixes, examples, and focused code changes are welcome.

## Start with an issue

Search existing issues and pull requests first. For a larger change, open an
issue to discuss the behavior before writing code. A bug report should include
the Aiython command, CPython version, expected and actual behavior, and a small
reproducible script. Remove credentials and private data from logs and examples.

## Make a change

Fork the repository and create a branch from `main`, such as `fix/error-message`
or `docs/getting-started`. Install CPython 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked aiython --explain examples/recipes/03_loop.py
uv run --locked python -m unittest discover -s tests -q
```

The local tests mock provider calls and need no API key. Use `--explain` to
inspect AI boundaries without running the program. When changing a capability
or provider, add an offline test for its request, response, and error behavior.

For documentation changes, run:

```bash
uv run --locked zensical build --clean --strict
uv run --locked python scripts/check_docs_site.py
```

## Open a pull request

Push your branch and open a pull request against `main`. Explain what changed,
why, and how you tested it. Update a relevant guide or example when behavior
changes. Keep Python in charge of execution order and side effects, and keep
code and documentation in English.

`main` accepts changes through pull requests. For documentation-only changes,
CI checks the docs and skips the package, Python, and patched CPython tests.
Code and workflow changes run the full suite on Python 3.11–3.14. The release
workflow publishes PyPI and the documentation site from version tags.

Contributions are licensed under the [MIT license](LICENSE).
