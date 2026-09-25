# Contributing to Aiython

Thanks for helping improve Aiython. Bug reports, documentation fixes, examples,
and focused code changes are welcome.

## Before you change code

- Search existing issues and pull requests for related work. For a larger change,
  open an issue first so the behavior and scope can be discussed.
- Keep Python in charge of execution order and side effects. New behavior should
  apply generally, rather than depend on a particular prompt or example.
- Keep code, examples, documentation, and user-facing messages in English.

## Develop locally

Install CPython 3.11+ and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run aiython --explain examples/recipes/03_loop.py
uv run python -m unittest discover -s tests -q
```

The default test suite mocks provider calls and requires no API key. Use
`--explain` to inspect an example without executing it. If a change affects a
provider or capability, add an offline contract test for its request, response,
and error behavior; live API checks are optional and may incur charges.

## Submit a change

Keep a pull request focused and describe the user-visible behavior, why it
changed, and how you verified it. Update the README or relevant guide when an
interface changes. Include a small reproducible program for runtime bugs and
check that Python statements and side effects still run in their normal order.

For a bug report, include the Aiython command, expected and actual behavior,
Python version, and a minimal source file. `--stats` output can help locate
latency or provider errors. Remove API keys, credentials, and private input
before sharing logs or source.

By contributing, you agree that your contribution is licensed under the
[MIT license](LICENSE).
