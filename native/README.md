# Native CPython experiment

`cpython.lock.toml` pins the exact CPython release commits used by the
compatibility work. `scripts/build_cpython_baseline.py` builds and checks an
unmodified interpreter before any VM changes are applied. The current patches
have been built and tested on Linux.

The 3.11.16–3.14.4 patches are **experimental VM hooks**, not Aiython
interpreters.
When `sys._aiython_before_store` exists, it calls that function with
`(frame, name, value)` before local, global, namespace, or closure stores.
The callback may raise, and Python's normal exception handlers receive that
exception. It skips empty stack references used by `STORE_FAST_MAYBE_NULL`,
and a thread-local guard stops recursive callbacks. With the attribute absent,
the interpreter follows CPython's normal store behavior.

When `sys._aiython_before_mutation` exists, it calls that function with
`(frame, kind, owner, key, value)` before attribute (`kind == "attr"`) and item
(`kind == "item"`) assignment, including slice assignment. Specialized
attribute, dict, and list store opcodes return to the generic opcode while
this hook is active, so enabling it after specialization still works.
The optimized string `+=` instruction returns to ordinary bytecode execution
while the store hook is active.

`sys._aiython_on_call(frame)` runs at the first `RESUME` of a function or
generator. `sys._aiython_on_return(frame, value)` runs at a return opcode.
These callbacks can raise. A return callback error raised at the return opcode
is outside the function's source-level `try` range in some cases, so an inner
`except` will not necessarily catch it. This remains an unresolved semantic
gap.

On Python 3.12–3.14, the standalone bridge also installs a scoped
`sys.monitoring.PY_YIELD` callback. It validates values from `yield`,
`yield from`, and async generators before they reach the caller. A validation
error follows CPython's normal exception path, including generator `finally`
blocks. Python 3.11 has no `sys.monitoring`, so typed yields there continue
through the managed bridge's boundary compiler.

The automated builder verifies the unmodified CPython baseline, applies the
matching patch, builds again, runs the VM hook smoke test, and runs eight
CPython test modules. It also runs `native/tests/typed_bridge.py`, which
checks type enforcement on original CPython code objects. Replace `VERSION`
with one of the exact releases in
`cpython.lock.toml`:

```sh
VERSION=3.13.15
uv run --locked --python "$VERSION" python scripts/build_cpython_hooks.py "$VERSION"
```

To apply a patch manually to a clean checkout:

```sh
VERSION=3.13.15
uv run --locked --python "$VERSION" python scripts/build_cpython_baseline.py "$VERSION" \
  --source-dir "/path/to/cpython-$VERSION"
git -C "/path/to/cpython-$VERSION" apply \
  "$PWD/native/patches/cpython-$VERSION-hooks.patch"
make -C "/path/to/cpython-$VERSION" regen-generated-cases
make -C "/path/to/cpython-$VERSION" -j4
"/path/to/cpython-$VERSION/python" -I native/tests/store_hook.py
```

For 3.12, use `make regen-cases` instead of `regen-generated-cases`. For 3.11,
skip the regeneration command: its interpreter is maintained directly in
`Python/ceval.c`.

`aiython.native_bridge.NativeTypeBridge.prepare_unit` is the normal compiler
entry point called by `Runtime.compile_source`. It selects the exact CPython
code object for ordinary source without imports, annotations, AI directives,
blocks, or configured recovery. For annotated source and configured recovery,
it delegates to Aiython's boundary compiler; that path checks yielded values
and can resume failed statements. A bridge bound to a `Runtime` offers the
same behavior through `compile_source`. This routing works on stock CPython
and the four pinned patched interpreters.

The standalone experimental `NativeTypeBridge.compile_source` and `installed`
API parses annotations, compiles the original source, and validates annotated
parameters, generator returns, local/global/nonlocal assignments, `Final`
rebinding, and class attributes through the VM hooks. On Python 3.12–3.14 it
also validates yielded values through `sys.monitoring`. It does not yet check
yielded values on Python 3.11, deletions, in-place mutation through method
calls, generic bindings, or provide AI recovery through VM callbacks. These
results do not establish full Python compatibility.
