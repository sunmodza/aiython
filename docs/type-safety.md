# Declared types are runtime contracts

Aiython checks explicit annotations throughout project code, including ordinary Python. It sends the same contract to the AI as an output schema. No special base class or framework is required.

```python
from typing import Annotated, Literal, TypedDict

class TicketAnalysis(TypedDict):
    category: Literal["availability", "performance", "security", "billing", "feature_request", "other"]
    severity: Literal["critical", "high", "medium", "low"]
    summary: Annotated[str, "Summarize the issue in one English sentence"]
    reason: Annotated[str, "Briefly explain this severity level"]
    recommended_team: str

analysis: TicketAnalysis = analyze the current ticket
```

`analysis` is a real Python dict. The model receives its fields, required keys, literal choices and descriptions, even when the type definition is outside the nearby source window. `Annotated` text guides the AI; it is not an executable assertion about sentence length or language. Use a registered validator for additional enforceable conditions.

## Enforcement

- Annotated assignments and later assignments to the same binding are checked. Invalid simple assignments are rejected before replacing the value.
- Function arguments, positional-only/keyword-only arguments, `*args`, `**kwargs`, explicit returns and implicit `None` returns are checked. Async functions use the same rules.
- The return value is checked again after `finally`, so cleanup cannot silently invalidate a return that was already checked.
- Typed class fields are checked on direct attribute assignment. Dataclass values are checked without conversion to dictionaries or reconstruction. Nested registered project instances are checked at execution boundaries.
- Mutable containers are checked deeply. Mutations through aliases are detected at statement boundaries. Enclosing scopes and globals are included; closures preserve referenced local annotation names.
- Generator functions check each yielded value, sent value and final return. Async generators and `yield from` retain their control-flow protocols.
- AI `evaluate`, `execute`, binding writes, terminal results and recovery replacement values use the same checker. Invalid AI output can be repaired within the bounded agent loop before assignment; completed capability side effects are retained.
- Project imports are instrumented too. Unannotated bindings remain dynamic; correctly typed Python does not call AI.

Expected types reach direct AI expressions in annotated assignments, later assignments to declared names, function returns, yields and arguments of locally declared functions. Runtime boundary checks still apply when an expected type cannot be inferred for the planner.

## Type forms

Contracts support primitives, `None`, `Any`, unions/Optional, Literal values, Annotated descriptions, nested list/dict/set/frozenset/tuple, concrete Sequence/Mapping values, TypedDict with Required/NotRequired, dataclasses, nominal classes, Self, type parameters, TypeVar constraints, NewType's underlying runtime type, type[T], and recursive/generic type aliases on Python 3.12+. Generic class fields are checked after substituting supplied type arguments.

Primitive checks are strict: no string-to-number conversion, and bool does not pass an int contract. `Any` is an explicit escape from value checking. Bare containers have unconstrained elements. `Final` bindings reject reassignment; `ClassVar` direct writes must target the class.

Annotations are interpreted rather than passed to `eval`. On Python 3.14+, Aiython uses string-format annotation introspection; on 3.11–3.13, it reads stored annotations without evaluating strings. Type aliases on 3.12–3.13 use Python's lazy alias value machinery, which can evaluate code supplied by the alias author. Custom annotation machinery and custom validators are trusted Python code, not sandboxed code.

`aiython.type_constraints.register_validator(Class, validator)` supplies a custom runtime predicate. A class validator returns true for a valid value. It can also provide the structural check for a Protocol that Aiython cannot prove automatically.

## Limits are explicit

Runtime checking is not a complete static type proof. Unsupported annotations fail with `UnsupportedType`; they are not silently reduced to `Any`. In particular, Callable signatures, ParamSpec/TypeVarTuple, unregistered Protocols, LiteralString provenance, ReadOnly mutation contracts and arbitrary annotation calls are not automatically proven. Lazy iterator objects supplied by other code are not consumed or wrapped merely to guess their element type; use a typed generator function to check values as they pass yield/send boundaries.

Python object identity and side effects are preserved. A failed mutation check does **not** roll back `append`, an external API call, a property setter, or arbitrary native code. Foreign code/threads are not instrumented internally; Aiython checks its own boundaries. Objects can be temporarily invalid before the next boundary check. For a guarantee that invalid values can never enter an object, a different object model or isolation boundary is needed.

`TypeViolation` and `UnsupportedType` are TypeError subclasses and Aiython contract errors. Ordinary Python contract failures stop without asking AI to reinterpret the declared type. Errors identify the value path (for example `analysis['severity']`) and uncaught errors include the project filename/line.

See [the typed result example](https://github.com/sunmodza/aiython/blob/main/examples/recipes/01_typed_result.py) and [the type safety tests](https://github.com/sunmodza/aiython/blob/main/tests/test_type_safety.py).
