"""Exercise Aiython's experimental type bridge on patched CPython."""
import asyncio
import sys

from aiython.native_bridge import NativeTypeBridge
from aiython.type_constraints import TypeViolation


SOURCE = '''
from typing import AsyncGenerator, Final, Generator

yield_cleanup = []

answer: int = 1
fixed: Final[int] = 1

class Box:
    value: int

    def __init__(self, value: int):
        self.value = value

def update(value: int) -> int:
    local: int = value
    return local

def change_global(value: int):
    global answer
    answer = value

def change_final():
    global fixed
    fixed = 2

def change_box(box: Box, value):
    box.value = value

def local_error():
    local: int = "invalid"

def parameter_reassignment(value: int):
    value = "invalid"

def local_final():
    value: Final[int] = 1
    value = 2

def return_error() -> int:
    return "invalid"

def native_feature(value):
    match value:
        case [first, *rest]:
            return first, rest

def identity(function):
    return function

@identity
def decorated(value: int) -> int:
    return value

if True:
    def conditional(value: int) -> int:
        return value

def variadic(*values: int, **named: int):
    return values, named

def make_setter():
    current: int = 1
    def set_current(value):
        nonlocal current
        current = value
        return current
    return set_current

def make_final_setter():
    current: Final[int] = 1
    def set_current():
        nonlocal current
        current = 2
    return set_current

def generator() -> Generator[int, None, None]:
    yield 1

def generator_error() -> Generator[int, None, int]:
    yield 1
    return "invalid"

def yield_error() -> Generator[int, None, None]:
    try:
        yield "invalid"
    finally:
        yield_cleanup.append("closed")

def yield_from_error() -> Generator[int, None, None]:
    yield from (1, "invalid")

async def async_yield_error() -> AsyncGenerator[int, None]:
    yield "invalid"
'''


def rejected(action):
    try:
        action()
    except TypeViolation:
        return
    raise AssertionError('A type violation was not raised')


def main():
    bridge = NativeTypeBridge()
    code = bridge.compile_source(SOURCE, '<native-bridge-test>')
    native = compile(SOURCE, '<native-bridge-test>', 'exec', dont_inherit=True)
    assert (code.co_code, code.co_consts, code.co_linetable, code.co_flags) == (
        native.co_code, native.co_consts, native.co_linetable, native.co_flags)
    namespace = {'__name__': '__main__'}
    with bridge.installed():
        exec(code, namespace)
        assert namespace['update'](3) == 3
        assert namespace['native_feature']([1, 2, 3]) == (1, [2, 3])
        rejected(lambda: namespace['update']('invalid'))
        rejected(lambda: namespace['decorated']('invalid'))
        rejected(lambda: namespace['conditional']('invalid'))
        assert namespace['variadic'](1, extra=2) == ((1,), {'extra': 2})
        rejected(lambda: namespace['variadic']('invalid'))
        rejected(lambda: namespace['variadic'](extra='invalid'))
        setter = namespace['make_setter']()
        assert setter(2) == 2
        rejected(lambda: setter('invalid'))
        rejected(namespace['make_final_setter']())
        assert list(namespace['generator']()) == [1]
        rejected(lambda: list(namespace['generator_error']()))
        if hasattr(sys, 'monitoring'):
            rejected(lambda: list(namespace['yield_error']()))
            assert namespace['yield_cleanup'] == ['closed']
            rejected(lambda: list(namespace['yield_from_error']()))

            async def consume_async():
                async for _ in namespace['async_yield_error']():
                    pass

            rejected(lambda: asyncio.run(consume_async()))
        rejected(namespace['local_error'])
        rejected(namespace['local_final'])
        rejected(lambda: namespace['parameter_reassignment'](1))
        rejected(namespace['return_error'])
        namespace['change_global'](4)
        assert namespace['answer'] == 4
        rejected(lambda: namespace['change_global']('invalid'))
        assert namespace['answer'] == 4
        rejected(namespace['change_final'])
        assert namespace['fixed'] == 1
        box = namespace['Box'](5)
        rejected(lambda: namespace['change_box'](box, 'invalid'))
        assert box.value == 5
    assert all(not hasattr(sys, name) for name in NativeTypeBridge._HOOKS)
    print('Native type bridge passed')


if __name__ == '__main__':
    main()
