import ast
import asyncio
from collections import ChainMap, Counter, OrderedDict, defaultdict, deque
import contextlib
from dataclasses import dataclass
import enum
import io
from pathlib import Path
import re
import sys
import typing
import types
from types import SimpleNamespace
from typing import Any, Annotated, Callable, ClassVar, Final, Generic, Literal, NewType, Optional, Protocol, Required, Self, TypeGuard, TypeVar, TypeVarTuple, TypedDict, Unpack
import unittest
from unittest.mock import patch
from typing_extensions import TypeAlias, TypeIs

from aiython import type_constraints as tc


class ContractEdgeTests(unittest.TestCase):
    def test_bare_abstract_annotations_do_not_consume_values(self):
        namespace = {'typing': typing}
        iterator = (item for item in range(2))
        for annotation in ('typing.Iterable', 'typing.Iterator', 'typing.Generator'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                contract.validate(iterator)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(1)
        self.assertEqual(next(iterator), 0)
        iterator.close()

        async def source():
            yield 1

        async def value():
            return 1

        async_iterator = source()
        for annotation in ('typing.AsyncIterable', 'typing.AsyncIterator',
                           'typing.AsyncGenerator'):
            tc.compile_contract(annotation, namespace).validate(async_iterator)
        coroutine = value()
        for annotation in ('typing.Awaitable', 'typing.Coroutine'):
            tc.compile_contract(annotation, namespace).validate(coroutine)
        coroutine.close()
        asyncio.run(async_iterator.aclose())

        tc.compile_contract('typing.ContextManager', namespace).validate(
            contextlib.nullcontext())
        tc.compile_contract('typing.AsyncContextManager', namespace).validate(
            contextlib.AsyncExitStack())
        tc.compile_contract('typing.ByteString', namespace).validate(bytearray(b'abc'))
        with self.assertRaises(tc.TypeViolation):
            tc.compile_contract('typing.ByteString', namespace).validate(memoryview(b'abc'))
        tc.compile_contract('typing.Type', namespace).validate(int)
        with self.assertRaises(tc.TypeViolation):
            tc.compile_contract('typing.Type', namespace).validate(1)

    def test_mapping_views_check_live_members(self):
        namespace = {'typing': typing}
        cases = (
            ('typing.KeysView[str]', {'x': 1}.keys(), {1: 'x'}.keys()),
            ('typing.ValuesView[int]', OrderedDict(x=1).values(),
             OrderedDict(x='wrong').values()),
            ('typing.ItemsView[str, int]', {'x': 1}.items(),
             {'x': 'wrong'}.items()),
            ('typing.MappingView[tuple[str, int]]', OrderedDict(x=1).items(),
             OrderedDict(x='wrong').items()),
        )
        for annotation, valid, invalid in cases:
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['type'], 'array')
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)
        tc.compile_contract('typing.MappingView', namespace).validate({'x': 1}.keys())

    def test_chainmap_checks_every_underlying_mapping(self):
        namespace = {'ChainMap': ChainMap, 'typing': typing}
        for annotation in ('ChainMap[str, int]', 'typing.ChainMap[str, int]'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['type'], 'object')
                contract.validate(ChainMap({'x': 1}, {'y': 2}))
                with self.assertRaisesRegex(tc.TypeViolation, r'maps\[1\]'):
                    contract.validate(ChainMap({'x': 1}, {'x': 'hidden wrong value'}))
        tc.compile_contract('typing.ChainMap', namespace).validate(
            ChainMap({'x': 1}, {2: 'other'}))

    def test_nominal_abstract_annotations_without_members(self):
        for annotation, valid, invalid in (
            ('typing.Hashable', 3, []),
            ('typing.Sized', [1], 3),
        ):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, {'typing': typing})
                self.assertIn('x-python-abc', contract.schema())
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)

    def test_abstract_collections_check_known_concrete_values(self):
        namespace = {'typing': typing}
        cases = (
            ('typing.MutableSequence[int]', [1, 2], (1, 2)),
            ('typing.AbstractSet[str]', frozenset({'x'}), frozenset({1})),
            ('typing.MutableSet[int]', {1}, frozenset({1})),
            ('typing.Collection[int]', {1: 'value'}, ['wrong']),
            ('typing.Container[int]', {1: 'value'}, {'wrong': 1}),
            ('typing.Reversible[str]', OrderedDict(x=1), OrderedDict({1: 'x'})),
            ('typing.Sequence[int]', memoryview(b'abc'), ['wrong']),
        )
        for annotation, valid, invalid in cases:
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)
        tc.compile_contract('typing.MutableSequence[int]', namespace).validate(bytearray(b'a'))
        tc.compile_contract('typing.Reversible[int]', namespace).validate(deque([1, 2]))
        with self.assertRaises(tc.TypeViolation):
            tc.compile_contract('typing.Reversible[int]', namespace).validate({1, 2})
        for annotation, value in (('typing.List', [1, 'x']),
                                  ('typing.Dict', {'x': 1}),
                                  ('typing.Collection', {1: 2}),
                                  ('typing.Container', {1: 2}),
                                  ('typing.Reversible', range(2)),
                                  ('typing.Sequence', memoryview(b'a')),
                                  ('typing.Tuple', (1, 'x'))):
            with self.subTest(annotation=annotation):
                tc.compile_contract(annotation, namespace).validate(value)
        empty = tc.compile_contract('typing.Tuple[()]', namespace)
        empty.validate(())
        with self.assertRaises(tc.TypeViolation):
            empty.validate((1,))

    def test_abstract_mappings_accept_known_concrete_implementations(self):
        namespace = {'typing': typing}
        for annotation in ('typing.Mapping[str, int]', 'typing.MutableMapping[str, int]'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['type'], 'object')
                for value in (dict(x=1), OrderedDict(x=1),
                              defaultdict(int, x=1), Counter(x=1)):
                    contract.validate(value)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(OrderedDict(x='wrong'))
        with self.assertRaises(tc.TypeViolation):
            tc.compile_contract('dict[str, int]', {}).validate(OrderedDict(x=1))

    def test_io_annotations_use_standard_stream_classes(self):
        text = io.StringIO('alpha')
        binary = io.BytesIO(b'beta')
        for annotation, valid, invalid, stream_type in (
            ('typing.IO[str]', text, binary, 'str'),
            ('typing.IO[bytes]', binary, text, 'bytes'),
            ('typing.TextIO', text, binary, 'str'),
            ('typing.BinaryIO', binary, text, 'bytes'),
        ):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, {'typing': typing})
                self.assertEqual(contract.schema()['x-python-io'], stream_type)
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)
        tc.compile_contract('typing.IO', {'typing': typing}).validate(text)
        tc.compile_contract('typing.IO[Any]', {'typing': typing}).validate(binary)
        generic = tc.compile_contract('typing.IO[typing.AnyStr]', {'typing': typing})
        returned = tc.compile_contract('typing.AnyStr', {'typing': typing})
        for stream, matching, wrong in ((text, 'text', b'wrong'),
                                        (binary, b'binary', 'wrong')):
            with self.subTest(stream=type(stream).__name__):
                bindings = {}
                generic.validate(stream, bindings=bindings)
                returned.validate(matching, bindings=bindings)
                with self.assertRaises(tc.TypeViolation):
                    returned.validate(wrong, bindings=bindings)

    def test_regex_generic_annotations_check_input_type(self):
        namespace = {'re': re, 'typing': typing}
        cases = (
            ('re.Pattern[str]', re.compile('a'), re.compile(b'a'), 'pattern'),
            ('typing.Pattern[str]', re.compile('a'), re.compile(b'a'), 'pattern'),
            ('re.Match[bytes]', re.match(b'a', b'a'), re.match('a', 'a'), 'match'),
            ('typing.Match[str]', re.match('a', 'a'), re.match(b'a', b'a'), 'match'),
        )
        for annotation, valid, invalid, kind in cases:
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['x-python-regex'], kind)
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)

    def test_concrete_collections_preserve_generic_member_checks(self):
        namespace = {'typing': typing, 'deque': deque, 'defaultdict': defaultdict,
                     'OrderedDict': OrderedDict, 'Counter': Counter}
        cases = (
            ('deque[int]', deque([1]), deque(['wrong']), 'array'),
            ('typing.Deque[int]', deque([1]), deque(['wrong']), 'array'),
            ('defaultdict[str, int]', defaultdict(int, {'x': 1}),
             defaultdict(int, {'x': 'wrong'}), 'object'),
            ('typing.DefaultDict[str, int]', defaultdict(int, {'x': 1}),
             defaultdict(int, {1: 2}), 'object'),
            ('OrderedDict[str, int]', OrderedDict([('x', 1)]),
             OrderedDict([('x', 'wrong')]), 'object'),
            ('Counter[str]', Counter({'x': 2}), Counter({'x': 'wrong'}), 'object'),
            ('typing.Counter[str]', Counter({'x': 2}), Counter({1: 2}), 'object'),
        )
        for annotation, valid, invalid, schema_type in cases:
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['type'], schema_type)
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)

    def test_type_alias_marker_allows_alias_declaration(self):
        marker = tc.compile_contract('TypeAlias', {'TypeAlias': TypeAlias})
        self.assertEqual(marker.kind, 'any')
        marker.validate(list[int])
        marker.validate('list[int]')
        alias = tc.compile_contract('Values', {'Values': list[int]})
        alias.validate([1, 2])
        with self.assertRaises(tc.TypeViolation):
            alias.validate(['wrong'])

    def test_type_narrowing_annotations_check_boolean_results(self):
        namespace = {'TypeGuard': TypeGuard, 'TypeIs': TypeIs}
        for annotation in ('TypeGuard[int]', TypeGuard[int],
                           'TypeIs[int]', TypeIs[int], 'TypeGuard[Undefined]'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                self.assertEqual(contract.schema()['type'], 'boolean')
                contract.validate(True)
                contract.validate(False)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(1)

    def test_inherited_generic_fields_use_base_type_arguments(self):
        variable = TypeVar('T')
        item = TypeVar('Item')

        class Base(Generic[variable]):
            value: variable
            def __init__(self, value): self.value = value

        class Middle(Base[list[item]], Generic[item]):
            pass

        class Leaf(Middle[int]):
            pass

        contract = tc.compile_contract(Leaf, locals())
        contract.validate(Leaf([1, 2]))
        with self.assertRaises(tc.TypeViolation):
            contract.validate(Leaf(['wrong']))

    @unittest.skipIf(sys.version_info < (3, 12), 'generic class syntax requires Python 3.12')
    def test_inherited_pep695_fields_keep_each_class_parameter_scope(self):
        namespace = {'__name__': __name__}
        exec('class Base[T]:\n'
             '    value: T\n'
             '    def __init__(self, value): self.value = value\n'
             'class Child[T](Base[str]):\n'
             '    other: T\n'
             '    def __init__(self, value, other):\n'
             '        super().__init__(value)\n'
             '        self.other = other\n', namespace)
        contract = tc.compile_contract('Child[int]', namespace)
        contract.validate(namespace['Child']('ok', 1))
        for value, other in ((1, 1), ('ok', 'wrong')):
            with self.subTest(value=value, other=other), self.assertRaises(tc.TypeViolation):
                contract.validate(namespace['Child'](value, other))

    @unittest.skipIf(sys.version_info < (3, 12), 'generic class syntax requires Python 3.12')
    def test_inherited_parameters_with_matching_names_keep_their_identity(self):
        namespace = {'__name__': __name__}
        exec('class Base[T, U]:\n'
             '    left: T\n'
             '    right: U\n'
             '    def __init__(self, left, right): self.left, self.right = left, right\n'
             'class Child[T, U](Base[U, T]): pass\n', namespace)
        contract = tc.compile_contract('Child[int, str]', namespace)
        contract.validate(namespace['Child']('left', 1))
        with self.assertRaises(tc.TypeViolation):
            contract.validate(namespace['Child'](1, 'right'))

    @unittest.skipIf(sys.version_info < (3, 13), 'type parameter defaults require Python 3.13')
    def test_inherited_default_can_reference_earlier_base_parameter(self):
        namespace = {'__name__': __name__}
        exec('class Base[T, U = list[T]]:\n'
             '    value: U\n'
             '    def __init__(self, value): self.value = value\n'
             'class Child(Base[int]): pass\n', namespace)
        contract = tc.compile_contract('Child', namespace)
        contract.validate(namespace['Child']([1]))
        with self.assertRaises(tc.TypeViolation):
            contract.validate(namespace['Child'](['wrong']))

    @unittest.skipIf(sys.version_info < (3, 12), 'variadic class syntax requires Python 3.12')
    def test_inherited_variadic_generic_fields_expand_arguments(self):
        namespace = {'__name__': __name__}
        exec('class Base[*Ts]:\n'
             '    value: tuple[*Ts]\n'
             '    def __init__(self, value): self.value = value\n'
             'class Child[*Us](Base[*Us]): pass\n', namespace)
        child = namespace['Child']
        for annotation, valid, invalid in (('Child[int, str]', (1, 'x'), (1, 2)),
                                           ('Child[()]', (), (1,))):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                contract.validate(child(valid))
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(child(invalid))

    @unittest.skipIf(sys.version_info < (3, 12), 'generic class syntax requires Python 3.12')
    def test_variadic_generic_class_specialization(self):
        namespace = {'__name__': __name__}
        exec('class Box[*Ts]:\n'
             '    value: tuple[*Ts]\n'
             '    def __init__(self, value): self.value = value\n', namespace)
        contract = tc.compile_contract('Box[int, str]', namespace)
        contract.validate(namespace['Box']((1, 'x')))
        with self.assertRaises(tc.TypeViolation):
            contract.validate(namespace['Box']((1, 2)))

    @unittest.skipIf(sys.version_info < (3, 13), 'type parameter defaults require Python 3.13')
    def test_defaulted_type_alias_parameters(self):
        namespace = {'__name__': __name__}
        exec('type Pair[T, U = str] = tuple[T, U]\n'
             'type Again[T, U = T] = tuple[T, U]\n'
             'type Variadic[T, *Ts, U = str] = tuple[T, *Ts, U]\n', namespace)
        pair = tc.compile_contract('Pair[int]', namespace)
        pair.validate((1, 'x'))
        with self.assertRaises(tc.TypeViolation):
            pair.validate((1, 2))
        again = tc.compile_contract('Again[int]', namespace)
        again.validate((1, 2))
        with self.assertRaises(tc.TypeViolation):
            again.validate((1, 'x'))
        for annotation, value in (('Variadic[int]', (1, 'x')),
                                  ('Variadic[int, bool]', (1, True)),
                                  ('Variadic[int, bool, str]', (1, True, 'x'))):
            with self.subTest(annotation=annotation):
                tc.compile_contract(annotation, namespace).validate(value)

    @unittest.skipIf(sys.version_info < (3, 13), 'type parameter defaults require Python 3.13')
    def test_defaulted_generic_class_parameters(self):
        namespace = {'__name__': __name__}
        exec('class Pair[T, U = str]:\n'
             '    left: T\n'
             '    right: U\n'
             '    def __init__(self, left, right):\n'
             '        self.left, self.right = left, right\n', namespace)
        contract = tc.compile_contract('Pair[int]', namespace)
        contract.validate(namespace['Pair'](1, 'x'))
        with self.assertRaises(tc.TypeViolation):
            contract.validate(namespace['Pair'](1, 2))

    @unittest.skipIf(sys.version_info < (3, 12), 'type statements require Python 3.12')
    def test_variadic_type_alias_specialization(self):
        namespace = {'__name__': __name__}
        exec('type TupleAlias[*Ts] = tuple[*Ts]\n'
             'type Mixed[T, *Ts, U] = tuple[T, *Ts, U]\n', namespace)
        specialized = tc.compile_contract('TupleAlias[int, str]', namespace)
        specialized.validate((1, 'x'))
        with self.assertRaises(tc.TypeViolation):
            specialized.validate((1, 2))
        tc.compile_contract('TupleAlias', namespace).validate((1, 'x', True))
        empty = tc.compile_contract('TupleAlias[()]', namespace)
        empty.validate(())
        with self.assertRaises(tc.TypeViolation):
            empty.validate((1,))
        mixed = tc.compile_contract('Mixed[int, str, bool]', namespace)
        mixed.validate((1, 'x', True))
        with self.assertRaises(tc.TypeViolation):
            mixed.validate((1, 'x', 3))

    def test_callable_contract_checks_callable_without_claiming_signature(self):
        for annotation in ('Callable[[int], str]', 'Callable[..., str]', 'Callable'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, {'Callable': Callable})
                self.assertTrue(contract.schema()['x-python-callable'])
                contract.validate(str)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(3)

    def test_variadic_tuple_contract_keeps_fixed_members(self):
        parameters = {'Ts': TypeVarTuple('Ts'), 'Unpack': Unpack}
        for annotation in ('tuple[*Ts]', 'tuple[Unpack[Ts]]'):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, parameters)
                self.assertEqual(contract.kind, 'tuple_many')
                contract.validate((1, 'x'))
        contract = tc.compile_contract('tuple[int, *Ts, str]', parameters)
        schema = contract.schema()
        self.assertEqual(schema['minItems'], 2)
        self.assertEqual(schema['prefixItems'][0]['type'], 'integer')
        self.assertEqual(schema['x-python-suffixItems'][0]['type'], 'string')
        contract.validate((1, 2, 'x'))
        for value in ((1,), ('bad', 2, 'x'), (1, 2, 3), [1, 2, 'x']):
            with self.subTest(value=value), self.assertRaises(tc.TypeViolation):
                contract.validate(value)

    def test_schema_variants_and_recursive_contract(self):
        integer = tc.compile_contract('int', {})
        cases = [
            (tc.Contract('any', 'Any'), {}),
            (tc.Contract('never', 'Never'), {'not': {}}),
            (tc.Contract('union', 'int | str', (integer, tc.compile_contract('str', {}))), {'anyOf': []}),
            (tc.Contract('null', 'None'), {'type': 'null'}),
            (tc.Contract('tuple', 'tuple[int]', (integer,)), {'prefixItems': []}),
            (tc.Contract('tuple_many', 'tuple[int, ...]', (integer,)), {'items': {}}),
            (tc.Contract('dict', 'dict[str, int]', (tc.compile_contract('str', {}), integer)), {'type': 'object'}),
            (tc.Contract('typevar', 'T', (integer,)), {'anyOf': []}),
            (tc.Contract('typevar', 'T'), {}),
        ]
        for contract, expected in cases:
            with self.subTest(contract=contract.name):
                schema = contract.schema()
                for key in expected:
                    self.assertIn(key, schema)
        self.assertTrue(integer.schema({id(integer)})['x-recursive'])
        self.assertEqual(tc.Contract('unsupported', 'custom').schema()['x-python-type'], 'custom')
        self.assertEqual(tc.compile_contract('Literal[b"yes"]', {}).schema()['x-python-literals'],
                         ["b'yes'"])

        class Level(enum.Enum):
            LOW = 'low'
            HIGH = 'high'
        schema = tc.compile_contract('Literal[Level.HIGH]', {'Level': Level}).schema()
        self.assertEqual(len(schema['x-python-enum-members']), 1)
        self.assertTrue(schema['x-python-enum-members'][0].endswith('Level.HIGH'))
        class Strange(enum.Enum):
            ITEM = object()
        with self.assertRaisesRegex(tc.UnsupportedType, 'scalar'):
            tc.compile_contract('Strange', {'Strange': Strange}).schema()

    def test_validation_rejects_invalid_scalar_container_and_typevar_values(self):
        cases = [
            ('Never', 1), ('Literal[1]', True), ('None', 1), ('int', True),
            ('list[int]', (1,)), ('list[int]', [1, 'bad']),
            ('set[int]', {'bad'}), ('frozenset[int]', frozenset({'bad'})),
            ('tuple[int, str]', (1,)), ('tuple[int, str]', (1, 2)),
            ('dict[str, int]', [('x', 1)]), ('dict[str, int]', {1: 2}),
            ('dict[str, int]', {'x': 'bad'}), ('type[int]', str),
            ('type[int]', 1), ('Iterator[int]', iter([1])),
        ]
        for annotation, value in cases:
            with self.subTest(annotation=annotation, value=value):
                with self.assertRaises((tc.TypeViolation, tc.UnsupportedType)):
                    tc.compile_contract(annotation, {}).validate(value)
        tc.compile_contract('Sequence[int]', {}).validate((1, 2))
        with self.assertRaisesRegex(tc.TypeViolation, 'concrete sequences'):
            tc.compile_contract('Sequence[int]', {}).validate(iter([1]))
        tc.compile_contract('Mapping[str, int]', {}).validate({'x': 1})
        tc.compile_contract('type[int]', {}).validate(int)

        variable = TypeVar('Variable', int, str)
        contract = tc.compile_contract(variable, {})
        bindings = {}
        contract.validate(1, bindings=bindings)
        self.assertEqual(bindings[variable], int)
        with self.assertRaisesRegex(tc.TypeViolation, 'inconsistent'):
            contract.validate('later', bindings=bindings)
        with self.assertRaises(tc.TypeViolation):
            contract.validate(1.5)

    def test_type_contracts_accept_unions_aliases_and_typevars(self):
        namespace = {'typing': typing}
        for annotation, valid, invalid in (
            ('type[int | str]', int, float),
            ('typing.Type[typing.Union[int, str]]', str, float),
            ('type[None]', type(None), int),
            ('type[typing.Annotated[int, "metadata"]]', int, str),
        ):
            with self.subTest(annotation=annotation):
                contract = tc.compile_contract(annotation, namespace)
                contract.validate(valid)
                with self.assertRaises(tc.TypeViolation):
                    contract.validate(invalid)

        alias = tc.TypeAliasType('ClassAlias', int | str)
        tc.compile_contract('type[ClassAlias]', {'ClassAlias': alias}).validate(str)
        recursive = tc.TypeAliasType('RecursiveClass', 'int | RecursiveClass')
        recursive_contract = tc.compile_contract('type[RecursiveClass]',
                                                 {'RecursiveClass': recursive})
        recursive_contract.validate(int)
        with self.assertRaises(tc.TypeViolation):
            recursive_contract.validate(str)

        variable = TypeVar('ClassVariable', int, str)
        namespace['ClassVariable'] = variable
        class_contract = tc.compile_contract('type[ClassVariable]', namespace)
        value_contract = tc.compile_contract('ClassVariable', namespace)
        bindings = {}
        class_contract.validate(int, bindings=bindings)
        value_contract.validate(3, bindings=bindings)
        with self.assertRaises(tc.TypeViolation):
            value_contract.validate('wrong', bindings=bindings)
        with self.assertRaises(tc.TypeViolation):
            class_contract.validate(str, bindings=bindings)
        with self.assertRaises(tc.TypeViolation):
            class_contract.validate(float, bindings={})
        reverse_bindings = {}
        value_contract.validate('first', bindings=reverse_bindings)
        class_contract.validate(str, bindings=reverse_bindings)

    def test_class_custom_validator_and_missing_field(self):
        class Choice:
            value: int
        contract = tc.compile_contract(Choice, {})
        instance = Choice()
        contract.validate(instance)
        instance.value = 'bad'
        with self.assertRaises(tc.TypeViolation):
            contract.validate(instance)
        @dataclass
        class Item:
            count: int
        item = object.__new__(Item)
        with self.assertRaisesRegex(tc.TypeViolation, 'missing attribute'):
            tc.compile_contract(Item, {}).validate(item)

        class External:
            pass
        tc.register_validator(External, lambda value: getattr(value, 'valid', False))
        self.addCleanup(tc.VALIDATORS.pop, External)
        external = External()
        with self.assertRaisesRegex(tc.TypeViolation, 'custom validator'):
            tc.compile_contract(External, {}).validate(external)
        external.valid = True
        tc.compile_contract(External, {}).validate(external)

    def test_direct_cycle_and_unsupported_contract_paths(self):
        recursive = tc.Contract('list', 'recursive')
        recursive.args = (recursive,)
        value = []
        value.append(value)
        recursive.validate(value)
        with self.assertRaises(tc.UnsupportedType):
            tc.Contract('unknown', 'unknown').validate(1)


class CompilerEdgeTests(unittest.TestCase):
    def test_legacy_annotation_module_import_fallback(self):
        real_import = __import__
        def imported(name, *args, **kwargs):
            if name == 'annotationlib':
                raise ModuleNotFoundError('annotationlib unavailable')
            return real_import(name, *args, **kwargs)
        module = types.ModuleType('aiython._legacy_type_constraints')
        module.__package__ = 'aiython'
        module.__file__ = tc.__file__
        sys.modules[module.__name__] = module
        try:
            with patch('builtins.__import__', side_effect=imported):
                exec(compile(Path(tc.__file__).read_text(), tc.__file__, 'exec'), module.__dict__)
        finally:
            sys.modules.pop(module.__name__)
        self.assertIsNone(module.annotationlib)
        class Sample:
            value: int
        self.assertEqual(module.annotations_of(Sample), {'value': int})

    def test_alias_generic_protocol_and_class_field_paths(self):
        variable = TypeVar('T')
        alias = tc.TypeAliasType('Items', list[variable], type_params=(variable,))
        compiler = tc.Compiler({'Items': alias})
        unspecialized = compiler.compile(alias)
        self.assertEqual(unspecialized.schema()['type'], 'array')
        unspecialized.validate([1, 2])
        with patch.object(tc, 'annotationlib', None):
            compiled = compiler.compile('Items[int]')
        self.assertEqual(compiled.schema()['type'], 'array')
        args = (tc.compile_contract('int', {}),)
        self.assertIs(compiler.alias(alias, compiler.names, args, 'Items[int]'),
                      compiler.alias(alias, compiler.names, args, 'Items[int]'))

        class Box(Generic[variable]):
            value: variable
        with self.assertRaisesRegex(tc.UnsupportedType, 'Generic type argument count'):
            compiler.generic(Box, (), 'Box[]', {})
        class MissingValidator(Protocol):
            def run(self) -> int: ...
        with self.assertRaisesRegex(tc.UnsupportedType, 'Protocol requires'):
            compiler.class_contract(MissingValidator, compiler.names)
        class Settings:
            count: int
            shared: ClassVar[int]
        settings = compiler.class_contract(Settings, compiler.names)
        self.assertEqual(set(settings.fields), {'count'})
        self.assertIs(compiler.class_contract(Settings, compiler.names), settings)

        class Payload(TypedDict):
            required: Required[int]
        payload = compiler.class_contract(Payload, compiler.names)
        self.assertIn('required', payload.required)

    def test_compiler_direct_boundaries_and_dynamic_values(self):
        compiler = tc.Compiler({'number': 2})
        contract = tc.compile_contract('int', {})
        self.assertIs(compiler.compile(contract), contract)
        self.assertIs(compiler.value(contract, compiler.names), contract)
        with self.assertRaisesRegex(tc.UnsupportedType, 'module or class'):
            compiler.lookup(ast.parse('number.real', mode='eval').body, compiler.names)
        with self.assertRaisesRegex(tc.UnsupportedType, 'Qualifier requires'):
            compiler.generic(Final, (), 'Final[]', {})
        with self.assertRaisesRegex(tc.UnsupportedType, 'ReadOnly needs'):
            tc.compile_contract('ReadOnly[int]', {'ReadOnly': tc.ReadOnly})
        self.assertEqual(compiler.value(Annotated[int, 'note'], compiler.names).description, 'note')
        self.assertEqual(compiler.value(__import__('typing').ForwardRef('int'), compiler.names).kind, 'int')
        with self.assertRaisesRegex(tc.UnsupportedType, 'supported Python type'):
            compiler.value(3, compiler.names)
        variable = TypeVar('T')
        self.assertIs(compiler.value(variable, {'T': contract}), contract)
        self.assertEqual(tc.ContractCache().compile(int, {}).kind, 'int')
        with self.assertRaisesRegex(tc.UnsupportedType, 'Unsupported generic'):
            compiler.generic(object, (), 'object[]', {})

    def test_annotationlib_access_and_alias_evaluation_when_available(self):
        fake = SimpleNamespace(
            Format=SimpleNamespace(STRING='string'),
            get_annotations=lambda target, format: {'value': 'int'},
            call_evaluate_function=lambda function, format: int,
        )
        class Sample:
            value: int
        alias = SimpleNamespace(__type_params__=(), __name__='Alias',
                                evaluate_value=lambda: int)
        with patch.object(tc, 'annotationlib', fake):
            self.assertEqual(tc.annotations_of(Sample), {'value': 'int'})
            self.assertEqual(tc.Compiler({}).alias(alias, {}).args[0].kind, 'int')

    def test_compiler_rejects_unsafe_and_unresolved_annotations(self):
        cases = [
            'Missing', 'danger()', 'Unknown.attribute', 'Literal[1.5]',
            'ReadOnly[int]', 'Self', 'LiteralString',
            'list[int, str]', 'dict[str]', 'Generator[int, str]', 'type[int, str]',
        ]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(tc.UnsupportedType):
                tc.compile_contract(source, {})
        self.assertEqual(tc.compile_contract('Self', {'self': 1}).name, 'int')
        self.assertEqual(tc.compile_contract('NewTypeValue',
                             {'NewTypeValue': NewType('NewTypeValue', int)}).name, 'int')
        with self.assertRaises(tc.UnsupportedType):
            tc.compile_contract('int[0]', {})

    def test_compiler_literals_qualifiers_and_plain_types(self):
        self.assertEqual(tc.compile_contract(None, {}).kind, 'null')
        self.assertEqual(tc.compile_contract(Any, {}).kind, 'any')
        self.assertEqual(tc.compile_contract(Final, {}).marker, 'Final')
        self.assertEqual(tc.compile_contract(ClassVar, {}).marker, 'ClassVar')
        self.assertEqual(tc.compile_contract('Optional[int]', {}).kind, 'union')
        self.assertEqual(tc.compile_contract('tuple[int, ...]', {}).kind, 'tuple_many')
        self.assertEqual(tc.compile_contract(tuple, {}).kind, 'tuple_many')
        self.assertEqual(tc.compile_contract(dict, {}).kind, 'dict')
        self.assertEqual(tc.compile_contract('Annotated[int, "a", 2]', {}).description, 'a')
        self.assertEqual(tc.compile_contract(Literal[1], {}).kind, 'literal')
        self.assertEqual(tc.compile_contract('int | str', {}).kind, 'union')
        self.assertEqual(tc.Compiler({}).generic(__import__('typing').Union,
            (tc.compile_contract('int', {}), tc.compile_contract('str', {})),
            'Union[int, str]', {}).kind, 'union')
        self.assertIsNone(tc.describe_output(None, {}))
        self.assertIsNone(tc.validate_output(1, None, {}))

    def test_cache_evicts_oldest_and_avoids_invalid_annotation(self):
        cache = tc.ContractCache()
        self.assertEqual(cache.compile('int', {}).kind, 'int')
        self.assertIs(cache.compile('int', {}), cache.compile('int', {}))
        with self.assertRaises(tc.UnsupportedType):
            cache.compile('int[', {})
        cache.entries = OrderedDict(((str(i), i), None) for i in range(257))
        cache.compile('str', {})
        self.assertEqual(len(cache.entries), 257)
        self.assertNotIn(('0', 0), cache.entries)


if __name__ == '__main__':
    unittest.main()
