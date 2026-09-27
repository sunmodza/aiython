import ast
from collections import OrderedDict
from dataclasses import dataclass
import enum
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, Annotated, Callable, ClassVar, Final, Generic, Literal, NewType, Optional, Protocol, Required, Self, TypeVar, TypeVarTuple, TypedDict, Unpack
import unittest
from unittest.mock import patch

from aiython import type_constraints as tc


class ContractEdgeTests(unittest.TestCase):
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
        with self.assertRaisesRegex(tc.UnsupportedType, 'Generic alias requires'):
            compiler.compile(alias)
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
