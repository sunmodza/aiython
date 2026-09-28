import ast
import dataclasses
import enum
import inspect
import typing
import weakref
from types import SimpleNamespace
from typing import ClassVar, Final
import unittest
from unittest.mock import patch

from aiython.frontend import RUNTIME_NAME
from aiython import typed_runtime as tr
from aiython.type_constraints import TypeViolation
from aiython.typed_runtime import ExpectedTypes, SCOPE, Scope, TypeRuntime, TypedTransformer


class TypeRuntimeEdgeTests(unittest.TestCase):
    def test_legacy_functions_expose_explicit_generic_parameters(self):
        parameter = typing.TypeVar('__Hidden')
        visible = typing.TypeVar('Visible')

        def standalone(value):
            return value

        standalone.__type_params__ = (parameter,)
        self.runtime.register_function(standalone)
        self.assertEqual(self.runtime.function_type_params[standalone.__code__],
                         (parameter,))

        class Owner(typing.Generic[parameter, visible]):
            def method(self):
                return runtime.namespace(inspect.currentframe())

        runtime = self.runtime
        Owner.method.__type_params__ = (parameter,)
        self.runtime.register_class(Owner)
        namespace = Owner().method()
        self.assertIs(namespace['__Hidden'], parameter)
        self.assertIs(namespace['_Owner__Hidden'], parameter)
        self.assertIs(namespace['Visible'], visible)
        self.assertEqual(self.runtime.function_type_params[Owner.method.__code__],
                         (parameter,))
        self.assertTrue(tr.unconstrained_variadic('*Ts', 'args', {}))

    def test_custom_dataclass_constructor_may_omit_initvar(self):
        @dataclasses.dataclass
        class Custom:
            supplied: dataclasses.InitVar[int]

            def __init__(self):
                self.ready = True

        self.runtime.register_class(Custom)
        self.assertTrue(Custom().ready)

    def test_method_owner_fallbacks_for_incomplete_mro_and_unrelated_receiver(self):
        runtime = self.runtime

        class Owner:
            def method(self):
                return runtime.method_self_owner(inspect.currentframe())

        class Other:
            method = Owner.method

        runtime.method_owners[Owner.method.__code__] = weakref.WeakSet([Owner])
        self.assertIs(Other().method(), Owner)

        class TypeProxy:
            def __call__(self, value):
                return type(value)

            def __getattribute__(self, name):
                if name == '__getattribute__':
                    return lambda candidate, attribute: []
                return object.__getattribute__(self, name)

        with patch.object(tr, 'type', TypeProxy(), create=True), \
                patch.object(tr, 'issubclass', lambda candidate, parent: False, create=True):
            self.assertIs(Owner().method(), Owner)

    def test_registered_instance_with_unreadable_or_nonmapping_dict(self):
        class Unreadable:
            def __get__(self, instance, owner):
                raise AttributeError('unreadable')

        class NonMapping:
            def __get__(self, instance, owner):
                return []

        class First:
            __dict__ = Unreadable()

        class Second:
            __dict__ = NonMapping()

        self.runtime.register_class(First)
        self.runtime.register_class(Second)
        frame = inspect.currentframe()
        with patch.object(tr.types, 'GetSetDescriptorType', Unreadable):
            self.runtime._check_instances(First(), {First}, set(), frame)
        with patch.object(tr.types, 'GetSetDescriptorType', NonMapping):
            self.runtime._check_instances(Second(), {Second}, set(), frame)

    def test_ambiguous_generic_method_owner_does_not_invent_binding(self):
        parameter = typing.TypeVar('Parameter')

        class First(typing.Generic[parameter]):
            pass

        class Second(typing.Generic[parameter]):
            pass

        def inspect_namespace(value):
            return self.runtime.namespace(inspect.currentframe())

        self.runtime.method_owners[inspect_namespace.__code__] = weakref.WeakSet([First, Second])
        with patch.object(self.runtime, 'method_self_owner', return_value=None):
            namespace = inspect_namespace(1)
        self.assertNotIn('Parameter', namespace)

    def test_enum_metadata_recursive_guard_and_classmethod_registration(self):
        class Choice(enum.Enum):
            first: int = 1
            description: str

        self.runtime.register_class(Choice)
        value = object()
        token = self.runtime._active_instance_checks.set(frozenset({id(value)}))
        try:
            self.runtime._validate_instance(value, 'value', inspect.currentframe())
        finally:
            self.runtime._active_instance_checks.reset(token)
        method = classmethod(lambda cls: cls)
        self.assertIs(self.runtime.register_function(method), method)
        self.assertIs(self.runtime.register_function(42), 42)

    def test_method_owner_is_discovered_when_called_before_registration(self):
        class Owner:
            def method(self):
                return self_runtime.method_self_owner(inspect.currentframe(), discover=True)

        self_runtime = self.runtime
        self.assertIs(Owner().method(), Owner)

    def test_final_class_placeholder_cannot_be_reassigned(self):
        runtime = self.runtime
        with self.assertRaisesRegex(TypeViolation, 'Final binding'):
            class Owner:
                __aiython_type_scope__ = Scope(final_names={'value'})
                value = runtime.assignment_in(inspect.currentframe(), dataclasses.field(),
                                              'value')

    def setUp(self):
        self.runtime = TypeRuntime()

    def test_registration_scopes_and_expression_validation(self):
        self.assertEqual(self.runtime.register_class(42), 42)
        class Sample:
            count: int
            def __init__(self): self.count = 1
        self.assertIs(self.runtime.register_class(Sample), Sample)
        self.assertIsNone(self.runtime.validate_output(1, None, inspect.currentframe()))
        self.assertIsNone(self.runtime.describe_output(None, inspect.currentframe()))
        self.assertEqual(self.runtime.expression(1, 'int'), 1)
        with self.assertRaisesRegex(TypeViolation, 'expression'):
            self.runtime.expression('wrong', 'int')

        def scoped():
            nonlocal_scope = Scope(declarations={'x': 'int'})
            __aiython_type_scope__ = Scope(declarations={'y': 'str'})
            globals()[SCOPE] = nonlocal_scope
            try:
                scopes = self.runtime.scopes(inspect.currentframe())
                self.assertEqual(len(scopes), 2)
            finally:
                globals().pop(SCOPE)
        scoped()

    def test_final_reassignment_and_missing_scope_values(self):
        def final():
            __aiython_type_scope__ = Scope(final_names={'answer'})
            self.runtime.reassigning(['answer'])
        with self.assertRaisesRegex(TypeViolation, 'Final binding'):
            final()

        def declared_but_unbound():
            __aiython_type_scope__ = Scope(contracts={
                'future_value': self.runtime.contract('int', locals())})
            self.runtime.check_frame(inspect.currentframe())
            self.assertIn('future_value', __aiython_type_scope__.contracts)
        declared_but_unbound()

    def test_registered_slotted_class_and_container_are_inspected(self):
        class Slot:
            __slots__ = ('value', '__weakref__')
            value: int
            def __init__(self): self.value = 2
        self.runtime.register_class(Slot)
        item = Slot()
        items = [item]
        self.runtime.check_frame(inspect.currentframe())
        self.assertEqual(items[0].value, 2)

    def test_annotation_without_class_descriptor_still_checks_instance(self):
        class Descriptor:
            def __set__(self, instance, value):
                pass

        class Item:
            value: Descriptor

        item = Item()
        with self.assertRaises(TypeViolation):
            self.runtime.assign_attribute(item, 'value', 5)
        value = Descriptor()
        self.runtime.assign_attribute(item, 'value', value)
        self.assertIs(item.value, value)

    def test_return_yield_abort_leave_and_send(self):
        self.assertEqual(self.runtime.returned(3), 3)
        self.assertEqual(self.runtime.yielded(3), 3)
        self.assertEqual(self.runtime.sent('ok'), 'ok')
        self.runtime.aborted()
        self.runtime.leaving()

        def invalid_yield():
            __aiython_type_scope__ = Scope(returned=self.runtime.contract('int', locals()))
            self.runtime.yielded(1)
        with self.assertRaisesRegex(TypeViolation, 'Generator return annotation'):
            invalid_yield()

        def failed_scope():
            __aiython_type_scope__ = Scope()
            self.runtime.aborted()
            self.assertTrue(__aiython_type_scope__.failed)
            self.runtime.leaving()
        failed_scope()

    def test_delegated_generator_send_throw_and_close(self):
        def child():
            try:
                value = yield 1
                while True:
                    try:
                        value = yield len(value)
                    except ValueError:
                        yield 3
            finally:
                closed.append(True)
        closed = []
        def delegated():
            __aiython_type_scope__ = Scope(returned=self.runtime.contract(
                'Generator[int, str, int]', {'Generator': __import__('typing').Generator}))
            return self.runtime.delegate(child())
        iterator = delegated()
        self.assertEqual(next(iterator), 1)
        self.assertEqual(iterator.send('ok'), 2)
        self.assertEqual(iterator.throw(ValueError('retry')), 3)
        iterator.close()
        self.assertEqual(closed, [True])

        def finished():
            return 7
            yield
        with self.assertRaises(StopIteration) as stop:
            next(self.runtime.delegate(finished()))
        self.assertEqual(stop.exception.value, 7)

        delegated_list = self.runtime.delegate(iter([1, 2]))
        self.assertEqual(next(delegated_list), 1)
        self.assertEqual(delegated_list.send(None), 2)
        with self.assertRaisesRegex(ValueError, 'cannot forward'):
            delegated_list.throw(ValueError('cannot forward'))

        def recover():
            try:
                yield 1
            except ValueError:
                return 9
        delegated_recovery = self.runtime.delegate(recover())
        self.assertEqual(next(delegated_recovery), 1)
        with self.assertRaises(StopIteration) as recovered:
            delegated_recovery.throw(ValueError('handled'))
        self.assertEqual(recovered.exception.value, 9)

    def test_attribute_contracts_for_classvar_and_final(self):
        class Owner:
            pass
        owner = Owner()
        with self.assertRaisesRegex(TypeViolation, 'ClassVar'):
            self.runtime.assign_attribute(owner, 'shared', 1, 'ClassVar[int]')
        self.runtime.assign_attribute(owner, 'constant', 1, 'Final[int]')
        with self.assertRaisesRegex(TypeViolation, 'Final attribute'):
            self.runtime.assign_attribute(owner, 'constant', 2, 'Final[int]')
        self.assertEqual(owner.constant, 1)


class TypedTransformerEdgeTests(unittest.TestCase):
    def test_legacy_parser_preserves_variadic_and_synthetic_type_parameters(self):
        transformer = TypedTransformer()
        transformer.body(ast.parse('pass\n').body,
                         parameters={'values': ('*Ts', 'args')})
        function = ast.parse('def identity(value):\n    return value\n').body[0]
        function.type_params = [object()]
        transformer.visit_FunctionDef(function)
        self.assertTrue(function.decorator_list)
        self.assertEqual(function.decorator_list[-1].attr, 'register_function')

    def test_docstring_only_class_keeps_scope_exit_and_unmangled_name(self):
        tree = ast.parse('class Empty:\n    "class documentation"\n')
        transformer = TypedTransformer()
        transformer.visit(tree)
        ast.fix_missing_locations(tree)
        compile(tree, '<empty-class>', 'exec')
        self.assertIn('exit_scope', ast.unparse(tree))
        transformer.class_name = '_'
        self.assertEqual(transformer.attribute_name('__secret'), '__secret')

    def test_bare_return_annotation_only_and_named_expression(self):
        tree = ast.parse('''def answer() -> int:
    value: int
    if (selected := 1):
        return
''')
        transformed = TypedTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        compile(transformed, 'test.py', 'exec')
        self.assertIn('returned', ast.unparse(transformed))
        self.assertIn('assignment', ast.unparse(transformed))

    def test_expected_type_flows_through_conditional_expression(self):
        blocks = {'first': SimpleNamespace(output_type=None),
                  'second': SimpleNamespace(output_type=None)}
        visitor = ExpectedTypes(blocks)
        expression = ast.parse(
            f'{RUNTIME_NAME}.execute("first") if True else {RUNTIME_NAME}.execute("second")',
            mode='eval').body
        visitor.apply(expression, 'int')
        self.assertEqual([block.output_type for block in blocks.values()], ['int', 'int'])


if __name__ == '__main__':
    unittest.main()
