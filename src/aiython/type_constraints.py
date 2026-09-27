"""Shared type contracts for Python boundaries, AI context and returned values.

Annotation syntax is interpreted, never eval'ed. Unsupported contracts fail closed.
This checks live values; it does not coerce values or roll back mutations.
"""
from __future__ import annotations

import ast
import builtins
import collections.abc as abc
from dataclasses import dataclass, field
import dataclasses
import enum
from functools import lru_cache
import inspect
import sys
import types
import typing
from typing_extensions import ReadOnly, TypeAliasType

try:
    import annotationlib
except ModuleNotFoundError:  # Python 3.11–3.13
    annotationlib = None

from .capabilities import CapabilityError

VALIDATORS = {}
PRIMITIVE_KINDS = frozenset(('str', 'int', 'float', 'bool', 'bytes', 'complex'))
TYPE_ALIAS_TYPES = tuple({TypeAliasType, getattr(typing, "TypeAliasType", TypeAliasType)})
READ_ONLY_TYPES = tuple({ReadOnly, getattr(typing, "ReadOnly", ReadOnly)})


class TypeViolation(CapabilityError, TypeError):
    """A declared contract was violated; no automatic coercion was performed."""


class UnsupportedType(CapabilityError, TypeError):
    """A declared type cannot be checked without weakening its contract."""


def register_validator(cls, validator):
    VALIDATORS[cls] = validator


def annotations_of(target):
    if annotationlib is not None:
        return annotationlib.get_annotations(target, format=annotationlib.Format.STRING)
    return inspect.get_annotations(target, eval_str=False)


@dataclass
class Contract:
    kind: str
    name: str
    args: tuple = ()
    python_type: object = None
    fields: dict = field(default_factory=dict)
    required: set = field(default_factory=set)
    description: str = ''
    qualifier: str | None = None

    @property
    def marker(self):
        if self.qualifier: return self.qualifier
        if self.kind in ('annotated','alias') and self.args: return self.args[0].marker
        return None

    def schema(self, seen=None):
        seen = set() if seen is None else seen
        if id(self) in seen:
            return {'x-python-type': self.name, 'x-recursive': True}
        seen = seen | {id(self)}
        kind = self.kind
        if kind == 'any': result = {}
        elif kind == 'never': result = {'not': {}}
        elif kind == 'union': result = {'anyOf': [a.schema(seen) for a in self.args]}
        elif kind == 'literal':
            if any(isinstance(v,bytes) for v in self.args):
                return {'x-python-type':self.name, 'x-python-literals':[repr(v) for v in self.args]}
            result = {'enum': [v.value if isinstance(v,enum.Enum) else v for v in self.args]}
            members = [type(v).__qualname__ + '.' + v.name for v in self.args if isinstance(v,enum.Enum)]
            if members: result['x-python-enum-members'] = members
        elif kind == 'null': result = {'type': 'null'}
        elif kind == 'callable': result = {'x-python-callable': True}
        elif kind in ('str', 'int', 'float', 'bool'):
            result = {'type': {'str':'string', 'int':'integer', 'float':'number', 'bool':'boolean'}[kind]}
        elif kind in ('list', 'set', 'frozenset', 'sequence'):
            result = {'type': 'array', 'items': self.args[0].schema(seen)}
        elif kind == 'tuple':
            result = {'type': 'array', 'prefixItems': [a.schema(seen) for a in self.args],
                      'minItems': len(self.args), 'maxItems': len(self.args)}
        elif kind == 'tuple_unpacked':
            pivot = next(index for index, item in enumerate(self.args) if item.kind == 'unpack_any')
            result = {'type': 'array', 'minItems': len(self.args) - 1,
                      'prefixItems': [item.schema(seen) for item in self.args[:pivot]],
                      'x-python-suffixItems': [item.schema(seen) for item in self.args[pivot + 1:]]}
        elif kind == 'tuple_many': result = {'type':'array', 'items':self.args[0].schema(seen)}
        elif kind in ('dict', 'mapping'):
            result = {'type':'object', 'additionalProperties':self.args[1].schema(seen), 'x-key-schema':self.args[0].schema(seen)}
        elif kind == 'class' and isinstance(self.python_type,type) and issubclass(self.python_type,enum.Enum):
            values = [v.value for v in self.python_type]
            if any(type(v) not in (str,int,bool,float,type(None)) for v in values):
                raise UnsupportedType('Enum schema requires scalar member values')
            result = {'enum': values, 'x-python-enum-members': list(self.python_type.__members__)}
        elif kind in ('typeddict', 'class') and self.fields:
            result = {'type':'object', 'properties':{k:v.schema(seen) for k,v in self.fields.items()},
                      'required': sorted(self.required)}
        elif kind in ('annotated', 'qualifier', 'alias'):
            result = self.args[0].schema(seen)
        elif kind == 'typevar':
            result = {'anyOf':[a.schema(seen) for a in self.args]} if self.args else {}
        else: result = {}
        result = {**result, 'x-python-type': self.name}
        if self.description: result['description'] = self.description
        return result

    def validate(self, value, path='value', *, bindings=None, seen=None):
        # Scalars cannot contain cycles or bind type variables. Keep the common
        # successful path free of sets, closures and diagnostic strings.
        kind = self.kind
        if kind == 'any' or (kind == 'null' and value is None):
            return
        if kind == 'callable':
            if not callable(value):
                raise TypeViolation(f'{path}: expected {self.name}, got {type(value).__name__}')
            return
        if kind in PRIMITIVE_KINDS and type(value) is self.python_type:
            return
        bindings = {} if bindings is None else bindings
        seen = set() if seen is None else seen
        pair = (id(self), id(value))
        if pair in seen: return
        seen = seen | {pair}
        def fail(detail=None):
            raise TypeViolation(f'{path}: expected {self.name}, got {type(value).__name__}' + (f' ({detail})' if detail else ''))
        def child(contract, item, suffix):
            contract.validate(item, path + suffix, bindings=bindings, seen=seen)
        if kind in ('alias', 'annotated', 'qualifier'):
            child(self.args[0], value, '')
        elif kind == 'never': fail('this boundary must not return')
        elif kind == 'union':
            for contract in self.args:
                branch = dict(bindings)
                try: contract.validate(value, path, bindings=branch, seen=seen)
                except TypeViolation: continue
                bindings.update(branch)
                return
            fail()
        elif kind == 'literal':
            if not any(type(value) is type(v) and value == v for v in self.args): fail('not an allowed literal')
        elif kind == 'null':
            if value is not None: fail()
        elif kind in ('str','int','float','bool','bytes','complex'):
            if type(value) is not self.python_type: fail()
        elif kind in ('list','set','frozenset','sequence','tuple_many','tuple'):
            expected = {'list':list,'set':set,'frozenset':frozenset,'tuple':tuple,'tuple_many':tuple}.get(kind)
            if expected is not None and type(value) is not expected: fail()
            if kind == 'sequence' and type(value) not in (list,tuple,str,bytes,range):
                fail('only non-consuming concrete sequences can be checked')
            if kind == 'tuple' and len(value) != len(self.args): fail('wrong tuple length')
            if kind != 'tuple' and self.args[0].kind in PRIMITIVE_KINDS:
                member = self.args[0]
                expected_type = member.python_type
                for index, item in enumerate(value):
                    if type(item) is not expected_type:
                        member.validate(item, f'{path}[{index}]', bindings=bindings, seen=seen)
                return
            for index, item in enumerate(value):
                child(self.args[index] if kind == 'tuple' else self.args[0], item, f'[{index}]')
        elif kind == 'tuple_unpacked':
            if type(value) is not tuple: fail()
            pivot = next(index for index, item in enumerate(self.args) if item.kind == 'unpack_any')
            suffix = len(self.args) - pivot - 1
            if len(value) < pivot + suffix: fail('wrong tuple length')
            for index, contract in enumerate(self.args[:pivot]):
                child(contract, value[index], f'[{index}]')
            for index, contract in enumerate(self.args[pivot + 1:]):
                position = len(value) - suffix + index
                child(contract, value[position], f'[{position}]')
        elif kind in ('dict','mapping'):
            if type(value) is not dict: fail('a concrete dict is required for deep checking')
            for index, (key,item) in enumerate(value.items()):
                child(self.args[0],key,f'.keys[{index}]')
                child(self.args[1],item,f'[{key!r}]' if type(key) in (str,int) else f'.values[{index}]')
        elif kind == 'typeddict':
            if type(value) is not dict: fail()
            missing = self.required - value.keys()
            if missing: fail('missing fields: ' + ', '.join(sorted(missing)))
            for key, contract in self.fields.items():
                if key in value: child(contract,value[key],f'[{key!r}]')
        elif kind == 'class':
            validator = VALIDATORS.get(self.python_type)
            if not (validator and getattr(self.python_type,'_is_protocol',False)) and not isinstance(value,self.python_type): fail()
            if validator is not None:
                if not validator(value): fail('custom validator rejected value')
                return
            for name, contract in self.fields.items():
                try: item = object.__getattribute__(value,name)
                except AttributeError:
                    if dataclasses.is_dataclass(self.python_type): fail(f'missing attribute {name}')
                    continue
                child(contract,item,'.'+name)
        elif kind == 'type':
            if not isinstance(value,type): fail()
            target = self.args[0]
            if target.kind != 'any' and (target.python_type is None or not issubclass(value,target.python_type)):
                fail()
        elif kind in ('generator','async_generator'):
            raise UnsupportedType('Lazy iterable contracts must be checked at yield/send boundaries, not by consuming the object')
        elif kind == 'typevar':
            if self.args:
                Contract('union',self.name,self.args).validate(value,path,bindings=bindings,seen=seen)
            previous = bindings.get(self.python_type)
            if previous is not None and type(value) is not previous: fail('inconsistent TypeVar binding')
            bindings[self.python_type] = type(value)
        else:
            raise UnsupportedType(f'Unsupported contract {self.name}')


class Compiler:
    def __init__(self, namespace):
        self.names = dict(vars(builtins)) | vars(typing) | {"typing": typing} | dict(namespace)
        self.cache = {}

    def lookup(self, node, names):
        if isinstance(node,ast.Name):
            if node.id not in names:
                raise UnsupportedType(f'Unresolved output type: {node.id}')
            return names[node.id]
        if isinstance(node,ast.Attribute):
            owner = self.lookup(node.value,names)
            if not isinstance(owner,(types.ModuleType,type)):
                raise UnsupportedType('Annotation attributes must reference a module or class')
            return vars(owner)[node.attr]
        raise UnsupportedType('Unsupported output type expression; annotation calls are not executed')

    def compile(self, annotation, names=None):
        names = self.names if names is None else names
        if isinstance(annotation,Contract):
            return annotation
        if isinstance(annotation,typing.ForwardRef): annotation = annotation.__forward_arg__
        if isinstance(annotation,str): return self.node(annotation_node(annotation),names)
        return self.value(annotation,names)

    def node(self,node,names):
        if isinstance(node,ast.Constant):
            if node.value is None: return Contract('null','None')
            if isinstance(node.value,str): return self.compile(node.value,names)
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitOr):
            return Contract('union',ast.unparse(node),(self.node(node.left,names),self.node(node.right,names)))
        if isinstance(node, ast.Starred) and isinstance(node.value, ast.Name):
            parameter = names.get(node.value.id)
            if isinstance(parameter, typing.TypeVarTuple):
                return Contract('unpack_any', ast.unparse(node), python_type=parameter)
        if isinstance(node,ast.Subscript):
            base = self.lookup(node.value,names)
            nodes = node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
            if base in (typing.Callable, abc.Callable):
                if len(nodes) != 2:
                    raise UnsupportedType('Callable requires parameters and a return type')
                return Contract('callable', ast.unparse(node))
            if (base is typing.Unpack and len(nodes) == 1 and isinstance(nodes[0], ast.Name)
                    and isinstance(names.get(nodes[0].id), typing.TypeVarTuple)):
                return Contract('unpack_any', ast.unparse(node), python_type=names[nodes[0].id])
            if base is typing.Literal:
                values = tuple(self.lookup(n,names) if isinstance(n,ast.Attribute) else ast.literal_eval(n) for n in nodes)
                if any(type(v) not in (str,int,bool,bytes,type(None)) and not isinstance(v,enum.Enum) for v in values):
                    raise UnsupportedType('Unsupported Literal value')
                return Contract('literal',ast.unparse(node),values)
            if base is typing.Annotated:
                metadata = [ast.literal_eval(n) for n in nodes[1:]]
                return Contract('annotated',ast.unparse(node),(self.node(nodes[0],names),),
                                description='; '.join(v for v in metadata if isinstance(v,str)))
            args = tuple(Ellipsis if isinstance(n,ast.Constant) and n.value is Ellipsis else self.node(n,names) for n in nodes)
            return self.generic(base,args,ast.unparse(node),names)
        return self.value(self.lookup(node,names),names)

    def generic(self,base,args,label,names):
        origin = typing.get_origin(base) or base
        if origin is abc.Callable:
            return Contract('callable',label)
        if origin in (typing.Union,types.UnionType):
            return Contract('union',label,args)
        if base is typing.Optional:
            return Contract('union',label,args+(Contract('null','None'),))
        if base in READ_ONLY_TYPES:
            raise UnsupportedType('ReadOnly needs mutation interception; it cannot be silently reduced to a value type')
        qualifiers = (typing.Required,typing.NotRequired,typing.Final,typing.ClassVar)
        if base in qualifiers:
            if len(args) != 1:
                raise UnsupportedType('Qualifier requires one type')
            return Contract('qualifier',label,args,qualifier=base._name)
        containers = {list:'list',set:'set',frozenset:'frozenset',dict:'dict',abc.Sequence:'sequence',abc.Mapping:'mapping'}
        if origin in containers:
            expected = 2 if origin in (dict,abc.Mapping) else 1
            if len(args) != expected:
                raise UnsupportedType(f'{label}: wrong number of type parameters')
            return Contract(containers[origin],label,args)
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis: return Contract('tuple_many',label,args[:1])
            unpacked = [index for index, arg in enumerate(args)
                        if isinstance(arg, Contract) and arg.kind == 'unpack_any']
            if len(unpacked) == 1:
                if len(args) == 1: return Contract('tuple_many',label,(Contract('any','Any'),))
                return Contract('tuple_unpacked',label,args)
            if unpacked:
                raise UnsupportedType('Only one variadic tuple parameter can be checked')
            return Contract('tuple',label,args)
        if origin in (abc.Generator,abc.Iterator,abc.Iterable,abc.AsyncGenerator,abc.AsyncIterator,abc.AsyncIterable):
            async_kind = origin in (abc.AsyncGenerator,abc.AsyncIterator,abc.AsyncIterable)
            expected = 3 if origin is abc.Generator else 2 if origin is abc.AsyncGenerator else 1
            if len(args) != expected:
                raise UnsupportedType('Wrong iterator type argument count')
            padded = args + (Contract('null','None'),)*(3-len(args))
            return Contract('async_generator' if async_kind else 'generator',label,padded)
        if origin is type:
            if len(args) != 1:
                raise UnsupportedType('type requires one parameter')
            return Contract('type',label,args)
        if isinstance(base,TYPE_ALIAS_TYPES): return self.alias(base,names,args,label)
        if isinstance(base,type) and (getattr(base,'__type_params__',()) or getattr(base,'__parameters__',())):
            parameters = getattr(base,'__type_params__',()) or base.__parameters__
            if len(parameters) != len(args):
                raise UnsupportedType('Generic type argument count mismatch')
            return self.class_contract(base,names | {p.__name__: a for p,a in zip(parameters,args)},label)
        raise UnsupportedType(f'Unsupported generic output type: {label}; no unchecked fallback is allowed')

    def alias(self,alias,names,args=(),label=None):
        parameters = alias.__type_params__
        if parameters and len(args) != len(parameters):
            raise UnsupportedType('Generic alias requires its type arguments')
        scope = self.module_names(alias,names) | {p.__name__:a for p,a in zip(parameters,args)}
        key = (id(alias),tuple(id(a) for a in args))
        if key in self.cache:
            return self.cache[key]
        result = Contract('alias',label or alias.__name__)
        self.cache[key] = result
        if annotationlib is not None and hasattr(alias,'evaluate_value'):
            source = annotationlib.call_evaluate_function(alias.evaluate_value,annotationlib.Format.STRING)
        else:
            source = alias.__value__
        result.args = (self.compile(source,scope),)
        return result

    @staticmethod
    def module_names(target,names):
        module = sys.modules.get(getattr(target,'__module__',''))
        # Defining module names win over unrelated caller aliases.
        return names | (vars(module) if module else {})

    def class_contract(self,target,names,label=None):
        key = (id(target),label or target.__qualname__)
        if key in self.cache:
            return self.cache[key]
        if getattr(target,'_is_protocol',False) and target not in VALIDATORS:
            raise UnsupportedType('Protocol requires a registered validator; structural call signatures cannot be proven by isinstance')
        record = typing.is_typeddict(target)
        result = Contract('typeddict' if record else 'class',label or target.__qualname__,python_type=target)
        self.cache[key] = result
        if target in VALIDATORS: return result
        scope = self.module_names(target,names) | {target.__name__:target}
        scope.update({p.__name__:p for p in (getattr(target,'__type_params__',()) or getattr(target,'__parameters__',()))})
        # isinstance can call a user's __getattribute__('__class__') here.
        scope.update({k:v for k,v in names.items() if issubclass(type(v), Contract)})
        fields = {}
        for base in reversed(target.__mro__):
            if base in (object,dict): continue
            fields.update(annotations_of(base))
        for name, source in fields.items():
            contract = self.compile(source,scope)
            if contract.marker == 'ClassVar':
                continue
            result.fields[name] = contract
        if record:
            result.required = set(target.__required_keys__)
            for name, contract in result.fields.items():
                if contract.marker == 'Required':
                    result.required.add(name)
                elif contract.marker == 'NotRequired':
                    result.required.discard(name)
        else:
            result.required = set(result.fields)
        return result

    def value(self,target,names):
        if isinstance(target,Contract): return target
        if target is None or target is type(None): return Contract('null','None')
        if target is typing.Any: return Contract('any','Any')
        if target in (typing.Callable, abc.Callable):
            return Contract('callable',str(target))
        if target in (typing.Final,typing.ClassVar):
            return Contract('qualifier',str(target),(Contract('any','Any'),),qualifier=target._name)
        if target in (typing.Never,typing.NoReturn): return Contract('never',str(target))
        if target is typing.Self:
            owner = names.get('self',names.get('cls'))
            if owner is None:
                raise UnsupportedType('Self requires an instance or class scope')
            return self.value(owner if isinstance(owner,type) else type(owner),names)
        if target is typing.LiteralString:
            raise UnsupportedType('LiteralString requires static provenance checking; use str for a runtime string contract')
        if isinstance(target,TYPE_ALIAS_TYPES): return self.alias(target,names)
        if isinstance(target,typing.TypeVar):
            substituted = names.get(target.__name__)
            if isinstance(substituted,Contract):
                return substituted
            choices = target.__constraints__ or ((target.__bound__,) if target.__bound__ else ())
            return Contract('typevar',target.__name__,tuple(self.compile(v,names) for v in choices),python_type=target)
        if isinstance(target,typing.NewType): return self.compile(target.__supertype__,names)
        if isinstance(target,(str,typing.ForwardRef)):
            return self.compile(target,names)
        origin,args = typing.get_origin(target),typing.get_args(target)
        if origin is typing.Literal: return Contract('literal',str(target),args)
        if origin is typing.Annotated:
            return Contract('annotated',str(target),(self.compile(args[0],names),),description='; '.join(v for v in args[1:] if isinstance(v,str)))
        if origin is not None:
            return self.generic(origin,tuple(Ellipsis if a is Ellipsis else self.compile(a,names) for a in args),str(target),names)
        if target in (int,str,float,bool,bytes,complex): return Contract(target.__name__,target.__name__,python_type=target)
        if target in (list,set,frozenset,dict,tuple):
            any_type = Contract('any','Any')
            if target is tuple: return Contract('tuple_many','tuple',(any_type,))
            return self.generic(target,(any_type,any_type) if target is dict else (any_type,),target.__name__,names)
        if not isinstance(target,type):
            raise UnsupportedType('Annotation is not a supported Python type')
        return self.class_contract(target,names)


@lru_cache(maxsize=512)
def annotation_node(annotation):
    # Compiler.node only reads this tree. Namespace resolution still happens on
    # every compile, including forward references and mutable class annotations.
    return ast.parse(annotation, mode='eval').body


class ContractCache:
    """Bounded cache for contracts whose dependencies are immutable builtins.

    Classes, aliases, TypeVars, attributes and string forward references always
    take the dynamic path. Keys include resolved identities, never just text.
    """
    def __init__(self):
        from collections import OrderedDict
        self.entries = OrderedDict()

    @staticmethod
    @lru_cache(maxsize=512)
    def names(annotation):
        node = annotation_node(annotation)
        nodes = tuple(ast.walk(node))
        if any(isinstance(n, ast.Attribute) or
               isinstance(n, ast.Constant) and isinstance(n.value, str) for n in nodes):
            return None
        return tuple(sorted({n.id for n in nodes if isinstance(n, ast.Name)}))

    def compile(self, annotation, namespace):
        key = None
        if type(annotation) is str:
            try:
                names = self.names(annotation)
            except SyntaxError:
                names = None
            if names is not None:
                allowed = (int, str, float, bool, bytes, complex, list, tuple,
                           dict, set, frozenset, typing.Any, typing.Optional,
                           typing.Union, typing.Literal, typing.Final,
                           typing.ClassVar, typing.Required, typing.NotRequired)
                resolved = tuple(namespace.get(name, vars(typing).get(name, vars(builtins).get(name)))
                                 for name in names)
                if all(any(value is item for item in allowed) for value in resolved):
                    key = (annotation, tuple(id(value) for value in resolved))
        if key is not None and key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        contract = compile_contract(annotation, namespace)
        if key is not None:
            self.entries[key] = contract
            if len(self.entries) > 256:
                self.entries.popitem(last=False)
        return contract


def compile_contract(annotation,namespace):
    try:
        return Compiler(namespace).compile(annotation)
    except UnsupportedType:
        raise
    except (SyntaxError,KeyError,TypeError,ValueError,AttributeError) as exc:
        raise UnsupportedType(f'Cannot resolve type contract safely ({type(exc).__name__})') from None


def describe_output(annotation,namespace):
    return compile_contract(annotation,namespace).schema() if annotation else None


def validate_output(value,annotation,namespace):
    if annotation:
        compile_contract(annotation,namespace).validate(value,'output')
