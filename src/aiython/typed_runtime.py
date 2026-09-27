"""Annotation-aware AST boundaries; retains live Python values and identity."""
from __future__ import annotations

import ast
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import threading
import typing
import weakref

from .frontend import RUNTIME_NAME
from .type_constraints import (Contract, ContractCache, TypeViolation, compile_contract,
                               annotations_of, Compiler, SELF_OWNER)

SCOPE = '__aiython_type_scope__'
_FRAME_SCOPES = ContextVar('aiython_frame_scopes', default=())
_GENERATOR_SCOPES = {}
_GENERATOR_SCOPES_LOCK = threading.RLock()


def frame_scope(frame):
    for active, scope in reversed(_FRAME_SCOPES.get()):
        if active is frame:
            return scope
    with _GENERATOR_SCOPES_LOCK:
        return _GENERATOR_SCOPES.get(id(frame))


def original_class(value):
    # Bypass user __getattribute__ while reading typing's instance metadata.
    try:
        return object.__getattribute__(value, '__orig_class__')
    except AttributeError:
        return type(value)


def unconstrained_variadic(source, mode, namespace):
    if mode == 'args' and source.startswith('*'):
        return True
    parameter_name, separator, attribute = source.partition('.')
    if (mode in ('args', 'kwargs') and separator and attribute == mode and
            isinstance(namespace.get(parameter_name), typing.ParamSpec)):
        return True
    if mode != 'args':
        return False
    annotation = ast.parse(source, mode='eval').body
    if not isinstance(annotation, ast.Subscript) or not isinstance(annotation.slice, ast.Name):
        return False
    unpack = annotation.value
    is_unpack = ((isinstance(unpack, ast.Name) and namespace.get(unpack.id) is typing.Unpack) or
                 (isinstance(unpack, ast.Attribute) and unpack.attr == 'Unpack' and
                  isinstance(unpack.value, ast.Name) and namespace.get(unpack.value.id) is typing))
    return is_unpack and isinstance(namespace.get(annotation.slice.id), typing.TypeVarTuple)


@dataclass
class Scope:
    declarations: dict = field(default_factory=dict)
    contracts: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    final_names: set = field(default_factory=set)
    returned: Contract | None = None
    has_return: bool = False
    return_value: object = None
    failed: bool = False
    recovery_counts: dict = field(default_factory=dict)


class TypeRuntime:
    def __init__(self, manager=None):
        self.manager = manager
        self.classes = weakref.WeakSet()
        self._classes_lock = threading.RLock()
        self._contract_cache = threading.local()

    def contract(self, annotation, namespace):
        cache = getattr(self._contract_cache, 'value', None)
        if cache is None:
            cache = self._contract_cache.value = ContractCache()
        return cache.compile(annotation, namespace)

    def describe_output(self, annotation, frame):
        return self.contract(annotation, self.namespace(frame)).schema() if annotation else None

    def validate_output(self, value, annotation, frame):
        if annotation:
            self.contract(annotation, self.namespace(frame)).validate(value, 'output')

    def register_class(self, cls):
        if isinstance(cls, type):
            with self._classes_lock:
                self.classes.add(cls)
        return cls

    @staticmethod
    def namespace(frame):
        namespace = dict(frame.f_globals) | dict(frame.f_locals)
        scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
        if isinstance(scope,Scope):
            namespace.update({parameter.__name__: bound for parameter,bound in scope.bindings.items()})
        return namespace

    def _initialize(self, frame, declarations, parameters, returns):
        scope = Scope(declarations=declarations)
        namespace = self.namespace(frame)
        for name, (source, mode) in (parameters or {}).items():
            # Unbound variadic type parameters have no concrete element types
            # to check at this call boundary.
            contract = (Contract('any', 'Any') if unconstrained_variadic(source, mode, namespace)
                        else self.contract(source,namespace))
            if mode == 'args': contract = Contract('tuple_many',source,(contract,))
            elif mode == 'kwargs':
                contract = (contract.args[0] if contract.kind == 'unpack_typeddict' else
                            Contract('dict',source,(compile_contract('str',namespace),contract)))
            contract.validate(frame.f_locals[name],name,bindings=scope.bindings)
            scope.contracts[name] = contract
        if returns:
            scope.returned = self.contract(returns,namespace)
        return scope

    def enter_scope(self, declarations, parameters=None, returns=None):
        frame = inspect.currentframe().f_back
        try:
            scope = self._initialize(frame, declarations, parameters, returns)
            _FRAME_SCOPES.set(_FRAME_SCOPES.get() + ((frame, scope),))
        finally:
            del frame

    def exit_scope(self):
        stack = _FRAME_SCOPES.get()
        _FRAME_SCOPES.set(stack[:-1])

    def enter_generator_scope(self, declarations, parameters=None, returns=None):
        frame = inspect.currentframe().f_back
        try:
            scope = self._initialize(frame, declarations, parameters, returns)
            with _GENERATOR_SCOPES_LOCK:
                _GENERATOR_SCOPES[id(frame)] = scope
        finally:
            del frame

    def exit_generator_scope(self):
        frame = inspect.currentframe().f_back
        try:
            with _GENERATOR_SCOPES_LOCK:
                _GENERATOR_SCOPES.pop(id(frame), None)
        finally:
            del frame

    @staticmethod
    def scopes(frame):
        local = frame_scope(frame) or frame.f_locals.get(SCOPE)
        global_scope = frame.f_globals.get(SCOPE)
        scopes = [(local,frame.f_locals)] if isinstance(local,Scope) else []
        if isinstance(global_scope,Scope) and global_scope is not local:
            scopes.append((global_scope,frame.f_globals))
        return scopes

    def assignment_in(self, frame, value, name, annotation=None):
        scopes = self.scopes(frame)
        if not scopes:
            scope = Scope()
            frame.f_locals[SCOPE] = scope
            scopes = [(scope,frame.f_locals)]
        local_binding = name in frame.f_code.co_varnames or name in frame.f_code.co_cellvars
        candidates = scopes[:1] if local_binding else scopes
        scope = next((s for s,values in candidates if name in s.declarations or name in s.contracts),candidates[0][0])
        source = annotation or scope.declarations.get(name)
        if annotation: scope.declarations[name] = annotation
        contract = scope.contracts.get(name)
        if contract is None and source:
            contract = self.contract(source,self.namespace(frame))
            scope.contracts[name] = contract
        if contract:
            if name in scope.final_names:
                raise TypeViolation(f'{name}: Final binding cannot be reassigned')
            contract.validate(value,name,bindings=scope.bindings)
            if contract.marker == 'Final': scope.final_names.add(name)
        with self._classes_lock:
            registered = type(value) in self.classes
        if registered:
            compile_contract(original_class(value),self.namespace(frame)).validate(value,name)
        return value

    def assignment(self,value,name,annotation=None):
        frame = inspect.currentframe().f_back
        try: return self.assignment_in(frame,value,name,annotation)
        finally: del frame

    def reassigning(self, names):
        frame = inspect.currentframe().f_back
        try:
            for scope,values in self.scopes(frame):
                for name in names:
                    if name in scope.final_names:
                        raise TypeViolation(f'{name}: Final binding cannot be reassigned')
        finally: del frame

    def expression(self,value,annotation):
        frame = inspect.currentframe().f_back
        try:
            self.contract(annotation,self.namespace(frame)).validate(value,'expression')
            return value
        finally: del frame

    def check_frame(self,frame):
        with self._classes_lock:
            classes = frozenset(self.classes)
        if classes:
            seen = set()
            if frame.f_code.co_name == '__init__' and 'self' in frame.f_locals:
                seen.add(id(frame.f_locals['self']))
            def check_instances(candidate):
                if id(candidate) in seen: return
                seen.add(id(candidate))
                cls = type(candidate)
                if cls in classes:
                    compile_contract(original_class(candidate),self.namespace(frame)).validate(candidate,cls.__qualname__)
                    try: state = object.__getattribute__(candidate,'__dict__')
                    except AttributeError: state = {}
                    check_instances(state)
                elif cls in (list,tuple,set,frozenset):
                    for item in candidate: check_instances(item)
                elif cls is dict:
                    for item in candidate.values(): check_instances(item)
            for namespace in (frame.f_locals,frame.f_globals):
                for name,candidate in namespace.items():
                    if not name.startswith('__'):
                        check_instances(candidate)
        for scope, values in self.scopes(frame):
            for name, source in scope.declarations.items():
                if name in values and name not in scope.contracts:
                    scope.contracts[name] = self.contract(source,self.namespace(frame))
            for name, contract in scope.contracts.items():
                if name in values:
                    contract.validate(values[name],name,bindings=scope.bindings)

    def checkpoint(self):
        frame = inspect.currentframe().f_back
        try:
            self.check_frame(frame)
            # Active enclosing scopes can hold annotated aliases to mutated values.
            parent = frame.f_back
            while parent:
                unit = self.manager.units.get(parent.f_code.co_filename) if self.manager is not None else None
                same_runtime = (any(value is self.manager for value in parent.f_code.co_consts) if unit else
                                self.manager is None and parent.f_globals.get(RUNTIME_NAME) is frame.f_globals.get(RUNTIME_NAME))
                if (frame_scope(parent) or SCOPE in parent.f_locals) and same_runtime:
                    self.check_frame(parent)
                parent = parent.f_back
        finally: del frame

    def returned(self,value):
        frame = inspect.currentframe().f_back
        try:
            self.check_frame(frame)
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            if isinstance(scope,Scope) and scope.returned:
                contract = scope.returned.args[2] if scope.returned.kind in ('generator','async_generator') and frame.f_code.co_flags & (inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR) else scope.returned
                contract.validate(value,'return',bindings=scope.bindings)
            if isinstance(scope,Scope):
                scope.has_return, scope.return_value = True, value
            return value
        finally: del frame

    def aborted(self):
        frame = inspect.currentframe().f_back
        try:
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            if isinstance(scope,Scope): scope.failed = True
        finally: del frame

    def leaving(self):
        frame = inspect.currentframe().f_back
        try:
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            if not isinstance(scope,Scope) or scope.failed:
                return
            self.check_frame(frame)
            if scope.has_return and scope.returned:
                contract = scope.returned.args[2] if scope.returned.kind in ('generator','async_generator') and frame.f_code.co_flags & (inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR) else scope.returned
                contract.validate(scope.return_value,'return',bindings=scope.bindings)
        finally: del frame

    def yielded(self,value):
        frame = inspect.currentframe().f_back
        try:
            self.check_frame(frame)
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            if isinstance(scope,Scope) and scope.returned:
                if scope.returned.kind not in ('generator','async_generator'):
                    raise TypeViolation('Generator return annotation must describe yielded values')
                scope.returned.args[0].validate(value,'yield',bindings=scope.bindings)
            return value
        finally: del frame

    def sent(self,value):
        frame = inspect.currentframe().f_back
        try:
            scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
            if isinstance(scope,Scope) and scope.returned:
                scope.returned.args[1].validate(value,'send',bindings=scope.bindings)
            return value
        finally: del frame

    def delegate(self,iterable):
        frame = inspect.currentframe().f_back
        scope = frame_scope(frame) or frame.f_locals.get(SCOPE)
        del frame
        contract = scope.returned if isinstance(scope,Scope) else None
        iterator = iter(iterable)
        # Delegation forwards send/throw/close and the terminal return value.
        def checked():
            import sys
            try: item = next(iterator)
            except StopIteration as stop: return stop.value
            while True:
                if contract:
                    contract.args[0].validate(item,'yield',bindings=scope.bindings)
                try:
                    sent = yield item
                except GeneratorExit:
                    close = getattr(iterator,'close',None)
                    if close: close()
                    raise
                except BaseException:
                    throw = getattr(iterator,'throw',None)
                    if throw is None:
                        raise
                    try: item = throw(sys.exception())
                    except StopIteration as stop: return stop.value
                else:
                    if contract:
                        contract.args[1].validate(sent,'send',bindings=scope.bindings)
                    try: item = next(iterator) if sent is None else iterator.send(sent)
                    except StopIteration as stop: return stop.value
        return checked()

    def assign_attribute(self,owner,name,value,annotation=None):
        frame = inspect.currentframe().f_back
        try:
            if annotation:
                contract = compile_contract(annotation,self.namespace(frame))
            else:
                target = owner if issubclass(type(owner), type) else type(owner)
                fields = {}
                for base in reversed(target.__mro__):
                    fields.update(annotations_of(base))
                source = fields.get(name)
                namespace = Compiler.module_names(target,self.namespace(frame))
                namespace[SELF_OWNER] = target
                namespace.update({p.__name__:p for p in (getattr(target,'__type_params__',()) or getattr(target,'__parameters__',()))})
                contract = compile_contract(source,namespace) if source else None
            if contract:
                if contract.marker == 'ClassVar' and not issubclass(type(owner), type):
                    raise TypeViolation(f'{name}: ClassVar must be assigned on the class')
                if contract.marker == 'Final':
                    try: inspect.getattr_static(owner,name)
                    except AttributeError: pass
                    else: raise TypeViolation(f'{name}: Final attribute cannot be reassigned')
                contract.validate(value,f'{type(owner).__name__}.{name}')
            setattr(owner,name,value)
        finally: del frame


def helper(name,*args,runtime_name=RUNTIME_NAME):
    return ast.Call(ast.Attribute(ast.Attribute(ast.Name(runtime_name,ast.Load()),'types',ast.Load()),name,ast.Load()),list(args),[])


def literal(value):
    return ast.parse(repr(value),mode='eval').body


class TypedTransformer(ast.NodeTransformer):
    def __init__(self, *, snippet=False, runtime_name=RUNTIME_NAME):
        self.snippet = snippet
        self.runtime_name = runtime_name
        self.declarations = {}
        self.function = False
        self.delegation_contract = False

    def helper(self, name, *args):
        return helper(name, *args, runtime_name=self.runtime_name)

    @staticmethod
    def declarations_in(body):
        result = {}
        def collect(node):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)): return
            if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
                result[node.target.id] = ast.unparse(node.annotation)
            for child in ast.iter_child_nodes(node): collect(child)
        for statement in body: collect(statement)
        return result

    def body(self,body,*,parameters=None,returns=None,initialize=True,inherited=None,generator_scope=False):
        previous = self.declarations
        self.declarations = dict(inherited or {}) | self.declarations_in(body)
        output = []
        header = []
        if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
            header.append(body[0]); body = body[1:]
        while body and isinstance(body[0],ast.ImportFrom) and body[0].module == '__future__':
            header.append(body[0]); body = body[1:]
        if initialize:
            initial = ast.Expr(self.helper('enter_generator_scope' if generator_scope else 'enter_scope',
                                      literal(self.declarations),literal(parameters),literal(returns)))
            ast.copy_location(initial,body[0] if body else header[-1] if header else ast.Constant(None,lineno=1,col_offset=0))
            initial._aiython_scope_initializer = True
            output.append(initial)
        # Capture lexical types used only in stringified contracts without executing them.
        names = set()
        for source in list(self.declarations.values()) + [p[0] for p in (parameters or {}).values()] + ([returns] if returns else []):
            try:
                annotation = ast.parse(source, mode='eval')
            except SyntaxError:
                # Variadic parameter annotations such as *args: *Ts need a
                # subscription context; their unparsed text is not an expression.
                annotation = ast.parse(f'tuple[{source}]', mode='eval')
            names.update(n.id for n in ast.walk(annotation) if isinstance(n,ast.Name))
        if self.function and names:
            capture = ast.If(ast.Constant(False),[ast.Expr(ast.Tuple([ast.Name(n,ast.Load()) for n in sorted(names)],ast.Load()))],[])
            ast.copy_location(capture,body[0] if body else output[0])
            output.append(capture)
        for statement in body:
            transformed = self.visit(statement)
            output.extend(transformed if isinstance(transformed,list) else [transformed])
            if not isinstance(statement,(ast.Return,ast.Raise,ast.Break,ast.Continue)):
                output.append(ast.copy_location(ast.Expr(self.helper('checkpoint')),statement))
        self.declarations = previous
        return header+output

    def visit_Module(self,node):
        if self.snippet:
            node.body = self.body(node.body,initialize=False)
            return node
        body = self.body(node.body)
        initial = next(i for i,item in enumerate(body) if getattr(item,'_aiython_scope_initializer',False))
        header, enter, statements = body[:initial], body[initial], body[initial+1:]
        exit_call = ast.copy_location(ast.Expr(self.helper('exit_scope')), enter)
        if statements:
            guard = ast.copy_location(ast.Try(statements, [], [], [exit_call]), enter)
            guard._aiython_module_guard = True
            node.body = header + [enter, guard]
        else:
            node.body = header + [enter, exit_call]
        return node

    def visit_FunctionDef(self,node):
        previous = self.function
        previous_contract = self.delegation_contract
        parent_declarations = dict(self.declarations) if previous else {}
        self.function = True
        parameters = {}
        for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
            if arg.annotation: parameters[arg.arg] = (ast.unparse(arg.annotation),'value')
        for arg, mode in ((node.args.vararg,'args'),(node.args.kwarg,'kwargs')):
            if arg and arg.annotation: parameters[arg.arg] = (ast.unparse(arg.annotation),mode)
        returns = ast.unparse(node.returns) if node.returns else None
        self.delegation_contract = returns is not None
        # Generator annotations require yield/send checks, not return-only checks.
        is_generator = any(isinstance(n,(ast.Yield,ast.YieldFrom)) for n in self.function_nodes(node))
        used = {n.id for n in self.function_nodes(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        nonlocal_names = {name for n in self.function_nodes(node) if isinstance(n,ast.Nonlocal) for name in n.names}
        assigned = {n.id for n in self.function_nodes(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)} - nonlocal_names
        inherited = {name:source for name,source in parent_declarations.items() if name in (used|nonlocal_names) and name not in assigned and name not in parameters}
        node.body = self.body(node.body,parameters=parameters,returns=returns,
                              inherited=inherited,generator_scope=is_generator)
        if isinstance(node,ast.AsyncFunctionDef) and is_generator:
            node.body.append(ast.copy_location(ast.Expr(self.helper('returned',ast.Constant(None))),node))
        else:
            node.body.append(ast.copy_location(ast.Return(self.helper('returned',ast.Constant(None))),node))
        initial = next(i for i,n in enumerate(node.body) if getattr(n,'_aiython_scope_initializer',False))
        handler = ast.ExceptHandler(ast.Attribute(ast.Name(self.runtime_name,ast.Load()),'scope_error_type',ast.Load()),None,
                                    [ast.Expr(self.helper('aborted')),ast.Raise()])
        final = ast.Expr(self.helper('leaving'))
        final = ast.Try([final],[],[],[ast.Expr(self.helper('exit_generator_scope' if is_generator else 'exit_scope'))])
        guarded = ast.Try(node.body[initial+1:],[handler],[],[final])
        guarded._aiython_type_guard = True
        ast.copy_location(guarded,node)
        node.body = node.body[:initial+1] + [guarded]
        self.function = previous
        self.delegation_contract = previous_contract
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    @staticmethod
    def function_nodes(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda,ast.ClassDef)): continue
            yield child
            yield from TypedTransformer.function_nodes(child)

    def visit_ClassDef(self,node):
        node.decorator_list.insert(0, ast.Attribute(ast.Attribute(ast.Name(self.runtime_name,ast.Load()),'types',ast.Load()),'register_class',ast.Load()))
        previous = self.function
        previous_contract = self.delegation_contract
        self.function = False
        self.delegation_contract = False
        body = self.body(node.body)
        header = body[:1] if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str) else []
        enter, *statements = body[len(header):]
        exit_call = ast.copy_location(ast.Expr(self.helper('exit_scope')), node)
        if statements:
            guard = ast.copy_location(ast.Try(statements, [], [], [exit_call]), node)
            guard._aiython_class_guard = True
            node.body = header + [enter, guard]
        else:
            node.body = header + [enter, exit_call]
        self.function = previous
        self.delegation_contract = previous_contract
        return node

    def visit_Lambda(self,node):
        previous_contract = self.delegation_contract
        self.delegation_contract = False
        node = self.generic_visit(node)
        self.delegation_contract = previous_contract
        return node

    def visit_Return(self,node):
        if node.value is None:
            return [ast.copy_location(ast.Expr(self.helper('returned',ast.Constant(None))),node),node]
        node.value = self.helper('returned',self.visit(node.value))
        return node

    def visit_Yield(self,node):
        value = self.visit(node.value) if node.value else ast.Constant(None)
        node.value = self.helper('yielded',value)
        return ast.copy_location(self.helper('sent',node),node)

    def visit_YieldFrom(self,node):
        node.value = self.visit(node.value)
        if self.delegation_contract:
            node.value = self.helper('delegate',node.value)
        return node

    def visit_AnnAssign(self,node):
        if node.value is None: return node
        annotation = ast.unparse(node.annotation)
        if isinstance(node.target,ast.Name):
            node.value = self.helper('assignment',self.visit(node.value),ast.Constant(node.target.id),ast.Constant(annotation))
            return node
        node.value = self.helper('expression',self.visit(node.value),ast.Constant(annotation))
        return node

    def visit_Assign(self,node):
        node.value = self.visit(node.value)
        if len(node.targets) == 1 and isinstance(node.targets[0],ast.Attribute):
            target = node.targets[0]
            assign = ast.Call(
                ast.Attribute(ast.Attribute(ast.Name(self.runtime_name,ast.Load()),'types',ast.Load()),
                              'assign_attribute',ast.Load()), [], [
                    ast.keyword(arg='value',value=node.value),
                    ast.keyword(arg='owner',value=self.visit(target.value)),
                    ast.keyword(arg='name',value=ast.Constant(target.attr)),
                ])
            return ast.copy_location(ast.Expr(assign),node)
        for target in node.targets:
            if isinstance(target,ast.Name):
                node.value = self.helper('assignment',node.value,ast.Constant(target.id))
        return node

    def visit_AugAssign(self,node):
        names = [node.target.id] if isinstance(node.target,ast.Name) else []
        return [ast.copy_location(ast.Expr(self.helper('reassigning',literal(names))),node),node]

    def visit_NamedExpr(self,node):
        node.value = self.helper('assignment',self.visit(node.value),ast.Constant(node.target.id))
        return node

    def visit_If(self,node):
        node.test = self.visit(node.test)
        node.body = self.nested(node.body)
        node.orelse = self.nested(node.orelse)
        return node

    def nested(self,body):
        output = []
        for statement in body:
            result = self.visit(statement)
            output.extend(result if isinstance(result,list) else [result])
            if not isinstance(statement,(ast.Return,ast.Raise,ast.Break,ast.Continue)):
                output.append(ast.copy_location(ast.Expr(self.helper('checkpoint')),statement))
        return output

    def visit_For(self,node):
        node.iter = self.visit(node.iter)
        node.body = [ast.copy_location(ast.Expr(self.helper('checkpoint')),node)] + self.nested(node.body)
        node.orelse = self.nested(node.orelse)
        return node
    visit_AsyncFor = visit_For

    def visit_While(self,node):
        node.test = self.visit(node.test)
        node.body = self.nested(node.body); node.orelse = self.nested(node.orelse)
        return node

    def visit_With(self,node):
        node.items = [self.visit(item) for item in node.items]
        node.body = [ast.copy_location(ast.Expr(self.helper('checkpoint')),node)] + self.nested(node.body)
        return node
    visit_AsyncWith = visit_With

    def visit_Try(self,node):
        node.body = self.nested(node.body)
        node.orelse = self.nested(node.orelse); node.finalbody = self.nested(node.finalbody)
        for handler in node.handlers: handler.body = self.nested(handler.body)
        return node
    visit_TryStar = visit_Try


class ExpectedTypes(ast.NodeVisitor):
    """Propagate declared contracts to direct AI values before code generation."""
    def __init__(self,blocks,runtime_name=RUNTIME_NAME):
        self.blocks = blocks
        self.runtime_name = runtime_name
        self.declarations = {}
        self.returns = None
        self.functions = {}
        self.yields = None

    @staticmethod
    def signatures(body):
        return {node.name: node for node in body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}

    def apply(self,node,annotation):
        if node is None or not annotation: return
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id == self.runtime_name
            and node.func.attr == 'execute' and node.args and isinstance(node.args[0],ast.Constant)):
            self.blocks[node.args[0].value].output_type = annotation
        elif isinstance(node,ast.IfExp):
            self.apply(node.body,annotation); self.apply(node.orelse,annotation)

    def visit_Module(self,node):
        self.declarations = TypedTransformer.declarations_in(node.body)
        self.functions = self.signatures(node.body)
        self.generic_visit(node)

    def visit_FunctionDef(self,node):
        previous, returned, functions, yielded = self.declarations, self.returns, self.functions, self.yields
        self.functions = dict(functions) | self.signatures(node.body)
        local_names = {n.id for n in TypedTransformer.function_nodes(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        external = {name for n in TypedTransformer.function_nodes(node) if isinstance(n,(ast.Global,ast.Nonlocal)) for name in n.names}
        self.declarations = {name:value for name,value in previous.items() if name not in local_names or name in external} | TypedTransformer.declarations_in(node.body)
        for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
            if arg.annotation: self.declarations[arg.arg] = ast.unparse(arg.annotation)
        self.returns = ast.unparse(node.returns) if node.returns else None
        self.yields = None
        if isinstance(node.returns,ast.Subscript) and any(isinstance(n,(ast.Yield,ast.YieldFrom)) for n in TypedTransformer.function_nodes(node)):
            args = node.returns.slice.elts if isinstance(node.returns.slice,ast.Tuple) else [node.returns.slice]
            self.yields = ast.unparse(args[0])
            self.returns = ast.unparse(args[2]) if len(args) == 3 else 'None'
        for statement in node.body: self.visit(statement)
        self.declarations, self.returns, self.functions, self.yields = previous, returned, functions, yielded
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self,node):
        previous = self.declarations
        self.declarations = self.declarations | TypedTransformer.declarations_in(node.body)
        for statement in node.body: self.visit(statement)
        self.declarations = previous

    def visit_Call(self,node):
        definition = self.functions.get(node.func.id) if isinstance(node.func,ast.Name) else None
        if definition:
            positional = definition.args.posonlyargs + definition.args.args
            for value, arg in zip(node.args,positional):
                if arg.annotation: self.apply(value,ast.unparse(arg.annotation))
            parameters = {arg.arg:arg for arg in positional + definition.args.kwonlyargs}
            for keyword in node.keywords:
                arg = parameters.get(keyword.arg)
                if arg and arg.annotation: self.apply(keyword.value,ast.unparse(arg.annotation))
        self.generic_visit(node)

    def visit_Assign(self,node):
        for target in node.targets:
            if isinstance(target,ast.Name): self.apply(node.value,self.declarations.get(target.id))
        self.generic_visit(node)

    def visit_AnnAssign(self,node):
        self.apply(node.value,ast.unparse(node.annotation))
        if node.value: self.visit(node.value)

    def visit_Yield(self,node):
        self.apply(node.value,self.yields)
        self.generic_visit(node)

    def visit_Return(self,node):
        self.apply(node.value,self.returns)
        self.generic_visit(node)
