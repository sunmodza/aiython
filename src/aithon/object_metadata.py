"""Small structural hints; no repr, descriptors, annotation evaluation or value dumps."""
import ast
from itertools import islice
import sys
import types


def source_node(value, manager, type_field):
    if type(value) is types.FunctionType:
        filename = value.__code__.co_filename
        name = value.__name__
        line = value.__code__.co_firstlineno
    elif isinstance(value, type):
        module = sys.modules.get(type_field(value, '__module__'))
        filename = vars(module).get('__file__') if type(module) is types.ModuleType else None
        name = type_field(value, '__name__')
        line = None
    else:
        return None
    index = manager.source_nodes.get(filename)
    if index is None:
        return None
    nodes = index[0].get((name, line), ()) if line is not None else index[1].get(name, ())
    return nodes[0] if len(nodes) == 1 else None


def structural_hint(value, manager, type_field):
    kind = type(value)
    typename = lambda v: type_field(type(v), '__name__')
    result = {}
    if kind is dict:
        result['size'] = len(value)
        fields = {}
        for key, item in islice(value.items(), 20):
            if type(key) is str and len(key) <= 80:
                fields[key] = typename(item)
        result['fields'] = fields
        if len(fields) < len(value):
            result['fields_truncated'] = True
    elif kind in (list, tuple):
        result['size'] = len(value)
        # Samples describe shape only, not an inferred contract for the whole list.
        result['sample_item_types'] = list(dict.fromkeys(typename(v) for v in islice(value, 3)))
        if value and type(value[0]) is dict:
            result['first_item_fields'] = structural_hint(value[0], manager, type_field).get('fields', {})
            result['first_item_fields_sampled'] = True
    elif kind is types.ModuleType:
        name = vars(value).get('__name__')
        if type(name) is str:
            result['name'] = name
    elif isinstance(value, type):
        result['name'] = type_field(value, '__qualname__')
    elif kind is types.FunctionType:
        code = value.__code__
        count = code.co_argcount + code.co_kwonlyargcount
        result['parameters'] = list(code.co_varnames[:count])[:20]
        result['name'] = value.__qualname__
    node = source_node(value, manager, type_field) if isinstance(value, type) or kind is types.FunctionType else None
    if isinstance(node, ast.ClassDef):
        fields = {n.target.id: ast.unparse(n.annotation)[:200] for n in node.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        result['fields'] = dict(islice(fields.items(), 20))
        if len(fields) > 20:
            result['fields_truncated'] = True
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        entries = []
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            entries.append(arg.arg + (': ' + ast.unparse(arg.annotation)[:200] if arg.annotation else ''))
        if args.vararg:
            entries.append('*' + args.vararg.arg)
        if args.kwarg:
            entries.append('**' + args.kwarg.arg)
        result['signature'] = '(' + ', '.join(entries[:20]) + ')'
        if node.returns:
            result['signature'] += ' -> ' + ast.unparse(node.returns)[:200]
        result.pop('parameters', None)
    return result
