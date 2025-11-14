import ast
from logging.config import valid_ident
import re
# from rich import print

from argparse import ArgumentParser

with open('tilelang/language/tir/op.py') as f:
    data = f.read()

tree = ast.parse(data)

def convert_tree(x):
    result = {}
    for fname, value in ast.iter_fields(x):
        if isinstance(value, list):
            result[fname] = [convert_tree(v) if isinstance(v, ast.AST) else v for v in value]
        elif isinstance(value, ast.AST):
            result[fname] = convert_tree(value)
        else:
            result[fname] = value
    return result

# print(convert_tree(tree))

funcs = {}

subst = {
    'Expr': 'PrimExpr',
    'UIntImm': 'IntImm',
    'tvm.Expr': 'PrimExpr'
}

for fdef in tree.body:
    if not isinstance(fdef, ast.FunctionDef):
        continue
    if not isinstance(fdef.body[0], ast.Expr):
        continue
    value = fdef.body[0].value
    if not isinstance(value, ast.Constant):
        continue
    data = value.value
    if not isinstance(data, str):
        continue
    lines = data.splitlines()
    ty = None
    annots = {}
    for i, line in enumerate(lines):
        if i > 0 and re.fullmatch(r'    \s*----+', line):
            annot = lines[i - 1]
            ty = None
            if annot == '    Parameters':
                ty = 'param'
            if annot == '    Returns':
                ty = 'return'
        if mat := re.fullmatch(r'\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s+(.*)', line):
            name, val = mat.groups()
            val = subst.get(val, val)
            if ty == 'param':
                annots[name] = val
            if ty == 'return':
                annots['return'] = val
    
    pe_arg = []
    span_arg = []
    other_arg = []
    for args in fdef.args.args:
        if args.arg in annots:
            annot = annots[args.arg]
            if annot == 'PrimExpr':
                pe_arg.append(args.arg)
            elif annot == 'Optional[Span]':
                span_arg.append(args.arg)
            else:
                other_arg.append(args.arg)
            try:
                args.annotation = ast.parse(annot).body[0].value
            except Exception as e:
                print(annot, repr(e))
        else:
            other_arg.append(args.arg)
    if 'return' in annots:
        try:
            fdef.returns = ast.parse(annots['return']).body[0].value
        except Exception as e:
            print(annots['return'], repr(e))
    if annots.get('return', None) == 'PrimExpr' and not other_arg:
        print('UT Prim: ', fdef.name)
        Tvar = ast.parse('_T').body[0].value
        for args in fdef.args.args:
            if args.arg in pe_arg:
                args.annotation = Tvar
        fdef.returns = Tvar
    fdef.body = [ast.parse('...')]
    # funcs.append(fdef)
    funcs[fdef.name] = fdef

# tree.body = funcs
# print(ast.unparse(tree))

with open('tilelang/language/tir/ir.py') as f:
    data = f.read()

all_funcs = []

for name in re.findall(r'([A-Za-z_][A-Za-z0-9_]*) = _op_wrapper', data):
    if name in funcs:
        print(name)
        all_funcs.append(funcs[name])


tree.body = all_funcs

with open('tilelang/language/tir/ir.pyi', 'w') as f:
    f.write(ast.unparse(tree))