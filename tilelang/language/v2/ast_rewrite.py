import ast
from typing import Dict, Optional, List, Any, Literal, Tuple

_span_attrs = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']

def ast_has_span(ast: ast.AST) -> bool:
    return all(hasattr(ast, attr) for attr in _span_attrs)

def ast_get_span(ast: ast.AST) -> Tuple[int, int, int, int]:
    if not ast_has_span(ast):
        return None
    return tuple(getattr(ast, attr) for attr in _span_attrs)

def ast_set_span(ast: ast.AST, span: Tuple[int, int, int, int]):
    if not ast_has_span(ast):
        return
    for attr, value in zip(_span_attrs, span):
        setattr(ast, attr, value)

class QuoteVisitor(ast.NodeTransformer):

    def __init__(self, names: Dict[str, ast.AST], passes: Optional[List[Any]] = None, span=None):
        self.names = names
        self.passes = passes or []
        self.span = span

    def generic_visit(self, node: ast.AST):
        if self.span is not None:
            ast_set_span(node, self.span)
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.names:
            return self.names[node.id]
        else:
            return node

    def visit_Pass(self, node: ast.Pass) -> Any:
        item = self.passes.pop(0)
        return item if item else node


def quote(expr: str, *, passes: Optional[List[Any]] = None, span=None, **kws) -> List[ast.AST]:
    tree = ast.parse(expr)
    if isinstance(span, ast.AST):
        span = ast_get_span(span)
    tree = QuoteVisitor(kws, passes, span).visit(tree)
    return tree.body


def quote1(expr: str, *, passes: Optional[List[Any]] = None, span=None, **kws) -> ast.AST:
    res = quote(expr, passes=passes, span=span, **kws)
    assert len(res) == 1
    return res[0]


def quote_expr(expr: str, **kws) -> List[ast.AST]:
    res = quote1(expr, **kws)
    assert isinstance(res, ast.Expr)
    return res.value


OpKind = Literal['add', 'sub', 'mul', 'matmul', 'div', 'mod', 'pow', 'lshift', 'rshift', 'or',
                 'xor', 'and', 'floor_div']

_aug_assign_op_map = {
    ast.Add: 'add',
    ast.Sub: 'sub',
    ast.Mult: 'mul',
    ast.MatMult: 'matmul',
    ast.Div: 'div',
    ast.Mod: 'mod',
    ast.Pow: 'pow',
    ast.LShift: 'lshift',
    ast.RShift: 'rshift',
    ast.BitOr: 'or',
    ast.BitXor: 'xor',
    ast.BitAnd: 'and',
    ast.FloorDiv: 'floor_div'
}


class DSLMutator(ast.NodeTransformer):

    def __init__(self):
        self.tmp_counter = 0

    def get_tmp(self) -> str:
        name = f"__{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        return quote(
            "with __tb.ctx_if(cond):\n"
            "  for _ in __tb.ctx_then():\n"
            "    pass\n"
            "  for _ in __tb.ctx_else():\n"
            "    pass\n",
            cond=node.test,
            passes=[node.body, node.orelse],
            span=node,
        )

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        return quote("__tb.eval(value)", value=node.value, span=node)

    def _parse_names(self, target: ast.expr):
        if isinstance(target, ast.Name):
            return f"'{target.id}'"
        elif isinstance(target, ast.Tuple):
            return ("(" + ",".join([self._parse_names(elt) for elt in target.elts]) + ",)")
        else:
            raise SyntaxError("Unsupported for target")

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        names = self._parse_names(node.target)
        return quote(
            f"for target in __tb.ctx_for({names}, range):\n  pass",
            target=node.target,
            range=node.iter,
            passes=[node.body],
            span=node,
        )

    def visit_Continue(self, node: ast.Continue):
        return quote("if __tb.ctx_continue(): continue", span=node)

    def visit_Break(self, node: ast.Break):
        return quote("if __tb.ctx_break(): break", span=node)

    def _emit_assign_target(self, target: ast.expr, rval: ast.expr) -> List[ast.AST]:
        if isinstance(target, ast.Name):
            return quote(f"name = __tb.bind('{target.id}', value)", name=target, value=rval, span=target)
        elif isinstance(target, ast.Subscript):
            return quote(
                "__tb.assign(lval, slice, value)",
                lval=target.value,
                slice=target.slice,
                value=rval,
                span=target,
            )
        else:
            unpacked = []
            def _visit_target(target: ast.expr) -> str:
                if isinstance(target, ast.Name):
                    tmp = self.get_tmp()
                    unpacked.append((tmp, target))
                    res = ast.Name(id=tmp, ctx=target.ctx)
                    ast_set_span(res, ast_get_span(target))
                    return res
                elif isinstance(target, ast.Subscript):
                    tmp = self.get_tmp()
                    unpacked.append((tmp, target))
                    res = ast.Name(id=tmp, ctx=target.ctx)
                    ast_set_span(res, ast_get_span(target))
                    return res
                elif isinstance(target, ast.Tuple):
                    elts = [_visit_target(elt) for elt in target.elts]
                    res = ast.Tuple(elts=elts, ctx=target.ctx)
                    ast_set_span(res, ast_get_span(target))
                    return res
            unpack_stmt = ast.Assign(targets=[_visit_target(target)], value=rval)
            ast_set_span(unpack_stmt, ast_get_span(target))
            stmts = [unpack_stmt]
            bind_lvals = []
            bind_rvals = []
            def flush_binds():
                if bind_lvals:
                    stmts.append(quote1(f'{", ".join(bind_lvals)}, = {", ".join(bind_rvals)},', span=target))
                    bind_lvals.clear()
                    bind_rvals.clear()
            for tmp, target in unpacked:
                if isinstance(target, ast.Name):
                    bind_lvals.append(target.id)
                    bind_rvals.append(f'__tb.bind("{target.id}", {tmp})')
                elif isinstance(target, ast.Subscript):
                    flush_binds()
                    stmts.append(quote1(f'__tb.assign(lval, slice, {tmp})', lval=target.value, slice=target.slice, span=target))
                else:
                    raise NotImplementedError(f'Unsupported target: {target}')
            flush_binds()
            return stmts

    def visit_Assign(self, node: ast.Assign) -> List[ast.AST]:
        node = self.generic_visit(node)
        rval = node.value
        stmts = []
        for target in reversed(node.targets):
            stmts.extend(self._emit_assign_target(target, rval))
            rval = target
        return stmts

    def visit_AugAssign(self, node: ast.AugAssign) -> List[ast.AST]:
        node = self.generic_visit(node)
        target, rval = node.target, node.value
        op = _aug_assign_op_map[type(node.op)]
        if isinstance(target, ast.Name):
            return quote(
                f"name = __tb.aug_assign('{op}', '{target.id}', value)", name=target, value=rval, span=node)
        elif isinstance(target, ast.Subscript):
            return quote(
                f"__tb.aug_assign('{op}', lval, slice, value)",
                lval=target.value,
                slice=target.slice,
                value=rval,
                span=node,
            )
        else:
            return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        all_args = node.args.posonlyargs + node.args.args
        if node.args.vararg is not None:
            all_args += node.args.vararg
        all_args += node.args.kwonlyargs
        stmts = []
        for arg in all_args:
            name = arg.arg
            if arg.annotation is not None:
                arg_stmt = quote1(
                    f'{name} = __tb.arg("{name}", {name}, annot)', annot=arg.annotation, span=arg)
            else:
                arg_stmt = quote1(f'{name} = __tb.arg("{name}", {name})', span=arg)
            stmts.append(arg_stmt)
        node.decorator_list.pop(0)
        node.body = stmts + node.body
        return quote1(f'def __closure(__tb):\n  pass\n  return {node.name}\n', passes=[node], span=node)

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.And):
            last = node.values[-1]
            for i in reversed(range(len(node.values) - 1)):
                last = quote_expr(
                    expr="__tb.logical_and(left, lambda: right)",
                    left=node.values[i],
                    right=last,
                    span=node,
                )
            return last
        elif isinstance(node.op, ast.Or):
            last = node.values[-1]
            for i in reversed(range(len(node.values) - 1)):
                last = quote_expr(
                    "__tb.logical_or(left, lambda: right)",
                    left=node.values[i],
                    right=last,
                    span=node,
                )
            return last
        else:
            return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        node = self.generic_visit(node)
        left = node.left
        splited = []
        for op, comp in zip(node.ops, node.comparators):
            cmp = ast.Compare(left=left, ops=[op], comparators=[comp])
            ast_set_span(cmp, ast_get_span(node))
            splited.append(cmp)
            left = comp
        last = splited[-1]
        for i in reversed(range(len(splited) - 1)):
            last = quote_expr(
                "__tb.logical_and(left, lambda: right)", left=splited[i], right=last, span=node)
        return last

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        return quote_expr(
            '__tb.ifexp(cond, lambda: then, lambda: otherwise)',
            cond=node.test,
            then=node.body,
            otherwise=node.orelse,
            span=node)

    def visit_Return(self, node: ast.Return):
        return quote("return __tb.ret(value)", value=node.value, span=node)

    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        for expr in node.items:
            expr.context_expr = quote_expr("__tb.ctx(e)", e=expr.context_expr, span=expr)
        return node
