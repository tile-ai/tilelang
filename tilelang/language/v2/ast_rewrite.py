import ast
from typing import Dict, Optional, List, Any, Literal


class QuoteVisitor(ast.NodeTransformer):

    def __init__(self, names: Dict[str, ast.AST], passes: Optional[List[Any]] = None):
        self.names = names
        self.passes = passes or []

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
    tree = QuoteVisitor(kws, passes).visit(tree)
    return tree.body


def quote1(expr: str, *, passes: Optional[List[Any]] = None, span=None, **kws) -> ast.AST:
    res = quote(expr, passes=passes, span=span, **kws)
    assert len(res) == 1
    return res[0]


def quote_expr(expr: str, **kws) -> List[ast.AST]:
    res = quote1(expr, **kws)
    assert isinstance(res, ast.Expr)
    return res.value


class DSLMutator(ast.NodeTransformer):

    def __init__(self):
        self.tmp_counter = 0

    def get_tmp(self) -> str:
        name = f"__tmp_{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        return quote(
            "with __tl.ctx_if(cond):\n"
            "  for _ in __tl.ctx_then():\n"
            "    pass\n"
            "  for _ in __tl.ctx_else():\n"
            "    pass\n",
            cond=node.test,
            passes=[node.body, node.orelse],
        )

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        return quote("__tl.eval(value)", value=node.value)

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
            f"for target in __tl.ctx_for({names}, range):\n  pass",
            target=node.target,
            range=node.iter,
            passes=[node.body],
        )

    def _emit_assign_tuple(self, targets: List[ast.expr], rval: ast.expr) -> List[ast.AST]:
        tmp_names = [self.get_tmp() for _ in range(len(targets))]
        unpack = quote1(",".join(tmp_names) + ", = value", value=rval)
        stmts = [unpack]
        for i, target in enumerate(targets):
            stmts.extend(
                self._emit_assign_target(target, ast.Name(id=tmp_names[i], ctx=ast.Load())))
        return stmts

    def _emit_assign_target(self, target: ast.expr, rval: ast.expr) -> List[ast.AST]:
        if isinstance(target, ast.Name):
            return quote(f"name = __tl.bind('{target.id}', value)", name=target, value=rval)
        elif isinstance(target, ast.Subscript):
            return quote(
                "__tl.assign(lval, slice, value)",
                lval=target.value,
                slice=target.slice,
                value=rval,
            )
        elif isinstance(target, ast.Tuple):
            return self._emit_assign_tuple(target.elts, rval)

    def visit_Assign(self, node: ast.Assign) -> List[ast.AST]:
        node = self.generic_visit(node)
        rval = node.value
        stmts = []
        for target in reversed(node.targets):
            stmts.extend(self._emit_assign_target(target, rval))
            rval = target
        return stmts

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
                    f'{name} = __tl.arg("{name}", {name}, annot)', annot=arg.annotation)
            else:
                arg_stmt = quote1(f'{name} = __tl.arg("{name}", {name})')
            stmts.append(arg_stmt)
        node.decorator_list.pop(0)
        node.body = stmts + node.body
        return quote1(f'def __closure(__tl):\n  pass\n  return {node.name}\n', passes=[node])

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.And):
            last = node.values[-1]
            for i in reversed(range(len(node.values) - 1)):
                last = quote_expr(
                    expr="__tl.logical_and(left, lambda: right)",
                    left=node.values[i],
                    right=last,
                )
            return last
        elif isinstance(node.op, ast.Or):
            last = node.values[-1]
            for i in reversed(range(len(node.values) - 1)):
                last = quote_expr(
                    "__tl.logical_or(left, lambda: right)",
                    left=node.values[i],
                    right=last,
                )
            return last
        else:
            return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        node = self.generic_visit(node)
        left = node.left
        splited = []
        for op, comp in zip(node.ops, node.comparators):
            splited.append(ast.Compare(left=left, ops=[op], comparators=[comp]))
            left = comp
        last = splited[-1]
        for i in reversed(range(len(splited) - 1)):
            last = quote_expr("__tl.logical_and(left, lambda: right)", left=splited[i], right=last)
        return last

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        return quote_expr(
            '__tl.ifexp(cond, lambda: then, lambda: otherwise)',
            cond=node.test,
            then=node.body,
            otherwise=node.orelse)

    def visit_Return(self, node: ast.Return):
        return quote("return __tl.ret(value)", value=node.value)

    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        for expr in node.items:
            expr.context_expr = quote_expr("__tl.ctx(e)", e=expr.context_expr)
        return node
