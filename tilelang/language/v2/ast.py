from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Callable, Generic, Any, Literal, TypeVar
from contextlib import AbstractContextManager
from collections.abc import Iterable

# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:  # Python < 3.10
    from typing_extensions import ParamSpec
import inspect

# from .utils import get_ast, get_compiled_object
from . import utils

_span_attrs = ["lineno", "col_offset", "end_lineno", "end_col_offset"]


def ast_has_span(ast: ast.AST) -> bool:
    return all(hasattr(ast, attr) for attr in _span_attrs)


def ast_get_span(ast: ast.AST) -> tuple[int, int, int, int]:
    if not ast_has_span(ast):
        return None
    return tuple(getattr(ast, attr) for attr in _span_attrs)


def ast_set_span(ast: ast.AST, span: tuple[int, int, int, int]):
    if not ast_has_span(ast):
        return
    for attr, value in zip(_span_attrs, span):
        setattr(ast, attr, value)


class QuoteVisitor(ast.NodeTransformer):
    def __init__(self, names: dict[str, ast.AST], passes: list[Any] | None = None, span=None):
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


def quote(expr: str, *, passes: list[Any] | None = None, span=None, **kws) -> list[ast.AST]:
    tree = ast.parse(expr)
    if isinstance(span, ast.AST):
        span = ast_get_span(span)
    tree = QuoteVisitor(kws, passes, span).visit(tree)
    return tree.body


def quote1(expr: str, *, passes: list[Any] | None = None, span=None, **kws) -> ast.AST:
    res = quote(expr, passes=passes, span=span, **kws)
    assert len(res) == 1
    return res[0]


def quote_expr(expr: str, **kws) -> ast.expr:
    res = quote1(expr, **kws)
    assert isinstance(res, ast.Expr)
    return res.value


Operator = Literal["Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv"]
BoolOp = Literal["And", "Or", "Not"]


def get_operator_name(operator: ast.operator) -> Operator:
    return operator.__class__.__name__


def get_boolop_name(boolop: ast.boolop) -> BoolOp:
    return boolop.__class__.__name__


_T = TypeVar("_T")


def eval_op(op: Operator, left: Any, right: Any) -> Any:
    if op == "Add":
        return left + right
    if op == "Sub":
        return left - right
    if op == "Mult":
        return left * right
    if op == "MatMult":
        return left @ right
    if op == "Div":
        return left / right
    if op == "Mod":
        return left % right
    if op == "Pow":
        return left**right
    if op == "LShift":
        return left << right
    if op == "RShift":
        return left >> right
    if op == "BitOr":
        return left | right
    if op == "BitXor":
        return left ^ right
    if op == "BitAnd":
        return left & right
    if op == "FloorDiv":
        return left // right
    raise ValueError(f"Unknown operator: {op}")


def eval_aug_assign(op: Operator, left: Any, sl: slice, right: Any) -> Any:
    if op == "Add":
        left[sl] += right
        return left
    if op == "Sub":
        left[sl] -= right
        return left
    if op == "Mult":
        left[sl] *= right
        return left
    if op == "MatMult":
        left[sl] @= right
        return left
    if op == "Div":
        left[sl] /= right
        return left
    if op == "Mod":
        left[sl] %= right
        return left
    if op == "Pow":
        left[sl] **= right
        return left
    if op == "LShift":
        left[sl] <<= right
        return left
    if op == "RShift":
        left[sl] >>= right
        return left
    if op == "BitOr":
        left[sl] |= right
        return left
    if op == "BitXor":
        left[sl] ^= right
        return left
    if op == "BitAnd":
        left[sl] &= right
        return left
    if op == "FloorDiv":
        left[sl] //= right
        return left
    raise ValueError(f"Unknown operator: {op}")


class _empty: ...


class BaseBuilder:
    empty = _empty

    def get_parent_locals(self):
        return inspect.currentframe().f_back.f_back.f_locals

    def ctx_if(self, cond) -> Iterable[_T]:
        yield cond

    def ctx_then(self, val: _T) -> Iterable[None]:
        if val:
            yield

    def ctx_else(self, val: _T) -> Iterable[None]:
        if not val:
            yield

    def eval(self, val: Any):  # noqa: B027
        pass

    def ctx_for(self, range: Iterable[Any]) -> Iterable[Any]:
        return range

    def ctx_continue(self) -> bool:
        return True

    def ctx_break(self) -> bool:
        return True

    def ctx_while(self, cond: Callable[[], Any]) -> Iterable[None]:
        while cond():
            yield

    def bind(self, name: str, value: Any, annot: Any = empty) -> Any:
        return value

    def unwrap_value(self, value):
        return value

    def assign_slice(self, lval: Any, sl: slice, value: Any, annot: Any = empty):
        lval[sl] = value

    def aug_assign(self, op: Operator, target: Any, aug_value: Any) -> Any:
        return eval_op(op, target, aug_value)

    def aug_assign_slice(self, op: Operator, target: Any, sl: slice, aug_value: Any):
        eval_aug_assign(op, target, sl, aug_value)

    def boolop(self, op: BoolOp, left: Any, right: Callable[[], Any] | None = None) -> Any:
        if op == "And":
            return left and right()
        if op == "Or":
            return left or right()
        if op == "Not":
            return not left
        raise ValueError(f"Unknown boolop: {op}")

    def ifexp(self, cond: Any, then: Callable[[], Any], otherwise: Callable[[], Any]) -> Any:
        return then() if cond else otherwise()

    def ret(self, value: Any) -> Any:
        return value

    def ctx_with(self, ctx: AbstractContextManager[Any]) -> AbstractContextManager[Any]:
        return ctx

    def assert_expr(self, cond: Any, msg: Any):
        assert cond, msg

    def rval(self, name: str, value: Any):
        return value

    def arg(self, name: str, value: Any):
        return value

    def override(self, name: str):
        return globals()[name]


class DSLMutator(ast.NodeTransformer):
    def __init__(self, closure_names: list[str]):
        self.tmp_counter = 0
        self.closure_names = closure_names

    def get_tmp(self) -> str:
        name = f"__{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        br = self.get_tmp()
        if len(node.orelse) == 0:
            return quote(
                f"for {br} in __tb.ctx_if(cond):\n  for _ in __tb.ctx_then({br}):\n    pass\n",
                cond=node.test,
                passes=[node.body],
                span=node,
            )
        return quote(
            f"for {br} in __tb.ctx_if(cond):\n  for _ in __tb.ctx_then({br}):\n    pass\n  for _ in __tb.ctx_else({br}):\n    pass\n",
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
            return "(" + ",".join([self._parse_names(elt) for elt in target.elts]) + ",)"
        else:
            s = ast.unparse(target)
            raise NotImplementedError(f"Unsupported for target `{s}`")

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        tmp = self.get_tmp()
        # names = self._parse_names(node.target)
        var = ast.Name(tmp, ctx=ast.Load())
        ast_set_span(var, ast_get_span(node.target))
        stmts = self._emit_assign_target(node.target, var)
        return quote(
            f"for {tmp} in __tb.ctx_for(range):\n  pass\n",
            target=node.target,
            range=node.iter,
            passes=[stmts + node.body],
            span=node,
        )

    def visit_Continue(self, node: ast.Continue):
        node = self.generic_visit(node)
        return quote("if __tb.ctx_continue(): continue", span=node)

    def visit_Break(self, node: ast.Break):
        node = self.generic_visit(node)
        return quote("if __tb.ctx_break(): break", span=node)

    def _emit_assign_target(self, target: ast.expr, rval: ast.expr, annot: ast.expr = None) -> list[ast.AST]:
        if isinstance(target, ast.Name):
            if annot is None:
                return quote(f"name = __tb.bind('{target.id}', value)", name=target, value=rval, span=target)
            else:
                return quote(f'name = __tb.bind("{target.id}", value, annot)', name=target, value=rval, annot=annot, span=target)
        elif isinstance(target, ast.Attribute):
            s = ast.unparse(target)
            raise NotImplementedError(f"Attribute assignment not supported yet, `{s}`")
        elif isinstance(target, ast.Subscript):
            if annot is None:
                return quote(
                    "__tb.assign_slice(lval, slice, value)",
                    lval=target.value,
                    slice=target.slice,
                    value=rval,
                    span=target,
                )
            else:
                return quote(
                    "__tb.assign_slice(lval, slice, value, annot)",
                    lval=target.value,
                    slice=target.slice,
                    value=rval,
                    annot=annot,
                    span=target,
                )
        else:
            # flatten nested tuple into a list of (tmp_name, target)
            unpacked = []

            def _visit_target(target: ast.expr) -> str:
                if isinstance(target, (ast.Name, ast.Subscript)):
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
                else:
                    s = ast.unparse(target)
                    raise NotImplementedError(f"Attribute assignment not supported yet, `{s}`")

            unpack_stmt = ast.Assign(targets=[_visit_target(target)], value=quote_expr("__tb.unwrap_value(rval)", rval=rval, span=rval))
            ast_set_span(unpack_stmt, ast_get_span(target))
            stmts = [unpack_stmt]
            bind_lvals = []
            bind_rvals = []

            def flush_binds():
                if bind_lvals:
                    stmts.append(quote1(f"{', '.join(bind_lvals)}, = {', '.join(bind_rvals)},", span=target))
                    bind_lvals.clear()
                    bind_rvals.clear()

            # the following code generate two phase binding to support swap like semantics
            # for example:
            #       a, b = b, a
            # 1 phase:
            #    _tmp_0, _tmp_1 = b, a
            #    => _tmp_0: T.int32 = b
            #    => _tmp_1: T.int32 = a
            # 2 phase:
            #    a, b = _tmp_0, _tmp_1
            #    => a = _tmp_0 => a[0] = _tmp_0
            #    => b = _tmp_1 => b[0] = _tmp_1

            # 1 phase: _tmp_0, _tmp_1 = __tb.bind('_', a), __tb.bind('_', b)
            for tmp, _target in unpacked:
                bind_lvals.append(tmp)
                bind_rvals.append(f'__tb.bind("_", {tmp})')

            flush_binds()

            # 2 phase: a, b = __tb.bind('a', _tmp_0), __tb.bind('b', _tmp_1)
            for tmp, target in unpacked:
                if isinstance(target, ast.Name):
                    bind_lvals.append(target.id)
                    bind_rvals.append(f'__tb.bind("{target.id}", {tmp})')
                elif isinstance(target, ast.Subscript):
                    flush_binds()
                    stmts.append(quote1(f"__tb.assign_slice(lval, slice, {tmp})", lval=target.value, slice=target.slice, span=target))
                else:
                    s = ast.unparse(target)
                    raise NotImplementedError(f"Unsupported target: {s}")
            flush_binds()
            return stmts

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST]:
        node = self.generic_visit(node)
        rval = node.value
        if len(node.targets) == 1:
            return self._emit_assign_target(node.targets[0], rval)
        else:
            tmp_name = self.get_tmp()
            tmp_store = ast.Name(tmp_name, ctx=ast.Store())
            tmp_load = ast.Name(tmp_name, ctx=ast.Load())
            ast_set_span(tmp_store, node.targets[0])
            ast_set_span(tmp_load, node.targets[0])
            stmt = self._emit_assign_target(tmp_store, rval)
            for target in node.targets:
                stmt.extend(self._emit_assign_target(target, tmp_load))
            return stmt

    def visit_AugAssign(self, node: ast.AugAssign) -> list[ast.AST]:
        node = self.generic_visit(node)
        target, rval = node.target, node.value
        op = get_operator_name(node.op)
        if isinstance(target, ast.Name):
            return quote(f"name = __tb.aug_assign('{op}', {target.id}, value)", name=target, value=rval, span=node)
        elif isinstance(target, ast.Subscript):
            return quote(
                f"__tb.aug_assign_slice('{op}', lval, slice, value)",
                lval=target.value,
                slice=target.slice,
                value=rval,
                span=node,
            )
        else:
            return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        node = self.generic_visit(node)
        rval = node.value or quote_expr("__tb.empty", span=node, annot=node)
        return self._emit_assign_target(node.target, rval, annot=node.annotation)

    def visit_While(self, node):
        node = self.generic_visit(node)
        return quote1("for _ in __tb.ctx_while(lambda: cond):\n  pass", cond=node.test, passes=[node.body], span=node)

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
                arg_stmt = quote1(f'{name} = __tb.arg("{name}", {name})', span=arg)
            else:
                arg_stmt = quote1(f'{name} = __tb.arg("{name}", {name})', span=arg)
            arg.annotation = None
            stmts.append(arg_stmt)
        node.body = stmts + node.body
        node.decorator_list.clear()
        return quote1(
            f"def make_closure({', '.join(self.closure_names)}):\n"
            f"  def {node.name}(__tb):\n"
            "    range = __tb.override('range')\n"
            "    pass\n"
            f"    return {node.name}\n"
            f"  return {node.name}",
            passes=[node],
        )

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        op_name = get_boolop_name(node.op)
        last = node.values[-1]
        for i in reversed(range(len(node.values) - 1)):
            last = quote_expr(
                expr=f"__tb.boolop('{op_name}', left, lambda: right)",
                left=node.values[i],
                right=last,
                span=node,
            )
        return last

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return quote_expr("__tb.boolop('Not', operand)", operand=node.operand, span=node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        node = self.generic_visit(node)
        left = node.left
        split = []
        for op, comp in zip(node.ops, node.comparators):
            cmp = ast.Compare(left=left, ops=[op], comparators=[comp])
            ast_set_span(cmp, ast_get_span(node))
            split.append(cmp)
            left = comp
        last = split[-1]
        for i in reversed(range(len(split) - 1)):
            last = quote_expr("__tb.boolop('And', left, lambda: right)", left=split[i], right=last, span=node)
        return last

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        node = self.generic_visit(node)
        return quote_expr(
            "__tb.ifexp(cond, lambda: then, lambda: otherwise)", cond=node.test, then=node.body, otherwise=node.orelse, span=node
        )

    def visit_Return(self, node: ast.Return):
        node = self.generic_visit(node)
        return quote("return __tb.ret(value)", value=node.value, span=node)

    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        for expr in node.items:
            expr.context_expr = quote_expr("__tb.ctx_with(e)", e=expr.context_expr, span=expr)
        return node

    def visit_Assert(self, node: ast.Assert):
        node = self.generic_visit(node)
        return quote("__tb.assert_expr(cond, msg)", cond=node.test, msg=node.msg, span=node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            return quote_expr(f"__tb.rval('{node.id}', node)", node=node, span=node)
        return node


_P = ParamSpec("_P")


@dataclass
class IRGenerator(Generic[_P, _T]):
    gen: Callable[[BaseBuilder], Callable[_P, _T]]
    source: str


def mutate(func: Callable[_P, _T]) -> IRGenerator[_P, _T]:
    """
    Transform a Python function into an IR (Intermediate Representation) generator.
    This function takes a regular Python function and performs AST (Abstract Syntax Tree)
    transformation to create an IRGenerator that can be used for code generation purposes.
    Args:
        func (Callable[_P, _T]): The Python function to be transformed. This should be a
            callable that will be analyzed and mutated at the AST level. The function's
            signature is preserved through generic type parameters _P (parameters) and
            _T (return type).
    Returns:
        IRGenerator[_P, _T]: An IRGenerator instance wrapping the transformed function.
            The generator contains:
            - gen: The compiled and mutated version of the original function
            - source: The unparsed source code of the transformed AST as a string
    Example:
        >>> @mutate
        ... def my_function(x: int) -> int:
        ...     return x * 2
        >>> # my_function is now an IRGenerator that can be used for code generation
    Note:
        - The original function's closure variables and captured context are preserved
        - The transformation is performed at compile-time through AST manipulation
        - The returned IRGenerator maintains type information from the original function
    """

    tree = utils.get_ast(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    nonlocals = utils.get_func_nonlocals(func)

    # DSLMutator generates a function named `make_closure`
    #   it accepts all names inside nonlocal, and returns the mutated function
    #   this is because we must separate the closure namespace form the global namespace
    #     if we directly inject closure variables into the global namespace,
    #     it generates a new `globals` dict, and the dict owns all reference to the original globalns
    #     which makes memory leak, because the original globalns cannot be freed
    #     ```py
    #     a = 123
    #     def foo():
    #       x = foo.__globals__ # OK, globals are maintained by python
    #       x = {**foo.__globals__, } # Not OK: globals are copied, and the original globals cannot be freed
    #       def bar(): x
    #       return bar
    #     ```
    tree = DSLMutator(nonlocals.keys()).visit(tree)

    make_closure = utils.get_compiled_object(
        tree,
        "make_closure",
        filename,
        func.__globals__,  # use the original globalns
    )
    fn = make_closure(**nonlocals)
    return IRGenerator(gen=fn, source=ast.unparse(tree))
