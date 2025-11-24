from tvm import tir
from tvm.tir import PrimFunc
from tvm.tir.transform import prim_func_pass
from tvm.tir.stmt_functor import ir_transform


def ASTPrinter():
    """
    Print the AST of a given tilelang module for debugging.
    """

    def pre_visit(statement: tir.Stmt) -> None:
        """
        Pre-order visitor to print all visited statements.
        """

        print(f"Visiting statement: {type(statement)}")

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        new_body = ir_transform(func.body, pre_visit, None)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
