import tvm
from tvm.tir import PyStmtExprMutator, BufferStore, For, ForKind, Var, PrimFunc
from tvm.tir.functor import mutator
from tvm.tir.transform import prim_func_pass


@mutator
class AddWrapperForSingleStoreMutator(PyStmtExprMutator):
    def __init__(self):
        self.inside_pfor = 0

    def visit_for_(self, op: For):
        pfor = op.kind == ForKind.PARALLEL
        self.inside_pfor += pfor
        res = super().visit_for_(op)
        self.inside_pfor -= pfor
        return res

    def visit_buffer_store_(self, op: BufferStore):
        if not self.inside_pfor:
            return For(Var("_", "int"), 0, 1, ForKind.PARALLEL, op)
        else:
            return super().visit_buffer_store_(op)


def AddWrapperForSingleBufStore():
    def pass_fn(func: PrimFunc, mod, ctx):
        mut = AddWrapperForSingleStoreMutator()
        new_body = mut.visit_stmt(func.body)
        return func.with_body(new_body)
    return prim_func_pass(pass_fn, opt_level=0)