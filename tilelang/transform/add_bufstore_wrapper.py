from tvm.tir import PyStmtExprMutator, PyStmtExprVisitor, BufferStore, For, AttrStmt, Block, ForKind, IterVar, Var, PrimFunc
from tvm.tir.functor import mutator, visitor
from tvm.tir.transform import prim_func_pass


@visitor
class FindVarUse(PyStmtExprVisitor):

    def __init__(self):
        self.used_var = set()

    def visit_var_(self, op: Var):
        self.used_var.add(op)
        super().visit_var_(op)


@mutator
class AddWrapperForSingleStoreMutator(PyStmtExprMutator):
    '''
    Add a dummy parallel for loop to wrap the single buffer store
      Condition:
        1. not inside a parallel for loop
        2. no custom thread binding, i.e. threadIdx.x, blockIdx.x
    '''

    def __init__(self):
        self.inside_pfor = 0
        self.thread_binding_var = set()

    def visit_block_(self, op: Block):
        super().visit_block_(op)
        return op

    def visit_attr_stmt_(self, op: AttrStmt):
        if op.attr_key == 'thread_extent':
            iter_var: IterVar = op.node
            self.thread_binding_var.add(iter_var.var)
        super().visit_attr_stmt_(op)
        return op

    def visit_for_(self, op: For):
        pfor = op.kind == ForKind.PARALLEL or 'num_stages' in op.annotations
        self.inside_pfor += pfor
        super().visit_for_(op)
        self.inside_pfor -= pfor
        return op

    def visit_buffer_store_(self, op: BufferStore):
        # This pass runs after LetInline, we find var inside the stmt
        fv = FindVarUse()
        fv.visit_stmt(op)
        used_binding = fv.used_var.intersection(self.thread_binding_var)
        if not self.inside_pfor and len(used_binding) == 0:
            return For(Var("_", "int"), 0, 1, ForKind.PARALLEL, op)
        else:
            super().visit_buffer_store_(op)
            return op


def AddWrapperForSingleBufStore():

    def pass_fn(func: PrimFunc, mod, ctx):
        mut = AddWrapperForSingleStoreMutator()
        new_body = mut.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
