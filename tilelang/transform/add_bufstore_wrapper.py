from tvm.tir import BufferStore, For, AttrStmt, ForKind, Var, PrimFunc
from tvm.tir.stmt_functor import ir_transform, post_order_visit
from tvm.tir.transform import prim_func_pass


def AddWrapperForSingleBufStore():

    def pass_fn(func: PrimFunc, mod, ctx):
        pfor = 0
        thread_binding_var = set()

        def get_used_var(op):
            used_var = set()

            def visit_fn(x):
                if isinstance(x, Var):
                    used_var.add(x)

            post_order_visit(op, visit_fn)
            return used_var

        def is_tile_op_for(op: For):
            return op.kind == ForKind.PARALLEL or 'num_stages' in op.annotations

        def pre_visit(stmt):
            nonlocal pfor
            if isinstance(stmt, AttrStmt) and stmt.attr_key == 'thread_extent':
                thread_binding_var.add(stmt.node.var)
            if isinstance(stmt, For):
                pfor += is_tile_op_for(stmt)

        def post_visit(stmt):
            nonlocal pfor
            if isinstance(stmt, For):
                pfor -= is_tile_op_for(stmt)
            if isinstance(stmt, BufferStore):
                used_var = get_used_var(stmt)
                used_binding = used_var.intersection(thread_binding_var)
                if not pfor and len(used_binding) == 0:
                    return For(Var("_", "int"), 0, 1, ForKind.PARALLEL, stmt)

        new_body = ir_transform(func.body, pre_visit, post_visit)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
