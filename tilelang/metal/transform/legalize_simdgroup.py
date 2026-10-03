"""Keep vector accesses within the two elements owned by a SIMD-group lane."""

from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx.transform import prim_func_pass


@prim_func_pass(opt_level=0)
def LegalizeSimdgroupVectorization(func, mod, ctx):
    matrices = set()
    mma = Op.get("tirx.simdgroup_multiply_accumulate")

    def collect(node):
        if isinstance(node, tir.Call) and node.op == mma:
            matrices.update(node.args[i] for i in (0, 2, 4, 6))

    tir.stmt_functor.post_order_visit(func.body, collect)
    if not matrices:
        return func

    def rewrite(node):
        if not isinstance(node, tir.For) or node.kind != tir.ForKind.VECTORIZED:
            return None
        if not isinstance(node.extent, tir.IntImm) or node.extent.value <= 2:
            return None
        accesses_matrix = False

        def visit(access):
            nonlocal accesses_matrix
            if isinstance(access, (tir.BufferLoad, tir.BufferStore)) and access.buffer.data in matrices:
                accesses_matrix = True

        tir.stmt_functor.post_order_visit(node.body, visit)
        if not accesses_matrix:
            return None
        width = 2 if node.extent.value % 2 == 0 else 1
        outer = tir.Var(node.loop_var.name + "_matrix", node.loop_var.dtype)
        body = tir.stmt_functor.substitute(node.body, {node.loop_var: node.min + outer * width + node.loop_var})
        inner = tir.For(node.loop_var, 0, width, tir.ForKind.VECTORIZED, body, annotations=node.annotations)
        return tir.For(outer, 0, node.extent // width, tir.ForKind.UNROLLED, inner)

    return func.with_body(tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.For"]))
