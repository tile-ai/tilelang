import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
from tvm_tirx import ir, tirx


def _apply_plan_update(func: tvm.tirx.PrimFunc) -> tvm.IRModule:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = CUDAPassPipelineBodyPrologue(mod, target)
        mod = tl.cuda.transform.LowerSharedTmem()(mod)
        mod = tl.transform.IfStmtBinding()(mod)
        mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    return mod


def _local_allocations(stmt):
    allocations = set()

    def _visit(node):
        if (
            isinstance(node, ir.Bind)
            and tirx.is_tensor_var(node.var)
            and node.var.ty.scope.name in ("local", "local.var")
            and isinstance(node.value, ir.Call)
            and node.value.op.name == "tirx.alloc_tensor"
        ):
            allocations.add(node.var.name)

    tvm.tirx.stmt_functor.post_order_visit(stmt, _visit)
    return allocations


def _find_first_for(stmt: tvm.tirx.Stmt) -> tvm.tirx.For:
    loops = []

    def _visit(node):
        if isinstance(node, tvm.tirx.For):
            loops.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, _visit)
    assert loops, "Expected at least one loop"
    return loops[0]


@tilelang.testing.requires_cuda
def test_plan_update_keeps_loop_header_local_var_outside_loop_body():
    @T.prim_func
    def func(x: T.Tensor((256,), "int64")):
        with T.Kernel(256, threads=128):
            a, b = T.alloc_var(T.int), T.alloc_var(T.int)
            T.fill(x[a:b], 0)

    mod = _apply_plan_update(func)
    main = mod["main"]

    assert {"a", "b"} <= _local_allocations(main.body)

    loop = _find_first_for(main.body)
    loop_body_local_vars = _local_allocations(loop.body)
    assert "b" not in loop_body_local_vars


if __name__ == "__main__":
    tilelang.testing.main()
