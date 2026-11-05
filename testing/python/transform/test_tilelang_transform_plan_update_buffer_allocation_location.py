import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue


def _apply_plan_update(func: tirx.Function) -> tvm_tirx.IRModule:
    target = tvm_tirx.target.Target("cuda")
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = CUDAPassPipelineBodyPrologue(mod, target)
        mod = tl.cuda.transform.LowerSharedTmem()(mod)
        mod = tl.transform.IfStmtBinding()(mod)
        mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    return mod


def _local_allocations(stmt):
    buffers = []

    def _visit(node):
        if (
            tirx.is_tensor_var(node.var)
            and str(node.var.ty.scope) == "local.var"
            and isinstance(node.value, ir.Call)
            and node.value.op == ir.Op.get("tirx.alloc_tensor")
        ):
            buffers.append(node.var)

    tvm_ffi.structural_walk(stmt, (ir.Bind, _visit), order="post")
    return buffers


def _find_first_for(stmt: ir.Stmt) -> ir.For:
    loops = []

    def _visit(node):
        loops.append(node)

    tvm_ffi.structural_walk(stmt, (ir.For, _visit), order="post")
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

    # The old root SBlock owned both allocations. In native IR the matching
    # scope is the kernel's metadata region inside the complete launch chain,
    # not every nested allocation reachable from the function.
    scope = main.body
    for tag in ("blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z"):
        assert len(scope) == 1
        launch = scope[0]
        assert isinstance(launch, ir.RegionStmt)
        assert launch.op == ir.Op.get("tirx.launch_thread")
        assert launch.args[0].value == tag
        scope = launch.body

    assert len(scope) == 1
    kernel_scope = scope[0]
    assert isinstance(kernel_scope, ir.RegionStmt)
    assert kernel_scope.op == ir.Op.get("tl.attr")
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(kernel_scope.args[0], kernel_scope.attrs["annotation_schema"])
    assert metadata["key"] == "tl.kernel"
    assert not kernel_scope.body_params and not kernel_scope.result_vars
    scope = kernel_scope.body

    direct_allocations = {
        statement.var.name: (index, statement.var)
        for index, statement in enumerate(scope)
        if isinstance(statement, ir.Bind) and _local_allocations(statement)
    }
    assert {"a", "b"} <= direct_allocations.keys()
    first_loop_index = next(index for index, statement in enumerate(scope) if isinstance(statement, ir.For))
    all_allocations = _local_allocations(main.body)
    for name in ("a", "b"):
        index, var = direct_allocations[name]
        assert index < first_loop_index
        assert sum(buffer.name == name for buffer in all_allocations) == 1
        assert next(buffer for buffer in all_allocations if buffer.name == name).same_as(var)

    assert tirx.analysis.check_well_formed(mod)
    assert not tirx.analysis.undefined_vars(main.body, list(main.params))

    loop = _find_first_for(main.body)
    loop_body_local_vars = {buf.name for buf in _local_allocations(loop.body)}
    assert "b" not in loop_body_local_vars


if __name__ == "__main__":
    tilelang.testing.main()
