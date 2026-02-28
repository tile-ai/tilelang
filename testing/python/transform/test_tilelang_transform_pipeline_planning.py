from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm.tir.stmt_functor import post_order_visit

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def _collect_pipeline_loop_annotations(func):
    annos = []

    def _visit(node):
        if isinstance(node, tvm.tir.For) and "software_pipeline_stage" in node.annotations:
            annos.append(node.annotations)

    post_order_visit(func.body, _visit)
    return annos


def test_simple_pipeline():
    @T.prim_func
    def before(A: T.Tensor((1024, 32), T.float32), B: T.Tensor((32, 1024), T.float32), C: T.Tensor((1024, 1024), T.float32)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    @T.prim_func
    def after(A: T.Tensor((1024, 32), T.float32), B: T.Tensor((32, 1024), T.float32), C: T.Tensor((1024, 1024), T.float32)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.serial(
                32,
                annotations={
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [T.int32(0), T.int32(1), T.int32(2)],
                    "software_pipeline_stage": [T.int32(3), T.int32(3), T.int32(3)],
                },
            ):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before, after)


def test_pipeline_planning_recognizes_explicit_cp_async_copy_stage():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.Pipelined(4, num_stages=2):
            with T.block():
                T.ptx_cp_async(
                    T.access_ptr(S[i * 4], "w", 4),
                    T.access_ptr(A[i * 4], "r", 4),
                    4,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            with T.block():
                B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    assert 0 in stages, "Expected explicit cp.async producer to be recognized as stage-0 copy stage"


def test_pipeline_planning_binds_commit_to_cp_async_stage():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.Pipelined(4, num_stages=2):
            with T.block():
                T.ptx_cp_async(
                    T.access_ptr(S[i * 4], "w", 4),
                    T.access_ptr(A[i * 4], "r", 4),
                    4,
                )
            with T.block():
                T.ptx_commit_group()
            with T.block():
                B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    orders = [int(v) for v in annos[0]["software_pipeline_order"]]
    assert len(stages) == 3, f"Expected 3 pipeline stages for 3 statements, got {len(stages)}"
    assert stages[0] == stages[1], f"Expected cp.async and commit to be in the same stage, got stages={stages}"
    assert orders[0] < orders[1], f"Expected cp.async to be ordered before commit in the same stage, got orders={orders}"


def test_pipeline_planning_binds_wait_to_cp_async_consumer_stage():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.Pipelined(4, num_stages=2):
            with T.block():
                T.ptx_cp_async(
                    T.access_ptr(S[i * 4], "w", 4),
                    T.access_ptr(A[i * 4], "r", 4),
                    4,
                )
            with T.block():
                T.ptx_commit_group()
            with T.block():
                T.ptx_wait_group(0)
            with T.block():
                B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    orders = [int(v) for v in annos[0]["software_pipeline_order"]]
    assert len(stages) == 4, f"Expected 4 pipeline stages for 4 statements, got {len(stages)}"
    assert stages[0] == stages[1], f"Expected cp.async and commit to be in the same stage, got stages={stages}"
    assert stages[2] == stages[3], f"Expected wait and its dependent consumer to be in the same stage, got stages={stages}"
    assert stages[2] >= stages[1], f"Expected wait stage to not precede commit stage, got stages={stages}"
    assert orders[2] < orders[3], f"Expected wait to stay ordered before consumer, got orders={orders}"


if __name__ == "__main__":
    tilelang.testing.main()
