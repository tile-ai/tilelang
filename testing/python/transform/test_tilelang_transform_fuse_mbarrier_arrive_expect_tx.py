# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.FuseMBarrierArriveExpectTx()(mod)
    return tir.transform.LowerOpaqueBlock()(mod)


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def test_fuse_simple_tma_expect_arrive():
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((16,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            if T.shuffle_elect(0):
                T.mbarrier_expect_tx(mbarrier[0], 16)
                T.tma_load(
                    A_desc,
                    mbarrier[0],
                    T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                    0,
                    0,
                    0,
                )
                T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(before)
    main = mod["main"]
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier_expect_tx")) == 1
    assert len(_collect_calls(main.body, "tl.mbarrier_expect_tx")) == 0
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier")) == 0


def test_fuse_requires_same_barrier():
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((16,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((2,), T.uint64, scope="shared.barrier")
            if T.shuffle_elect(0):
                T.mbarrier_expect_tx(mbarrier[0], 16)
                T.tma_load(
                    A_desc,
                    mbarrier[0],
                    T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                    0,
                    0,
                    0,
                )
                T.ptx_arrive_barrier(mbarrier[1])

    mod = _apply(before)
    main = mod["main"]
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier_expect_tx")) == 0
    assert len(_collect_calls(main.body, "tl.mbarrier_expect_tx")) == 1
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier")) == 1


def test_fuse_inside_warp_specialization_scope():
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        tx = T.launch_thread("threadIdx.x", 256)
        smem = T.decl_buffer((32,), T.uint8, scope="shared.dyn")
        mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
        with T.attr([128, 128], "kWarpSpecializationScope", 0):
            if tx >= 128:
                if T.shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier[0], 32)
                    T.tma_load(
                        A_desc,
                        mbarrier[0],
                        T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                        0,
                        0,
                        0,
                    )
                    T.tma_load(
                        A_desc,
                        mbarrier[0],
                        T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 16, 16, 2),
                        16,
                        0,
                        0,
                    )
                    T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(before)
    main = mod["main"]
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier_expect_tx")) == 1
    assert len(_collect_calls(main.body, "tl.mbarrier_expect_tx")) == 0
    assert len(_collect_calls(main.body, "tir.ptx_arrive_barrier")) == 0


if __name__ == "__main__":
    tilelang.testing.main()
