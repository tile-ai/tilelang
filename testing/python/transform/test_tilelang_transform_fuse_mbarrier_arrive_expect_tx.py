import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count

target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
pytestmark = pytest.mark.skipif(
    tvm.get_global_func("tl.cuda.transform.FuseMBarrierArriveExpectTx", allow_missing=True) is None,
    reason="FuseMBarrierArriveExpectTx is not compiled into this build",
)


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.cuda.transform.FuseMBarrierArriveExpectTx()(mod)
    return tl.transform.LowerOpaqueBlock()(mod)


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
    assert_call_count(main, op="tirx.ptx_arrive_barrier_expect_tx", count=1)
    assert_call_count(main, op="tl.mbarrier_expect_tx", count=0)
    assert_call_count(main, op="tirx.ptx_arrive_barrier", count=0)


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
    assert_call_count(main, op="tirx.ptx_arrive_barrier_expect_tx", count=0)
    assert_call_count(main, op="tl.mbarrier_expect_tx", count=1)
    assert_call_count(main, op="tirx.ptx_arrive_barrier", count=1)


def test_fuse_inside_warp_specialization_scope():
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        tx = T.launch_thread("threadIdx.x", 256)
        smem = T.decl_buffer((32,), T.uint8, scope="shared.dyn")
        mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
        with T.attr([128, 128], "kWarpSpecializationScope", 0):
            if tx >= 128:  # noqa: SIM102 - exercise nested warp and election guards
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
    assert_call_count(main, op="tirx.ptx_arrive_barrier_expect_tx", count=1)
    assert_call_count(main, op="tl.mbarrier_expect_tx", count=0)
    assert_call_count(main, op="tirx.ptx_arrive_barrier", count=0)


if __name__ == "__main__":
    tilelang.testing.main()
