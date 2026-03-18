from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    transformed = tir.transform.LowerOpaqueBlock()(transformed)
    transformed["main"] = transformed["main"].with_attr("tma_descriptor_args", {})

    # TODO: temporary remove this check
    # tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_lower_shared_barrier():
    """Test that LowerSharedBarrier converts shared.barrier buffers + barrier_init
    annotations into ptx_init_barrier_thread_count calls.

    This replaces the old test_lower_hopper_intrin_barrier which tested the
    removed tl.create_list_of_mbarrier intrinsic.
    """

    @T.prim_func
    def before():
        with T.Kernel(8):
            _ = T.launch_thread("threadIdx.x", 128)
            mbarrier = T.alloc_barrier([128, 128, 128, 128])  # noqa: F841

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.LowerSharedBarrier()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)

    main_func = mod["main"]
    body_text = main_func.script()

    # After LowerSharedBarrier, we should see ptx_init_barrier_thread_count calls
    assert "ptx_init_barrier_thread_count" in body_text
    # Should see fence_barrier_init
    assert "ptx_fence_barrier_init" in body_text
    # Should see storage_sync
    assert "tvm_storage_sync" in body_text


if __name__ == "__main__":
    tilelang.testing.main()
