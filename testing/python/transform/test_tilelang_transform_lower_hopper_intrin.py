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


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_descriptor_init_after_alloc_global():
    @T.prim_func
    def before():
        T.func_attr({"tir.is_entry_func": True, "tl.has_tma": T.bool(True)})
        Output_partial = T.allocate([32], "float16", "global")
        with T.launch_thread("threadIdx.x", 1):
            T.evaluate(
                T.create_tma_descriptor(
                    6,
                    4,
                    Output_partial,
                    8,
                    2,
                    2,
                    1,
                    2,
                    16,
                    32,
                    64,
                    8,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    func = mod["main"]

    assert not tvm.tir.analysis.undefined_vars(func.body, func.params)
    body_text = func.script()
    assert body_text.index('T.allocate([32], "float16", "global")') < body_text.index('T.call_packed("__tvm_tensormap_create_tiled"')


if __name__ == "__main__":
    tilelang.testing.main()
