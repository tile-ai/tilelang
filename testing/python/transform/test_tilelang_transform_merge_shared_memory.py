import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower as tilelang_lower
from tvm import tirx


def _apply_merge(func, enable_aggressive_merge: bool = True) -> str:
    target = tvm.target.Target("cuda -arch=sm_90a")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MergeSharedMemoryAllocations(
            enable_aggressive_merge=enable_aggressive_merge
        )(mod)
    return mod.script(show_meta=False)


def _make_sequential_shared_reuse_kernel():
    @T.prim_func
    def main(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "uint8")):
        with T.Kernel(1, threads=1):
            # Model a pipelined GEMM mainloop footprint followed by an epilogue
            # staging footprint.  The epilogue buffers are born only after the
            # A/B buffers' last use, so aggressive merge should pack them into
            # the completed mainloop footprint instead of summing all buffers.
            A_shared = T.alloc_shared((65536,), "uint8")
            B_shared = T.alloc_shared((131072,), "uint8")
            A_shared[0] = A[0]
            B_shared[0] = A_shared[0]
            B[0] = B_shared[0]

            C0_shared = T.alloc_shared((32768,), "uint8")
            C1_shared = T.alloc_shared((32768,), "uint8")
            C0_shared[0] = A[0]
            C1_shared[0] = C0_shared[0]
            B[0] = C1_shared[0]

    return main


def _lower_source(enable_aggressive_merge: bool) -> str:
    target = tvm.target.Target("cuda -arch=sm_90a")
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: enable_aggressive_merge,
    }
    tilelang.disable_cache()
    try:
        with target, tvm.transform.PassContext(config=pass_configs):
            artifact = tilelang_lower(
                _make_sequential_shared_reuse_kernel(),
                target=target,
            )
    finally:
        tilelang.enable_cache()
    return artifact.device_mod.script(show_meta=False)


def test_aggressive_merge_uses_statement_level_shared_liveness():
    source = _lower_source(enable_aggressive_merge=True)
    assert 'T.allocate([196608], "uint8", "shared.dyn")' in source
    assert 'T.allocate([262144], "uint8", "shared.dyn")' not in source


def _make_ws_tma_store_conflict_kernel():
    @T.prim_func
    def main(C_desc: T.handle("uint8x128", "grid_constant")):
        tx = T.launch_thread("threadIdx.x", 256)
        A_shared = T.decl_buffer((65536,), T.uint8, scope="shared.dyn")
        C_shared = T.decl_buffer((32768,), T.uint8, scope="shared.dyn")
        with T.attr([128, 128], "kWarpSpecializationScope", 0):
            if tx >= 128:
                A_shared[0] = T.uint8(1)
            else:
                C_shared[0] = A_shared[0]
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tma_store"),
                        C_desc,
                        T.tvm_access_ptr(
                            T.type_annotation(T.uint8),
                            C_shared.data,
                            0,
                            32768,
                            1,
                        ),
                        0,
                        0,
                    )
                )

    return main


def _make_branch_reuse_kernel():
    @T.prim_func
    def main(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "uint8")):
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        if bx == 0:
            A_shared = T.decl_buffer((65536,), T.uint8, scope="shared.dyn")
            A_shared[tx] = A[0]
            B[0] = A_shared[tx]
        else:
            B_shared = T.decl_buffer((32768,), T.uint8, scope="shared.dyn")
            B_shared[tx] = A[0]
            B[0] = B_shared[tx]

    return main


def test_ws_tma_store_source_does_not_reuse_pipeline_shared():
    source = _apply_merge(_make_ws_tma_store_conflict_kernel())
    assert 'T.allocate([98304], "uint8", "shared.dyn")' in source
    assert "T.tvm_access_ptr(T.type_annotation(\"uint8\"), C_shared" not in source


def test_mutually_exclusive_branches_still_reuse_shared_memory():
    source = _apply_merge(_make_branch_reuse_kernel())
    assert 'T.allocate([65536], "uint8", "shared.dyn")' in source
    assert 'T.allocate([98304], "uint8", "shared.dyn")' not in source

if __name__ == "__main__":
    test_aggressive_merge_uses_statement_level_shared_liveness()
