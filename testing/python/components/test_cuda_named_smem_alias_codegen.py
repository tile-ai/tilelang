import tilelang
import tilelang.language as T
import tilelang.testing


PASS_CFG_BASE = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _two_dyn_shared_buffers():
    @T.prim_func
    def main(
        A: T.Tensor((128,), T.float16),
        B: T.Tensor((128,), T.float16),
        C: T.Tensor((128,), T.float16),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((128,), T.float16, scope="shared.dyn")
            B_shared = T.alloc_shared((128,), T.float16, scope="shared.dyn")
            for i in T.Parallel(128):
                A_shared[i] = A[i]
                B_shared[i] = B[i]
                C[i] = A_shared[i] + B_shared[i]

    return main


def _pipelined_dyn_shared_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((256, 256), T.float16),
        B: T.Tensor((256, 256), T.float16),
        C: T.Tensor((256, 256), T.float16),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float16, scope="shared.dyn")
            B_shared = T.alloc_shared((32, 128), T.float16, scope="shared.dyn")
            C_local = T.alloc_fragment((128, 128), T.float16)
            T.clear(C_local)
            for k in T.Pipelined(8, num_stages=3):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * 128, bx * 128])

    return main


def _compile_kernel(pass_configs=None, program_fn=_two_dyn_shared_buffers):
    cfg = dict(PASS_CFG_BASE)
    if pass_configs:
        cfg.update(pass_configs)
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    try:
        return tilelang.compile(program_fn(), out_idx=[2], target="cuda", pass_configs=cfg)
    finally:
        if cache_enabled:
            tilelang.enable_cache()


def test_codegen_keeps_raw_dyn_smem_access_by_default():
    kernel = _compile_kernel()
    src = kernel.get_kernel_source()
    print(src)

    assert "buf_dyn_shmem" in src
    assert "A_shared = reinterpret_cast" not in src
    assert "B_shared = reinterpret_cast" not in src


@tilelang.testing.requires_cuda
def test_codegen_emits_named_dyn_smem_aliases_when_enabled():
    kernel = _compile_kernel({tilelang.PassConfigKey.TL_EMIT_NAMED_SMEM_POINTERS: True})
    src = kernel.get_kernel_source()
    print(src)

    assert "buf_dyn_shmem" in src
    assert "A_shared = reinterpret_cast<half_t*>(buf_dyn_shmem + 0)" in src
    assert "B_shared = reinterpret_cast<half_t*>(buf_dyn_shmem + 256)" in src
    assert "A_shared[((int)threadIdx.x)]" in src or "A_shared[threadIdx.x]" in src
    assert "B_shared[((int)threadIdx.x)]" in src or "B_shared[threadIdx.x]" in src


@tilelang.testing.requires_cuda
def test_codegen_emits_pattern_visitors_for_multiversion_dyn_smem_aliases():
    kernel = _compile_kernel(
        {tilelang.PassConfigKey.TL_EMIT_NAMED_SMEM_POINTERS: True},
        program_fn=_pipelined_dyn_shared_gemm,
    )
    src = kernel.get_kernel_source()
    print(src)

    assert "auto A_shared = tl::PatternVisitor" in src
    assert "auto B_shared = tl::PatternVisitor" in src
    assert "A_shared[(k % 3)]" in src or "A_shared[k % 3]" in src
    assert "B_shared[(k % 3)]" in src or "B_shared[k % 3]" in src
    assert "A_shared[1][0]" in src
    assert "B_shared[1][0]" in src


if __name__ == "__main__":
    tilelang.testing.main()
