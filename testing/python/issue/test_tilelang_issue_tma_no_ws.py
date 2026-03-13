import re

import tilelang
import tilelang.testing
from tilelang import language as T
import torch


def _compile_tvm_ffi(func, pass_configs, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs,
            **kwargs,
        )
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_lower_no_warp_specialized_injects_mbarrier():
    """Regression for Hopper TMA lowering when warp specialization is disabled.

    When `tl.disable_tma_lower=False` but `tl.disable_warp_specialized=True`, the
    optimization pipeline must still run the TMA barrier allocation/injection
    passes so generated CUDA source defines and uses `mbarrier[...]` correctly.
    """

    M, K = 16, 128
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    tilelang.disable_cache()
    kernel = _compile_tvm_ffi(tma_copy, pass_configs)

    src = kernel.get_kernel_source()
    print(src)
    assert "tl::tma_load" in src
    assert "mbarrier_mem" in src
    assert "arrive_and_expect_tx" in src

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    kernel(x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_lower_no_warp_specialized_2d_descriptor_uses_args1_barrier():
    """Cover the 2D-descriptor TMA barrier rewrite path (barrier at args[1])."""

    M, K = 16, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy_2d_desc(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

    kernel = _compile_tvm_ffi(tma_copy_2d_desc, pass_configs)

    src = kernel.get_kernel_source()
    assert "CUtensorMap" in src
    assert "tl::tma_load" in src

    flat_src = " ".join(src.split())
    pattern = r"tl::tma_load\([^,]+,\s*mbarrier\[[0-9]+\]"
    assert re.search(pattern, flat_src), (
        f"Expected regex {pattern!r} to match flattened CUDA source. Generated source (truncated):\n{src[:1000]}"
    )

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    kernel(x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_pure_tma_warp_specialized_does_not_emit_cp_async_arrive():
    """Pure TMA warp specialization should release with mbarrier.arrive only."""

    M = N = K = 128
    block_m = block_n = 128
    block_k = 32
    num_stages = 2
    threads = 128
    block_mask_shape = (M // block_m, N // block_n, K // block_k)

    @T.prim_func
    def sparse_gemm(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            B_shared = T.alloc_shared((block_k, block_n), dtype=T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)
            C_shared = T.alloc_shared((block_m, block_n), dtype=T.float16)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                if BlockMask[by, bx, k]:
                    T.copy(A[by * block_m, k * block_k], A_shared)
                    T.copy(B[k * block_k, bx * block_n], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(sparse_gemm, pass_configs, out_idx=[3])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "mbarrier_cp_async_arrive" not in src
    assert src.count(".init(1);") == 3
    assert src.count(".init(128);") == 2
    flat_src = " ".join(src.split())
    assert re.search(
        r"mbarrier\[\(k_1 & 1\)\]\.wait\([^;]+\);\s*if \(\(bool\)BlockMask",
        flat_src,
    )
    assert re.search(
        r"\}\s*if \(tl::tl_shuffle_elect<128>\(\)\)\s*\{\s*mbarrier\[\(k & 1\)\]\.arrive\(\);\s*\}",
        flat_src,
    )

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    block_mask = torch.ones(block_mask_shape, device="cuda", dtype=torch.bool)

    c = kernel(a, b, block_mask)
    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()


if __name__ == "__main__":
    # tilelang.testing.main()
    test_tma_lower_no_warp_specialized_injects_mbarrier()
