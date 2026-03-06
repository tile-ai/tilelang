import re

import tilelang
import tilelang.testing
from tilelang import language as T
import torch


def _compile_tvm_ffi(func, pass_configs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs,
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

    kernel = _compile_tvm_ffi(tma_copy, pass_configs)

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "mbarrier_mem" in src
    assert "expect_transaction" in src

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


if __name__ == "__main__":
    tilelang.testing.main()
