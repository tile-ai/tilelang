"""Tests for per-config pass_configs auto-tuning support."""

import itertools

import tilelang.testing
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner, autotune
from tilelang.transform import PassConfigKey


def _get_test_configs():
    """Generate test configs using Cartesian product, including pass_configs variations."""
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [64, 128]
    warp_spec = [True, False]
    _configs = list(itertools.product(block_M, block_N, block_K, warp_spec))
    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "pass_configs": {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: c[3]},
        }
        for c in _configs
    ]


def _get_test_configs_mixed():
    """Generate configs where some have pass_configs and others don't."""
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [64, 128]
    configs = []
    for bm, bn, bk in itertools.product(block_M, block_N, block_K):
        # Some configs with pass_configs
        configs.append(
            {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
                "pass_configs": {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
            }
        )
        # Some configs without pass_configs (use global default)
        configs.append(
            {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
            }
        )
    return configs


def _make_matmul_kernel(M, N, K):
    """Helper to create a simple matmul kernel factory."""

    def kernel(block_M=None, block_N=None, block_K=None):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


@tilelang.testing.requires_cuda
def test_autotune_with_per_config_pass_configs():
    """Test that per-config pass_configs are correctly applied during autotuning."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    configs = _get_test_configs()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None
    assert result.kernel is not None


@tilelang.testing.requires_cuda
def test_autotune_mixed_pass_configs():
    """Test configs where some have pass_configs and others don't."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    configs = _get_test_configs_mixed()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None


@tilelang.testing.requires_cuda
def test_autotune_pass_configs_merge_with_global():
    """Test that per-config pass_configs merge correctly over global pass_configs."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    # Global pass_configs set via set_compile_args
    global_pass_configs = {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}

    configs = _get_test_configs()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
            pass_configs=global_pass_configs,
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None


@tilelang.testing.requires_cuda
def test_autotune_decorator_with_per_config_pass_configs():
    """Test @autotune + @tilelang.jit decorator pattern with per-config pass_configs."""
    M, N, K = 128, 128, 128

    configs = _get_test_configs()

    @autotune(configs=configs, warmup=1, rep=3, skip_check=True, supply_type=tilelang.TensorSupplyType.Integer)
    @tilelang.jit(out_idx=[-1])
    def matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=64):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    kernel = matmul_kernel(M, N, K, __return_kernel=True)
    assert kernel is not None


@tilelang.testing.requires_cuda
def test_autotune_decorator_pass_configs_override_jit_global():
    """Test that per-config pass_configs in @autotune override @tilelang.jit's global pass_configs."""
    M, N, K = 128, 128, 128

    configs = _get_test_configs_mixed()

    @autotune(configs=configs, warmup=1, rep=3, skip_check=True, supply_type=tilelang.TensorSupplyType.Integer)
    @tilelang.jit(out_idx=[-1], pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=64):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    kernel = matmul_kernel(M, N, K, __return_kernel=True)
    assert kernel is not None


if __name__ == "__main__":
    tilelang.testing.main()
