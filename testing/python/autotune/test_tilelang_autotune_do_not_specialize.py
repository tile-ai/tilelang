"""Tests for the `do_not_specialize` parameter of @autotune decorator.

`do_not_specialize` allows users to specify parameters whose value changes
should NOT trigger re-autotuning.  The best config found on the first call
is reused regardless of the value of those parameters.
"""

import pytest
import tilelang
import tilelang.testing
from tilelang.autotuner import set_autotune_inputs
import tilelang.language as T
import torch


def get_configs():
    return [
        {"threads": 64},
        {"threads": 128},
    ]


# ---------------------------------------------------------------------------
# Kernel under test
# ---------------------------------------------------------------------------


@tilelang.autotune(
    configs=get_configs(),
    do_not_specialize=["N", "K"],
    warmup=5,
    rep=10,
    timeout=30,
)
@tilelang.jit
def matmul_do_not_spec(M, N, K, threads=128):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((64, 32), dtype)
            B_shared = T.alloc_shared((64, 32), dtype)
            C_local = T.alloc_fragment((64, 64), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=1):
                T.copy(A[by * 64, k * 32], A_shared)
                T.copy(B[bx * 64, k * 32], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * 64, bx * 64])

    return main


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _call_and_count_new_entries(fn, M, N, K, threads=None):
    """Call fn and return how many *new* cache entries were added."""
    prev_size = len(fn._tuner_cache)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float16, device="cuda")
    call_kwargs = {"M": M, "N": N, "K": K}
    if threads is not None:
        call_kwargs["threads"] = threads
    with set_autotune_inputs([a, b]):
        fn(**call_kwargs)
    new_size = len(fn._tuner_cache)
    return new_size - prev_size


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_autotune_cache():
    """Clear the shared autotune cache before each test so tests are isolated."""
    matmul_do_not_spec._tuner_cache.clear()


@tilelang.testing.requires_cuda
def test_do_not_specialize_same_key_for_different_values():
    """Changing N and K (in do_not_specialize) should reuse the same cache entry."""
    new1 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=256, K=256, threads=64)
    new2 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=512, K=512, threads=64)

    assert new1 == 1, f"First call should create 1 new cache entry, got {new1}"
    assert new2 == 0, f"do_not_specialize failed: second call with different N/K should reuse cache (0 new entries), got {new2}"


@tilelang.testing.requires_cuda
def test_do_not_specialize_new_key_for_m_param():
    """Changing M (NOT in do_not_specialize) should trigger a new autotune run."""
    new1 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=256, K=256, threads=64)
    new2 = _call_and_count_new_entries(matmul_do_not_spec, M=1024, N=256, K=256, threads=64)

    assert new1 == 1, f"First call should create 1 new cache entry, got {new1}"
    assert new2 == 1, f"Expected 1 new cache entry because M is NOT in do_not_specialize, got {new2}"


@tilelang.testing.requires_cuda
def test_do_not_specialize_kwargs_and_args():
    """do_not_specialize should work whether params are passed as args or kwargs."""
    a = torch.randn(512, 256, dtype=torch.float16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.float16, device="cuda")

    # First call: all kwargs
    with set_autotune_inputs([a, b]):
        matmul_do_not_spec(M=512, N=256, K=256, threads=64)
    prev = len(matmul_do_not_spec._tuner_cache)

    # Second call: positional args, N and K differ but are in do_not_specialize
    with set_autotune_inputs([a, b]):
        matmul_do_not_spec(512, 512, 512, threads=64)
    new = len(matmul_do_not_spec._tuner_cache)

    assert new == prev, f"do_not_specialize failed with positional args: cache grew from {prev} to {new} (expected no new entries)"


@tilelang.testing.requires_cuda
def test_do_not_specialize_threads_new_entry():
    """threads is NOT in do_not_specialize → different values = new cache entry."""
    new1 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=256, K=256, threads=64)
    new2 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=256, K=256, threads=128)

    assert new1 == 1
    assert new2 == 1, f"Changing threads (not in do_not_specialize) should create a new cache entry, got {new2}"


@tilelang.testing.requires_cuda
def test_do_not_specialize_all_specialized_params_same_cache():
    """When all params except do_not_specialize are the same, cache is shared."""
    new1 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=256, K=256, threads=64)
    new2 = _call_and_count_new_entries(matmul_do_not_spec, M=512, N=999, K=333, threads=64)

    assert new1 == 1
    assert new2 == 0, f"All non-specialized params (M, threads) are same → should reuse cache, got {new2} new entries"


if __name__ == "__main__":
    tilelang.testing.main()
