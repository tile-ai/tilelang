"""Tests for implicit SIMT floating-point RNG fusion in Ascend codegen."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T


def _rng_fill(groups, dtype, distribution, seed=42, seq=7, off=12):
    threads = 64
    n = groups * 2 * threads

    @T.prim_func
    def main(
        scalar_rng: T.Tensor((n,), dtype),
        parallel_rng: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1):
            scalar_ub = T.alloc_shared((n,), dtype)
            parallel_ub = T.alloc_shared((n,), dtype)

            with T.SimtVF(threads=threads):
                tx = T.get_thread_binding()
                T.rng_init(seed, seq=seq + tx, off=off)
                for i in T.serial(groups * 2):
                    scalar_ub[i * threads + tx] = T.cast(T.rng_rand_float(dist=distribution), dtype)

            with T.SimtVF(threads=threads):
                T.rng_init(seed, seq=seq + T.get_thread_binding(), off=off)
                for i in T.Parallel(n):
                    parallel_ub[i] = T.cast(T.rng_rand_float(dist=distribution), dtype)

            T.copy(scalar_ub, scalar_rng)
            T.copy(parallel_ub, parallel_rng)

    return main


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_parallel_rng_is_implicitly_fused_and_bitwise_equal_up_to_layout(dtype, distribution):
    groups = 16
    threads = 64
    n = groups * 2 * threads
    kernel = tilelang.compile(_rng_fill(groups, dtype, distribution), target="ascend")
    source = kernel.get_kernel_source()

    vector_name = f"philox_rand_{distribution}2"
    assert source.count(f"tl::{vector_name}(&") == 1
    assert f"{vector_name}_bfloat16" not in source
    assert ("__float22bfloat162_rn" in source) == (dtype == "bfloat16")

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    scalar = torch.empty(n, dtype=torch_dtype, device="npu")
    parallel = torch.empty_like(scalar)
    kernel(scalar, parallel)
    torch.npu.synchronize()

    integer_dtype = torch.int32 if dtype == "float32" else torch.int16
    scalar_bits = scalar.cpu().view(integer_dtype).view(groups, 2, threads).permute(0, 2, 1).contiguous().view(n)
    assert torch.equal(
        scalar_bits,
        parallel.cpu().view(integer_dtype),
    )


if __name__ == "__main__":
    tilelang.testing.main()
