"""Numeric integer-to-FP4 casts must not select the raw storage constructor."""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8", "uint32", "bool"])
def test_fp4_integer_cast_preserves_numeric_values(dtype, vectorized):
    @T.prim_func
    def main(A: T.Tensor((4,), dtype), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                scratch = T.alloc_local((4,), "float4_e2m1fn")
                if vectorized:
                    for i in T.vectorized(4):
                        scratch[i] = T.Cast("float4_e2m1fn", A[i])
                    for i in T.vectorized(4):
                        B[i] = T.Cast("float32", scratch[i])
                else:
                    for i in T.serial(4):
                        scratch[i] = T.Cast("float4_e2m1fn", A[i])
                    for i in T.serial(4):
                        B[i] = T.Cast("float32", scratch[i])

    kernel = tilelang.compile(main, target="cuda", target_host="c", execution_backend="cython")
    samples = {
        "int32": [[-5, -1, 1, 5], [-9, -6, 2, 9]],
        "int64": [[-(1 << 62), 0, 1 << 62, 5]],
        "uint8": [[0, 1, 2, 255], [3, 4, 5, 6]],
        "uint32": [[0, 1, 7, (1 << 32) - 1]],
        "bool": [[False, True, True, False]],
    }
    # E2M1 finite values in positive encoding order. At an integer tie (5),
    # encoding 6 (4.0) is even, so round-nearest-even chooses it over 6.0.
    representable = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype="float64")
    for values in samples[dtype]:
        magnitudes = np.minimum(np.abs(np.asarray(values, dtype="float64")), 6)
        nearest = np.argmin(np.abs(magnitudes[:, None] - representable), axis=1)
        expected = representable[nearest] * np.where(np.asarray(values) < 0, -1, 1)
        a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
        storage = torch.full((8,), 91.0, dtype=torch.float32, device="cuda")
        output = storage[:4]
        for _ in range(3):
            storage.fill_(91)
            kernel(a, output)
            torch.cuda.synchronize()
            np.testing.assert_array_equal(output.cpu().numpy(), expected.astype("float32"))
            assert torch.equal(storage[4:], torch.full_like(storage[4:], 91))
