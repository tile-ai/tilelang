"""Regression tests for sub-word integer wrap in C-like codegen (issue #2961)."""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.cpu.language as TC
import tilelang.language as T
import tilelang.testing

N = 64
# uint16 b * b overflows int and int16 has negative b << k: both must be done unsigned.
DTYPES = ["uint8", "int16", "uint16"]


def _kernel(lang, dtype, **launch):
    k = np.dtype(dtype).itemsize * 4

    @lang.prim_func
    def main(A: lang.Tensor((N,), dtype), B: lang.Tensor((N * 5,), dtype)):
        with lang.Kernel(1, **launch):
            for i in lang.serial(N):
                b = A[i]
                # Interleaved stores keep the scalar path; each intermediate must wrap.
                B[i * 5 + 0] = (b << k) >> k
                B[i * 5 + 1] = (b + b) >> k
                B[i * 5 + 2] = (b - lang.cast(100, dtype)) >> k
                B[i * 5 + 3] = (b * b) >> k
                B[i * 5 + 4] = (~b) >> k

    return main


def _reference(a):
    k = a.itemsize * 4
    with np.errstate(over="ignore"):
        cols = [a << k, a + a, a - a.dtype.type(100), a * a, ~a]
    return np.stack([c >> k for c in cols], axis=1).reshape(-1)


def _input(dtype):
    return (np.arange(N) * 1031).astype(dtype)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_subword_int_ops_wrap_cuda(dtype):
    a = _input(dtype)
    kernel = tilelang.compile(_kernel(T, dtype, threads=1), out_idx=[1], target="cuda")
    np.testing.assert_array_equal(kernel(torch.from_numpy(a).cuda()).cpu().numpy(), _reference(a))


@pytest.mark.parametrize("dtype", DTYPES)
def test_subword_int_ops_wrap_c(dtype):
    a = _input(dtype)
    kernel = tilelang.compile(_kernel(TC, dtype), out_idx=[1], target="c", target_host="c", execution_backend="cython")
    np.testing.assert_array_equal(kernel(torch.from_numpy(a)).numpy(), _reference(a))


@tilelang.testing.requires_cuda
def test_int32_ops_not_cast():
    src = tilelang.compile(_kernel(T, "int32", threads=1), out_idx=[1], target="cuda").get_kernel_source()
    assert "((int)(" not in src


if __name__ == "__main__":
    tilelang.testing.main()
