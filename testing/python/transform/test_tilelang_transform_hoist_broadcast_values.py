import tilelang
import tilelang.language as T
import torch
import re
import pytest
import tilelang.testing


@tilelang.jit
def qwq(dtype=torch.float8_e4m3fn):
    @T.prim_func
    def main(
        A: T.Tensor((32,), dtype),
        B: T.Tensor((16,), dtype),
        C: T.Tensor((8,), dtype),
        D: T.Tensor((4,), dtype),
        E: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, threads=32):
            var = T.alloc_var(dtype, 1.0)
            for i in T.vectorized(32):
                A[i] = var
            for i in T.vectorized(16):
                B[i] = 13.5
            for i in T.vectorized(8):
                C[i] = 3.14
            for i in T.vectorized(4):
                D[i] = 2.72
            for i in T.vectorized(2):
                E[i] = 430

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu, torch.float16])
def test_broadcast(dtype):
    kernel = qwq(dtype)
    print(kernel.get_kernel_source())
    matches = re.findall(r"(\w+) broadcast_var(_[0-9]+)? = \1", kernel.get_kernel_source())
    assert len(matches) == 4
    a = torch.empty((32,), device="cuda", dtype=dtype)
    b = torch.empty((16,), device="cuda", dtype=dtype)
    c = torch.empty((8,), device="cuda", dtype=dtype)
    d = torch.empty((4,), device="cuda", dtype=dtype)
    e = torch.empty((2,), device="cuda", dtype=dtype)
    kernel(a, b, c, d, e)


if __name__ == "__main__":
    tilelang.testing.main()
