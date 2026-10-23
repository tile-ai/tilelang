import pytest
import torch

import tilelang
import tilelang.cpu.language as T


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_cpu_clz(dtype):
    bits = int(dtype.lstrip("uint"))
    values = [0, -1, -(1 << 63), (1 << 63) - 1]
    values += [(1 << bit) + delta for bit in range(bits - 1) for delta in (-1, 0, 1)]
    size = len(values)

    @T.prim_func
    def main(A: T.Tensor((size,), "int64"), B: T.Tensor((size,), "int32")):
        for i in T.serial(size):
            B[i] = T.clz(T.cast(A[i], dtype))

    kernel = tilelang.compile(main, target="c", target_host="c", execution_backend="cython")
    assert f"tl_clz{bits}" in kernel.get_kernel_source()
    a = torch.tensor(values, dtype=torch.int64)
    b = torch.empty(size, dtype=torch.int32)
    kernel(a, b)
    expected = torch.tensor([bits - (value % (1 << bits)).bit_length() for value in values], dtype=torch.int32)
    torch.testing.assert_close(b, expected, rtol=0, atol=0)
