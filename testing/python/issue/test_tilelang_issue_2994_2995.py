import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T


@pytest.fixture(autouse=True)
def _disable_tilelang_cache():
    tilelang.disable_cache()
    try:
        yield
    finally:
        tilelang.enable_cache()


def _packed_input(dtype, n):
    values = [(((byte // 8) % 8) << 4) | (byte % 8) for byte in range(n // 2)]
    packed = torch.tensor(values, dtype=torch.uint8, device="cuda")
    return packed if dtype == "uint4" else packed.view(torch.int8)


def _copy_kernel(dtype, n, threads):
    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=threads):
            T.copy(A, B)

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("dtype", "storage_type"), [("int4", "int8_t"), ("uint4", "uint8_t")])
def test_width_two_packed_integer_copy_codegen_and_round_trip(dtype, storage_type):
    n = 256
    kernel = _copy_kernel(dtype, n, threads=128)
    compiled = tilelang.compile(kernel, out_idx=[1])

    assert storage_type in compiled.get_kernel_source()
    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


if __name__ == "__main__":
    tilelang.testing.main()
