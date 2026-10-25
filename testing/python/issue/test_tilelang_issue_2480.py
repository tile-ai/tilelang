import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


LANES = 32


def _assert_native_broadcast(source, dtype):
    carrier = "ulonglong4" if dtype == "uint8" else "longlong4"
    declaration = re.search(rf"\b{carrier}\s+(\w+);", source)
    assert declaration is not None
    vector = declaration.group(1)
    pattern = rf"reinterpret_cast<(?:signed char|unsigned char|uchar)\*>\(&\({re.escape(vector)}\)\)\[(\d+)\]\s*=\s*(\w+);"
    stores = re.findall(pattern, source)
    assert [int(index) for index, _ in stores] == list(range(LANES))
    assert len({value for _, value in stores}) == 1


@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [("int8", torch.int8, -37), ("uint8", torch.uint8, 213)],
)
@tilelang.testing.requires_cuda
def test_runtime_int8_broadcast_to_32_lanes(dtype, torch_dtype, value):
    @T.prim_func
    def main(src: T.Tensor((1,), dtype), out: T.Tensor((LANES,), dtype)):
        with T.Kernel(1, threads=1) as _:
            # Scalar global stores keep this test runnable on pre-SM100 GPUs.
            packed = T.alloc_local((LANES,), dtype)
            packed[0:LANES] = T.Broadcast(src[0], LANES)
            for i in T.serial(LANES):
                out[i] = packed[i]

    kernel = tilelang.compile(main, target="cuda")
    _assert_native_broadcast(kernel.get_kernel_source(), dtype)
    src = torch.tensor([value], dtype=torch_dtype, device="cuda")
    out = torch.empty(LANES, dtype=torch_dtype, device="cuda")
    kernel(src, out)

    torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [("int8", torch.int8, -37), ("uint8", torch.uint8, 213)],
)
@tilelang.testing.requires_cuda
def test_constant_int8_broadcast_to_32_lanes(dtype, torch_dtype, value):
    @T.prim_func
    def main(out: T.Tensor((LANES,), dtype)):
        with T.Kernel(1, threads=1) as _:
            packed = T.alloc_local((LANES,), dtype)
            packed[0:LANES] = T.Broadcast(T.cast(value, dtype), LANES)
            for i in T.serial(LANES):
                out[i] = packed[i]

    kernel = tilelang.compile(main, target="cuda")
    _assert_native_broadcast(kernel.get_kernel_source(), dtype)
    out = torch.empty(LANES, dtype=torch_dtype, device="cuda")
    kernel(out)

    torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
