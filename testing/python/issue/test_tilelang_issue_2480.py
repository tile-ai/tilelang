import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


LANES = 32


@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [("int8", torch.int8, -37), ("uint8", torch.uint8, 213)],
)
@tilelang.testing.requires_cuda
def test_runtime_int8_broadcast_to_32_lanes(dtype, torch_dtype, value):
    @T.prim_func
    def main(src: T.Tensor((1,), dtype), out: T.Tensor((LANES,), dtype)):
        with T.Kernel(1, threads=1) as _:
            out[0:LANES] = T.Broadcast(src[0], LANES)

    kernel = tilelang.compile(main, target="cuda")
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
            out[0:LANES] = T.Broadcast(T.cast(value, dtype), LANES)

    kernel = tilelang.compile(main, target="cuda")
    out = torch.empty(LANES, dtype=torch_dtype, device="cuda")
    kernel(out)

    torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
