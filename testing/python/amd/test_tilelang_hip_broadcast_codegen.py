import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _kernel(dtype):
    @T.prim_func
    def main(X: T.Tensor((128, 4), "float32"), O: T.Tensor((128, 4), dtype)):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((128, 4), dtype)
            for i, j in T.Parallel(128, 4):
                fragment[i, j] = T.cast(X[i, j], dtype) ^ T.cast(3, dtype)
            T.copy(fragment, O)

    return main


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_hip_vectorized_narrow_xor_codegen(dtype):
    target = {"kind": "hip", "mcpu": "gfx950"}
    with tvm.target.Target(target):
        artifact = tilelang.lower(_kernel(dtype), target=target)
    # This marker is emitted only for the packed four-lane broadcast path.
    assert "0x01010101u" in artifact.kernel_source


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_hip_vectorized_narrow_xor_runtime(dtype):
    x = torch.arange(512, device="cuda", dtype=torch.float32).reshape(128, 4) % 256
    if dtype == "int8":
        x = x - 128
    output = torch.empty_like(x, dtype=getattr(torch, dtype))
    kernel = tilelang.compile(_kernel(dtype), target="hip")
    kernel(x, output)
    torch.testing.assert_close(output, x.to(getattr(torch, dtype)) ^ 3, rtol=0, atol=0)


def _broadcast_kernel(dtype):
    @T.prim_func
    def main(value: T.int32, out: T.Tensor((4,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(4):
                out[i] = T.cast(value, dtype)

    return main


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_hip_dynamic_narrow_broadcast_codegen(dtype):
    target = {"kind": "hip", "mcpu": "gfx950"}
    with tvm.target.Target(target):
        artifact = tilelang.lower(_broadcast_kernel(dtype), target=target)
    assert "0x01010101u" in artifact.kernel_source


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype,values", [("int8", [-128, -1, 0, 127]), ("uint8", [0, 127, 128, 255])])
def test_hip_dynamic_narrow_broadcast_runtime(dtype, values):
    kernel = tilelang.compile(_broadcast_kernel(dtype), target="hip")
    for value in values:
        output = torch.empty(4, device="cuda", dtype=getattr(torch, dtype))
        kernel(value, output)
        torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)
