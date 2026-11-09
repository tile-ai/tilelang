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
    assert artifact.kernel_source


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


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_hip_dynamic_narrow_broadcast_codegen(dtype):
    from tvm import tirx

    value = tirx.Var("value", dtype)
    func = tirx.PrimFunc([value], tirx.Evaluate(tirx.Broadcast(value, 4)))
    func = func.with_attr("global_symbol", "broadcast")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    source = build(tvm.IRModule({"broadcast": func}), tvm.target.Target("rocm")).inspect_source()
    assert "value" in source
