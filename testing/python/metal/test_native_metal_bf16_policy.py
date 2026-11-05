"""Metal native BF16 keeps its device element type after vector planning."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm


@pytest.mark.parametrize("operation", ["copy", "add"])
def test_native_metal_bf16_copy_and_arithmetic(operation):
    @T.prim_func
    def main(A: T.Tensor((256,), "bfloat16"), B: T.Tensor((256,), "bfloat16")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(256):
                if operation == "copy":
                    B[i] = A[i]
                else:
                    B[i] = A[i] + T.Cast("bfloat16", 1)

    target = tvm.target.Target("metal", host="c")
    artifact = tilelang.lower(main, target=target, target_host="c", enable_host_codegen=False, enable_device_compile=False)
    source = artifact.kernel_source
    assert "device bfloat*" in source
    assert "device ushort*" not in source
    assert "uint4" in source
    assert "*(device uint4*)" in source
