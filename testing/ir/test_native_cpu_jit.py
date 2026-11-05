"""Exercise the public CPU JIT through the complete registered pass pipeline."""

import pytest
import torch
import tvm_tirx

import tilelang
import tilelang.cpu.language as T


def add_one(n):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                B[i] = A[i] + 1.0

    return main


@pytest.mark.parametrize("size", [0, 1, 17, 257])
def test_public_cpu_compile_and_dynamic_output(size):
    n = T.dynamic("n", "int64")
    kernel = tilelang.compile(
        add_one(n),
        out_idx=[1],
        target=tvm_tirx.target.Target({"kind": "c"}),
        target_host=tvm_tirx.target.Target({"kind": "c"}),
        execution_backend="tvm_ffi",
    )
    values = torch.arange(size, dtype=torch.float32)
    output = kernel(values)
    torch.testing.assert_close(output, values + 1, rtol=0, atol=0)
    assert output.shape == values.shape
