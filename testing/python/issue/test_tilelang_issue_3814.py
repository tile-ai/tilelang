# Regression test for https://github.com/tile-ai/tilelang/issues/3814
# On sm_100+ TileLang enables 256-bit vectorization for global-only access,
# which turns a 16-bit elementwise loop over 16 elements per thread into
# int16x16 / uint16x16. CUDA codegen only knew how to print 16-bit integer
# vectors up to x8, so lowering failed with:
#   Fatal: Cannot convert type int16x16 to CUDA type
# The source checks below need the CUDA backend but no Blackwell GPU (or even a
# CUDA device). The runtime check exercises the same kernel on whatever CUDA GPU
# runs the test suite.
import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T, tvm
from tilelang.engine.lower import lower as tilelang_lower


def _elementwise_kernel(dtype):
    @T.prim_func
    def main(X: T.Tensor((512,), dtype), O: T.Tensor((512,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(512):
                O[i] = X[i] + X[i]

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_120a"])
def test_16bit_integer_256bit_vector_codegen(dtype, arch):
    target = {"kind": "cuda", "arch": arch}
    with tvm.target.Target(target):
        artifact = tilelang_lower(_elementwise_kernel(dtype), target=target)

    source = artifact.kernel_source
    vector_type = "ulonglong4" if dtype.startswith("uint") else "longlong4"
    assert vector_type in source
    assert "tl::load_global_256" in source
    assert "tl::store_global_256" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype, torch_dtype, low, high",
    [
        ("int16", torch.int16, -1000, 1000),
        ("uint16", torch.uint16, 0, 30000),
    ],
)
def test_16bit_integer_vector_add_numeric(dtype, torch_dtype, low, high):
    kernel = tilelang.compile(_elementwise_kernel(dtype), out_idx=[1])
    x = torch.randint(low, high, (512,), dtype=torch_dtype, device="cuda")
    out = kernel(x)
    expected = (x.to(torch.int64) + x.to(torch.int64)).to(torch_dtype)
    assert torch.equal(out, expected)


if __name__ == "__main__":
    tilelang.testing.main()
