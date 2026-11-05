"""The native and source CUDA ABIs consume the same integer source defaults."""

import pytest
import torch

import tilelang
import tilelang.language as T


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "nvrtc"])
@pytest.mark.parametrize("extent", [0, 1, 7, 257])
def test_marked_nonlinear_shape_executes_on_cuda(execution_backend, extent):
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution requires an available CUDA device")
    n = T.int32(is_size_var=True)
    dimension = T.max(n, 0)

    @T.prim_func
    def kernel(A: T.Tensor((dimension,), "int32"), B: T.Tensor((dimension,), "int32")):
        with T.Kernel(T.max(T.ceildiv(n, 128), 1), threads=128) as block:
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < n:
                    B[index] = A[index] + 3

    executable = tilelang.compile(kernel, target="cuda", execution_backend=execution_backend)
    source = torch.arange(extent, dtype=torch.int32, device="cuda") * 5 - 4
    output = torch.full((extent,), -991, dtype=torch.int32, device="cuda")
    executable(source, output)
    torch.testing.assert_close(output, source + 3, rtol=0, atol=0)
