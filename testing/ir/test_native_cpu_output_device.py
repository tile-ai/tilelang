"""A CPU target chooses CPU output storage even if CUDA is available."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as C


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable Torch environment")
@pytest.mark.parametrize("callee", [False, True])
def test_cpu_scalar_input_does_not_allocate_cuda_output(callee):
    assert torch.cuda.is_available(), "This regression requires a GPU-capable Torch environment"

    @C.prim_func
    def main(output: C.Tensor((4,), "int32"), bias: C.int32):
        with C.Kernel(1):
            for index in C.Parallel(4):
                output[index] = index + bias

    function = main.with_attr("tilelang_callee_allocated_outputs", callee)
    kernel = tilelang.compile(
        function,
        out_idx=[0],
        execution_backend="tvm_ffi",
        target={"kind": "c"},
        target_host={"kind": "c"},
    )
    result = kernel(5)
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.arange(4, dtype=torch.int32) + 5, rtol=0, atol=0)
