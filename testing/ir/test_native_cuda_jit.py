"""Public CUDA DSL/JIT integration with actual device numerical oracles."""

import pytest
import torch

import tilelang
import tilelang.language as T
from test_native_backend_pipeline import elementwise, shared_copy


@pytest.mark.parametrize("case,size", [(elementwise, 257), (shared_copy, 128)])
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython"])
def test_public_cuda_no_ws_execution(case, size, execution_backend, tmp_path):
    kernel = tilelang.compile(
        case(),
        out_idx=[1],
        target={"kind": "cuda", "arch": "sm_100a"},
        target_host="c",
        execution_backend=execution_backend,
        pass_configs={"tl.disable_warp_specialized": True},
    )
    a = torch.arange(size, dtype=torch.float32, device="cuda")
    actual = kernel(a)
    expected = a + 1 if case is elementwise else a
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (tmp_path / "device.cu").write_text(kernel.get_kernel_source())


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython"])
@pytest.mark.parametrize("size", [0, 1, 257])
def test_dynamic_host_control_flow_and_stream(execution_backend, size, tmp_path):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        if n > 0:
            for step in T.serial(3):
                if step != 1:
                    with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
                        for i in T.Parallel(128):
                            index = block * 128 + i
                            if index < n:
                                B[index] = A[index] + T.cast(step, "float32")

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target={"kind": "cuda", "arch": "sm_100a"},
        target_host="c",
        execution_backend=execution_backend,
        pass_configs={"tl.disable_warp_specialized": True},
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.arange(size, dtype=torch.float32, device="cuda")
        actual = kernel(values)
        expected = values + 2
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (tmp_path / "device.cu").write_text(kernel.get_kernel_source())
