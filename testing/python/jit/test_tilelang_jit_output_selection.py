import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@T.prim_func
def copy_output(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
    with T.Kernel(1, threads=128):
        i = T.get_thread_binding(0)
        B[i] = A[i] + 1


@T.prim_func
def sum_output(
    A: T.Tensor((128,), "float32"),
    B: T.Tensor((128,), "float32"),
    C: T.Tensor((128,), "float32"),
    D: T.Tensor((128,), "float32"),
):
    with T.Kernel(1, threads=128):
        i = T.get_thread_binding(0)
        D[i] = A[i] + B[i] + C[i]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["cython", "nvrtc", "tvm_ffi"])
def test_reused_output_indices(execution_backend):
    output_indices = [-1]
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    b = torch.full_like(a, 1000)
    c = torch.full_like(a, 2000)

    for _ in range(2):
        copy_kernel = tilelang.compile(copy_output, out_idx=output_indices, execution_backend=execution_backend)
        sum_kernel = tilelang.compile(sum_output, out_idx=output_indices, execution_backend=execution_backend)
        for kernel, inputs, expected in [(copy_kernel, (a,), a + 1), (sum_kernel, (a, b, c), a + b + c)]:
            actual = kernel(*inputs)
            if not torch.equal(actual, expected):
                pytest.fail(f"Incorrect output selection for {execution_backend}: {actual.cpu().tolist()}")
        if output_indices != [-1]:
            pytest.fail(f"Compilation changed the caller's output indices to {output_indices}")
        a.add_(1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["cython", "nvrtc", "tvm_ffi"])
@pytest.mark.parametrize("output_indices", [2, -3, [2], [-3], [-1, 2]])
def test_invalid_output_indices(execution_backend, output_indices):
    original = output_indices.copy() if isinstance(output_indices, list) else output_indices
    with pytest.raises(ValueError):
        tilelang.compile(copy_output, out_idx=output_indices, execution_backend=execution_backend)
    if output_indices != original:
        pytest.fail(f"Failed compilation changed the caller's output indices to {output_indices}")


@tilelang.testing.requires_cuda
def test_grouped_output_selection_and_export(tmp_path):
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.jit.kernel import JITKernel

    args = CompileArgs(out_idx=[-1], execution_backend="tvm_ffi", target="cuda", target_host="c")
    results = compile_grouped_unit_tvm_ffi([(0, {"program": copy_output}), (1, {"program": sum_output})], args, lambda program: program)
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    for idx, _, kernel, error in results:
        assert error is None, error
        assert kernel.artifact.target_host.kind.name == "c"
        inputs, expected = ((a,), a + 1) if idx == 0 else ((a, a, a), a * 3)
        torch.testing.assert_close(kernel(*inputs), expected)
        path = str(tmp_path / f"kernel_{idx}.so")
        kernel.export_library(path)
        restored = JITKernel.from_database(
            func=kernel.prim_func,
            host_kernel_source=kernel.get_host_source(),
            device_kernel_source=kernel.get_kernel_source(),
            kernel_lib_path=path,
            params=kernel.params,
            target=kernel.target,
            target_host=kernel.target_host,
            out_idx=[-1],
            execution_backend="tvm_ffi",
        )
        torch.testing.assert_close(restored(*inputs), expected)
    assert len(results) == 2
    assert args.out_idx == [-1]


if __name__ == "__main__":
    tilelang.testing.main()
