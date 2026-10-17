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


if __name__ == "__main__":
    tilelang.testing.main()
