import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["cython", "nvrtc", "tvm_ffi"])
def test_uint64_tensor(execution_backend):
    @T.prim_func
    def main(A: T.Tensor((128,), "uint64"), B: T.Tensor((128,), "uint64")):
        with T.Kernel(1, threads=128):
            i = T.get_thread_binding(0)
            B[i] = A[i] + T.uint64(1)

    values = [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1] * 25 + [2, 3, 4]
    a = torch.tensor(values, dtype=torch.uint64, device="cuda")
    kernel = tilelang.compile(main, out_idx=[1], target="cuda", execution_backend=execution_backend)
    assert kernel(a).cpu().tolist() == [(x + 1) % (1 << 64) for x in values]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["cython", "nvrtc"])
def test_uint64_scalar(execution_backend):
    @T.prim_func
    def main(B: T.Tensor((1,), "uint64"), value: T.uint64):
        with T.Kernel(1, threads=1):
            B[0] = value

    kernel = tilelang.compile(main, target="cuda", execution_backend=execution_backend)
    b = torch.empty((1,), dtype=torch.uint64, device="cuda")
    for value in [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]:
        kernel(b, value)
        assert b.item() == value


if __name__ == "__main__":
    tilelang.testing.main()
