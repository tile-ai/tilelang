import tilelang.testing
from tilelang import tvm as tvm
from tilelang import language as T


@tilelang.jit
def test_kernel(
    A: T.Tensor((16, 16), dtype=T.float32),
):
    for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
        for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
            b = A[0, 0:4]
            A[0, 4:8] = b


@tilelang.testing.requires_cuda
def test_let_vectorize_load():
    kernel_source = test_kernel.get_kernel_source()
    assert "float4 b" in kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
