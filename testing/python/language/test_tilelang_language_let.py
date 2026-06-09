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


@tilelang.jit
def bind_kernel(A: T.Tensor((64,), T.float16), B: T.Tensor((64,), T.float16)):
    with T.Kernel(1, threads=32):
        A_shared = T.alloc_shared((32,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(A[k * 32], A_shared)
            for i in T.Parallel(32):
                x = A_shared[i] + 1.0
                B[k * 32 + i] = x


@tilelang.testing.requires_cuda
def test_bind_kernel():
    kernel_source = bind_kernel.get_kernel_source()
    assert "float x" in kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
