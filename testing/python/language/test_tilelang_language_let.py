import tilelang.testing
from tilelang import tvm as tvm
from tilelang import language as T
from tvm_tirx import ir
from tvm_tirx.ir import prim


@tilelang.jit
def test_kernel(
    A: T.Tensor((16, 16), dtype=T.float32),
):
    with T.Kernel(1, threads=128):
        b = ir.TensorLoad(A, [0, prim.Ramp(0, 1, 4)])
        T.buffer_store(A, b, [0, prim.Ramp(4, 1, 4)])


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


@tilelang.jit
def producer_bind_kernel(A: T.Tensor((64,), T.float16), B: T.Tensor((64,), T.float16)):
    with T.Kernel(1, threads=256):
        index_shared = T.alloc_shared((1,), T.int32)
        A_shared = T.alloc_shared((32,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            # Producer-side bind depends on a shared value produced in the same pipeline body.
            index_shared[0] = k * 32
            offset = index_shared[0]
            T.copy(A[offset], A_shared)
            for i in T.Parallel(32):
                B[k * 32 + i] = A_shared[i]


@tilelang.testing.requires_cuda
def test_producer_bind_kernel():
    import torch

    kernel_source = producer_bind_kernel.get_kernel_source()
    assert "tl::tma_load" in kernel_source
    source = torch.arange(64, dtype=torch.float16, device="cuda")
    output = torch.full_like(source, -1)
    producer_bind_kernel(source, output)
    torch.testing.assert_close(output, source, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
