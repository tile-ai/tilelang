import tilelang
import tilelang.testing
import tilelang.language as T
import pytest
import torch
from tilelang import tvm
from tilelang.contrib import nvcc


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                T.clear(A_shared)

                # Demonstrate parallelized copy from global to shared for B
                T.copy(B[bx * block_N, ko * block_K], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    program = matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    kernel = tilelang.compile(program, out_idx=[2])
    import torch

    a = torch.randn((M, K), dtype=dtype.as_torch()).cuda()
    b = torch.randn((N, K), dtype=dtype.as_torch()).cuda()
    c = kernel(a, b)
    assert torch.allclose(c, torch.zeros_like(c))


def test_matmul():
    run_matmul(1024, 1024, 1024, 128, 128, 32)


@tilelang.testing.requires_cuda_compute_version(10)
def test_shared_fill_cast_zero_uses_st_bulk():
    if nvcc.get_cuda_version() < (12, 8):
        pytest.skip("st.bulk shared lowering requires CUDA toolkit >= 12.8")

    @T.prim_func
    def main(out: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((128,), T.float32)
            for i in T.Parallel(128):
                smem[i] = T.float32(1.0)
            T.fill(smem, T.cast(0, T.float32))
            T.copy(smem, out)

    kernel = tilelang.compile(main, target="cuda")
    assert "tl::st_bulk_shared<512, 0>" in kernel.get_kernel_source()

    out = torch.empty((128,), dtype=torch.float32, device="cuda")
    kernel(out)
    torch.cuda.synchronize()
    assert torch.allclose(out, torch.zeros_like(out))


@tilelang.testing.requires_cuda
def test_fill_int8_negative():
    M, N = 8, 128

    def program(value):
        @T.prim_func
        def main(out: T.Tensor((M, N), "int8")):
            with T.Kernel(1, threads=128):
                smem = T.alloc_shared((M, N), "int8")
                T.fill(smem, value)
                T.copy(smem, out)

        return main

    for value in (-7, -128, -1, 5):
        kernel = tilelang.compile(program(value), out_idx=[0])
        out = kernel()
        torch.cuda.synchronize()
        ref = torch.full((M, N), value, dtype=torch.int8, device="cuda")
        torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("op", ["fill", "clear"])
@pytest.mark.parametrize(
    ("lo", "hi", "fits"),
    [
        (6, 12, False),  # min + extent = 12 > 8
        (2, 8, True),  # in bounds
    ],
)
def test_sliced_region_past_the_end_of_the_buffer_is_rejected(op, lo, hi, fits):
    """`extent <= shape` only bounds a region anchored at the start of the buffer.

    A sliced region may start further in, so its end is what has to fit: `As[6:12]`
    into a shape-8 buffer has a legal extent and still wrote four elements past the
    end. `T.clear` shares the same node, so both spellings are covered.
    """
    NBUF = 8

    def program():
        @T.prim_func
        def main(out: T.Tensor((NBUF,), "int32")):
            with T.Kernel(1, threads=32):
                As = T.alloc_shared((NBUF,), "int32")
                if op == "fill":
                    T.fill(As[lo:hi], 7)
                else:
                    T.clear(As[lo:hi])
                T.copy(As, out)

        return main

    # Lowering is where the Fill node validates the region, and it needs no device,
    # so these cases run with or without a GPU.
    target = {"kind": "cuda"}
    if fits:
        with tvm.target.Target(target):
            tilelang.lower(program(), target=target)
    else:
        with pytest.raises(Exception, match="exceeds"), tvm.target.Target(target):
            tilelang.lower(program(), target=target)


if __name__ == "__main__":
    test_matmul()
