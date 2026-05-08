"""Frontend int4 GEMM example for the T.gemm int4 path.

This file intentionally models the desired TileLang frontend API:
- A/B are declared as T.int4 tensors
- the matmul is expressed with T.gemm(...)

The example compiles the kernel, prints the generated CUDA source, and
checks correctness against a PyTorch reference.
"""

import torch

import tilelang
import tilelang.language as T


def matmul_nt_int4(M, N, K, block_M, block_N, block_K, threads=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int4),
        B: T.Tensor((N, K), T.int4),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.int4)
            B_shared = T.alloc_shared((block_N, block_K), T.int4)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                # Frontend expectation: T.gemm should accept int4 operands directly.
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def compile_int4_gemm(
    M=1024,
    N=1024,
    K=1024,
    block_M=128,
    block_N=128,
    block_K=64,
    threads=128,
    print_cuda_source=True,
):
    func = matmul_nt_int4(M, N, K, block_M, block_N, block_K, threads)
    kernel = tilelang.compile(func, out_idx=-1)
    print("Compilation succeeded.")
    if print_cuda_source:
        print(kernel.get_kernel_source())
    return func, kernel


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError(f"Expected torch.int8 logical int4 tensor, but got {tensor.dtype}.")
    if tensor.ndim == 0 or tensor.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of a logical int4 tensor must be even for int8 packing.")

    tensor_i16 = tensor.to(torch.int16)
    packed = (tensor_i16[..., ::2] & 0x0F) | ((tensor_i16[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.int8).contiguous()


def check_int4_gemm_correctness(
    M=1024,
    N=1024,
    K=1024,
    block_M=128,
    block_N=128,
    block_K=64,
    threads=128,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the int4 GEMM example.")

    _, kernel = compile_int4_gemm(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        threads=threads,
    )

    A_logical = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    B_logical = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)

    A_packed = pack_int4(A_logical)
    B_packed = pack_int4(B_logical)
    C = kernel(A_packed, B_packed)
    torch.cuda.synchronize()

    ref_c = torch.matmul(A_logical.cpu().to(torch.int32), B_logical.cpu().to(torch.int32).T)
    torch.testing.assert_close(C.cpu(), ref_c, rtol=0, atol=0)
    print("Correctness check passed.")
    return C, ref_c


def main():
    # check_int4_gemm_correctness(M=16, N=16, K=32, block_M=16, block_N=16, block_K=32)
    # check_int4_gemm_correctness(M=16, N=16, K=64, block_M=16, block_N=16, block_K=64)
    check_int4_gemm_correctness()


if __name__ == "__main__":
    main()
