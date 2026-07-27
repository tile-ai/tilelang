import tilelang
import tilelang.language as T
import torch


def make_matmul_prim(M, N, K, block_M=128, block_N=128, block_K=32, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def build_matmul_kernel(M=1024, N=1024, K=1024, target="cuda"):
    """Compile and return a callable kernel that takes (A, B) and returns C."""
    if target.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build CUDA kernel for host-check repros.")
    prim = make_matmul_prim(M, N, K)
    # out_idx=[2] means the 3rd param C is treated as output; wrapper takes (A,B)
    return tilelang.compile(prim, out_idx=[2], target=target)


def build_scalar_check_kernel(target="cuda"):
    @T.prim_func
    def scalar_check(x: T.int32, flag: T.bool()):
        T.evaluate(0)

    return tilelang.compile(scalar_check, target=target)
