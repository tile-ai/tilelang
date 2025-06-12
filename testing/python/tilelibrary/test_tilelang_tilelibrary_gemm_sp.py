# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
import torch
import tilelang
import tilelang.testing

from tilelang.utils.sparse import compress_sm90
from tilelang.layout import make_metadata_layout

torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=10000)


def matmul_sp(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    trans_A,
    trans_B,
):
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (K, N) if not trans_B else (N, K)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    import tilelang.language as T

    @T.prim_func
    def main(
            A_sparse: T.Tensor(A_sparse_shape, in_dtype),
            E: T.Tensor((M, K // 8), 'uint8'),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // 8), 'uint8')
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout({
                E:
                    make_metadata_layout(
                        E, mma_dtype="float16", arch="sm90", backend="cutlass", block_k=block_K),
                E_shared:
                    make_metadata_layout(
                        E_shared,
                        mma_dtype="float16",
                        arch="sm90",
                        backend="cutlass",
                        block_k=block_K),
            })
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // 8], E_shared)
                if trans_A:
                    T.copy(A_sparse[k * block_K // 2, by * block_M], A_shared)
                else:
                    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def generate_2_to_4_sparse_tensor(M: int, K: int, dtype=torch.float32, device='cpu', trans_A=False):
    if K % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    if trans_A:
        full_tensor = torch.randn(K * M, dtype=dtype, device=device).view(K, M)
        # full_tensor = torch.arange(1, K * M + 1, dtype=dtype, device=device).view(K, M)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for j in range(M):
            for i in range(0, K, 4):
                flat_idx = torch.randint(0, 4, (2,), dtype=torch.int64)
                while flat_idx[0] == flat_idx[1]:
                    flat_idx[1] = torch.randint(0, 4, (1,), dtype=torch.int64)
                # flat_idx = [0, 1]
                mask[i + flat_idx[0], j] = True
                mask[i + flat_idx[1], j] = True
    else:
        full_tensor = torch.randn((M, K), dtype=dtype, device=device).view(M, K)
        # full_tensor = torch.arange(1, K * M + 1, dtype=dtype, device=device).view(M, K)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for i in range(M):
            for j in range(0, K, 4):
                flat_idx = torch.randint(0, 4, (2,), dtype=torch.int64)
                while flat_idx[0] == flat_idx[1]:
                    flat_idx[1] = torch.randint(0, 4, (1,), dtype=torch.int64)
                mask[i, j + flat_idx[0]] = True
                mask[i, j + flat_idx[1]] = True

    return full_tensor * mask


def run_gemm_sp(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    block_M,
    block_N,
    block_K,
    num_stages,
    num_threads,
    trans_A=False,
    trans_B=False,
):
    program = matmul_sp(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        num_threads,
        trans_A,
        trans_B,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
    )
    A = generate_2_to_4_sparse_tensor(M, K, dtype=torch.float16, device='cuda', trans_A=trans_A)
    A_sparse, E = compress_sm90(A, block_K, trans_A)

    if trans_B:
        B = torch.randn((N, K), device='cuda', dtype=torch.float16)
    else:
        B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    C_sp = kernel(A_sparse, E, B).half()

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return torch.matmul(A, B)

    C = _matmul(A, B)
    torch.testing.assert_close(C_sp, C, atol=1e-3, rtol=1e-3)
    print("pass")


def test_gemm_sp():
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 32, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 32, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 128, 128, 128, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 128, 128, 128, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 128, 256, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 128, 256, 2, 128)
    
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, False, True)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, True, False)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, True, True)


if __name__ == "__main__":
    tilelang.testing.main()
    
