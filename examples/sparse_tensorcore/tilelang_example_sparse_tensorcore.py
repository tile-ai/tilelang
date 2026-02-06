import torch
import tilelang
from tilelang.utils.sparse import compress_sm90
from tilelang.layout import make_cutlass_metadata_layout
from tilelang import language as T
import tilelang.testing


@tilelang.jit(out_idx=[-1])
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
):
    A_sparse_shape = (M, K // 2)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K // 2)
    B_shared_shape = (block_K, block_N)

    @T.prim_func
    def main(
        A_sparse: T.Tensor(A_sparse_shape, in_dtype),
        E: T.Tensor((M, K // 8), "uint8"),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // 8), "uint8")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, arch="9.0", block_k=block_K),
                    E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16, arch="9.0", block_k=block_K),
                }
            )
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // 8], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def generate_2_to_4_sparse_tensor(shape, dtype=torch.float32, device="cpu"):
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    full_tensor = torch.randn(shape, dtype=dtype, device=device)
    group_count = shape[-1] // 4
    group_shape = shape[:-1] + (group_count, 4)

    rand_vals = torch.rand(group_shape, device=device)
    topk_indices = rand_vals.topk(k=2, dim=-1).indices
    mask = torch.zeros(group_shape, dtype=torch.bool, device=device)
    mask.scatter_(-1, topk_indices, True)
    mask = mask.view(shape)

    sparse_tensor = full_tensor * mask
    return sparse_tensor


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
):
    kernel = matmul_sp(
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
    )

    A = generate_2_to_4_sparse_tensor((M, K), dtype=torch.float16, device="cuda")
    A_sparse, E = compress_sm90(A, block_k=block_K, transposed=False)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    C_sp = kernel(A_sparse, E, B).half()
    C = torch.matmul(A, B)
    torch.testing.assert_close(C_sp, C, atol=1e-3, rtol=1e-3)
    print("pass")


def main():
    run_gemm_sp(512, 1024, 768, T.float16, T.float16, T.float32, 128, 128, 128, 2, 128)


def run_regression_perf():
    M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, num_threads = (
        512,
        1024,
        768,
        128,
        128,
        128,
        "float16",
        "float16",
        "float32",
        2,
        128,
    )
    kernel = matmul_sp(
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
    )
    A = generate_2_to_4_sparse_tensor((M, K), dtype=torch.float16, device="cuda")
    A_sparse, E = compress_sm90(A, block_k=block_K, transposed=False)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(A_sparse, E, B)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    main()
