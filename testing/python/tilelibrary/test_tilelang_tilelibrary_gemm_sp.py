# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
import torch
import tilelang
import tilelang.testing

from tilelang.utils.sparse import compress_sm90
from tilelang.layout import make_metadata_layout

torch.set_printoptions(threshold=float('inf'), edgeitems=float('inf'), linewidth=10000)
torch.manual_seed(42)

STR_TO_TYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "e4m3_float8": torch.float8_e4m3fn,
    "int8": torch.int8,
}

SPARSITY_MAP = {
    torch.float16: (2, 4),
    torch.bfloat16: (2, 4),
    torch.float8_e4m3fn: (2, 4),
    torch.int8: (2, 4),
}


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
    E_factor = 4 if in_dtype == "float32" else 8
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (K, N) if not trans_B else (N, K)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    import tilelang.language as T

    @T.prim_func
    def main(
            A_sparse: T.Tensor(A_sparse_shape, in_dtype),
            E: T.Tensor((M, K // E_factor), 'uint8'),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // E_factor), 'uint8')
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
            T.no_set_max_nreg()
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // E_factor], E_shared)
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


def generate_sparse_tensor_float32(M: int, K: int, dtype: torch.dtype, device='cpu', trans_A=False):
    elem, group = SPARSITY_MAP[dtype]
    if K % group != 0:
        raise ValueError(
            f"Last dimension must be divisible by {group} for {elem}:{group} sparsity.")

    if trans_A:
        full_tensor = torch.randn(K * M, dtype=torch.float32, device=device).view(K, M)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for j in range(M):
            for i in range(0, K, group):
                flat_idx = torch.randint(0, group, (elem,), dtype=torch.int64)
                for k in range(1, len(flat_idx)):
                    while flat_idx[k] in flat_idx[:k]:
                        flat_idx[k] = torch.randint(0, group, (1,), dtype=torch.int64)
                for idx in flat_idx:
                    mask[i + idx, j] = True
    else:
        full_tensor = torch.randn((M, K), dtype=torch.float32, device=device).view(M, K)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for i in range(M):
            for j in range(0, K, group):
                flat_idx = torch.randint(0, group, (elem,), dtype=torch.int64)
                for k in range(1, len(flat_idx)):
                    while flat_idx[k] in flat_idx[:k]:
                        flat_idx[k] = torch.randint(0, group, (1,), dtype=torch.int64)
                for idx in flat_idx:
                    mask[i, j + idx] = True

    return full_tensor * mask


def normalize(tensor, max_range=100.0):
    assert max_range <= 448.0
    max_v = tensor.abs().max().clamp(1e-4)
    scaler = max_range / max_v
    return tensor * scaler


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


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
    if in_dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = True

    kernel = tilelang.compile(
        program,
        out_idx=[-1],
    )
    A = generate_sparse_tensor_float32(
        M, K, dtype=STR_TO_TYPE[in_dtype], device='cuda', trans_A=trans_A)
    if trans_B:
        B = torch.randn((N, K), device='cuda', dtype=torch.float32)
    else:
        B = torch.randn((K, N), device='cuda', dtype=torch.float32)

    if "float8" in in_dtype or "int8" in in_dtype:
        A = normalize(A)
        B = normalize(B)

    A = A.to(STR_TO_TYPE[in_dtype])
    B = B.to(STR_TO_TYPE[in_dtype])

    A_sparse, E = compress_sm90(A, block_K, trans_A)

    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype or "int8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return torch.matmul(A, B).to(STR_TO_TYPE[out_dtype])

    C = _matmul(A, B)
    if 'float8' in in_dtype:
        diff = calc_diff(C_sp, C)
        assert diff < 1e-3, f"{diff=}"
    else:
        torch.testing.assert_close(C_sp, C, atol=1e-3, rtol=1e-3)
    print("pass")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_gemm_sp():
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 32, 2, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 32, 0, 256)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 128, 128, 128, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 128, 128, 128, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 128, 256, 0, 128)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 128, 256, 2, 128)

    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, False, True)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, True, False)
    run_gemm_sp(512, 1024, 768, "float16", "float16", "float32", 64, 64, 64, 0, 128, True, True)

    run_gemm_sp(512, 1024, 768, "e4m3_float8", "float16", "float16", 64, 64, 64, 2, 128, False,
                True)

    run_gemm_sp(512, 1024, 768, "int8", "int8", "int32", 64, 64, 64, 2, 128, False, True)


if __name__ == "__main__":
    tilelang.testing.main()
