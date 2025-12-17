import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T

from tilelang.utils.sparse import compress, randn_semi_sparse, randint_semi_sparse
from tilelang.layout import make_cutlass_metadata_layout
from tilelang.utils.tensor import torch_assert_close, map_torch_type
from tilelang.intrinsics.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter

torch.backends.cuda.matmul.allow_tf32 = False
# torch.manual_seed(42)  # only enable when debugging


def generate_dense_input(M, N, K, trans_A, trans_B, in_dtype):
    is_8bit = "8" in in_dtype
    is_unsigned = "uint" in in_dtype
    is_int = "int" in in_dtype
    if is_int:
        if is_8bit:
            low, high = (0, 4) if is_unsigned else (-2, 2)
        else:
            low, high = (0, 128) if is_unsigned else (-64, 64)
        A = randint_semi_sparse(M, K, low=low, high=high, dtype=map_torch_type(in_dtype), device="cuda", transposed=trans_A)
        B = torch.randint(size=(N, K) if trans_B else (K, N), low=low, high=high, dtype=map_torch_type(in_dtype), device="cuda")
    else:
        A = randn_semi_sparse(M, K, dtype=torch.float32, device="cuda", transposed=trans_A).to(map_torch_type(in_dtype))
        B = torch.randn((N, K) if trans_B else (K, N), device="cuda", dtype=torch.float32).to(map_torch_type(in_dtype))
    return A, B


def matmul_sp_sm90(
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
    E_factor = 4 if in_dtype == T.float32 else 8
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (K, N) if not trans_B else (N, K)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    @T.prim_func
    def main(
        A_sparse: T.Tensor(A_sparse_shape, in_dtype),
        E: T.Tensor((M, K // E_factor), "uint8"),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // E_factor), "uint8")
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=in_dtype, arch="9.0", block_k=block_K),
                    E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=in_dtype, arch="9.0", block_k=block_K),
                }
            )
            T.disable_warp_group_reg_alloc()
            T.clear(C_frag)
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
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def matmul_sp_sm80(
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
    is_8_bit = "8" in in_dtype
    metadata_dtype = T.int32 if is_8_bit else T.int16
    E_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[in_dtype][metadata_dtype]
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (K, N) if not trans_B else (N, K)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    @T.prim_func
    def main(
        A_sparse: T.Tensor(A_sparse_shape, in_dtype),
        E: T.Tensor((M, K // E_factor), metadata_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // E_factor), metadata_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=in_dtype, arch="8.0"),
                    E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=in_dtype, arch="8.0"),
                }
            )
            T.clear(C_frag)
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
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


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
    kernel,
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    block_K,
    trans_A,
    trans_B,
):
    kernel = tilelang.compile(
        kernel,
        out_idx=[-1],
    )
    A, B = generate_dense_input(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
    )
    A_sparse, E = compress(A, transposed=trans_A, block_k=block_K)

    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype or "int8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    if "float8" in in_dtype:
        diff = calc_diff(C_sp, C)
        assert diff < 1e-3, f"{diff=}"
    else:
        torch_assert_close(
            C_sp.to(torch.float32),
            C.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            base_name="tilelang_sp",
            ref_name="ref_dense",
        )
    print("pass")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def run_gemm_sp_sm90(
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
    trans_A,
    trans_B,
):
    kernel = matmul_sp_sm90(
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
    run_gemm_sp(
        kernel,
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        block_K,
        trans_A,
        trans_B,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def run_gemm_sp_sm80(
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
    trans_A,
    trans_B,
):
    kernel = matmul_sp_sm80(
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
    run_gemm_sp(
        kernel,
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        block_K,
        trans_A,
        trans_B,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B",
    [
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 32, 2, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 32, 0, 256, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 2, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 128, 128, 128, 0, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 128, 128, 128, 2, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 128, 256, 0, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 128, 256, 2, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 128, False, True),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 128, False, False),
        (512, 1024, 768, T.float8_e4m3fn, T.float16, T.float16, 64, 64, 64, 2, 128, False, True),
        (512, 1024, 768, T.int8, T.int32, T.int32, 64, 64, 64, 2, 128, False, True),
    ],
)
def test_gemm_sp_sm90(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B):
    run_gemm_sp_sm90(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B",
    [
        (512, 1024, 768, T.float16, T.float32, T.float32, 32, 32, 32, 0, 32, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 32, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 32, 32, 64, 0, 32, False, True),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 32, False, True),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 0, 128, False, True),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 1, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 2, 128, False, False),
        (512, 1024, 768, T.float16, T.float32, T.float32, 64, 64, 64, 3, 128, False, False),
        (512, 1024, 768, T.int8, T.int32, T.int32, 32, 32, 64, 0, 32, False, True),
        (512, 1024, 768, T.int8, T.int32, T.int32, 64, 64, 64, 0, 32, False, True),
        (512, 1024, 768, T.int8, T.int32, T.int32, 128, 128, 128, 0, 128, False, True),
        (512, 1024, 768, T.int8, T.int32, T.int32, 64, 64, 64, 1, 128, False, True),
        (512, 1024, 768, T.int8, T.int32, T.int32, 64, 64, 64, 2, 128, False, True),
    ],
)
def test_gemm_sp_sm80(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B):
    run_gemm_sp_sm80(M, N, K, in_dtype, out_dtype, accum_dtype, block_M, block_N, block_K, num_stages, num_threads, trans_A, trans_B)


if __name__ == "__main__":
    tilelang.testing.main()
