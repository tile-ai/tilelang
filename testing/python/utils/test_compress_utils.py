import pytest
import torch
import tilelang
from tilelang.utils.sparse import get_e_factor
import tilelang.testing
import tilelang.language as T

from tilelang.utils.sparse import compress, randn_semi_sparse, randint_semi_sparse, torch_compress
from tilelang.utils.tensor import torch_assert_close


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    metadata_dtype,
    E_factor,
    num_stages,
    threads,
):
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    E_shape = (M, K // E_factor) if not trans_A else (K // E_factor, M)
    E_shared_shape = (block_M, block_K // E_factor) if not trans_A else (block_K // E_factor, block_M)

    @T.prim_func
    def main(
        A_sparse: T.Tensor(A_sparse_shape, in_dtype),
        E: T.Tensor(E_shape, metadata_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared(E_shared_shape, metadata_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(E[k * block_K // E_factor, by * block_M], E_shared)
                    T.copy(A_sparse[k * block_K // 2, by * block_M], A_shared)
                else:
                    T.copy(E[by * block_M, k * block_K // E_factor], E_shared)
                    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, trans_A, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def generate_dense_input(N, trans_A, trans_B, in_dtype, seed=0):
    torch.manual_seed(seed)
    is_8bit = "8" in str(in_dtype)
    is_unsigned = "uint" in str(in_dtype)
    is_int = "int" in str(in_dtype)
    if is_int:
        if is_8bit:
            low, high = (0, 128) if is_unsigned else (-64, 64)
        else:
            low, high = (0, 258) if is_unsigned else (-128, 128)
        A = randint_semi_sparse(N, N, low=low, high=high, dtype=in_dtype, device="cuda", transposed=trans_A)
        B = torch.randint(size=(N, N) if trans_B else (N, N), low=low, high=high, dtype=in_dtype, device="cuda")
    else:
        A = randn_semi_sparse(N, N, dtype=in_dtype, device="cuda", transposed=trans_A)
        B = torch.randn((N, N) if trans_B else (N, N), device="cuda", dtype=torch.float32).to(in_dtype)
    return A, B


def _test_compress(dtype, meta_dtype):
    A, B = generate_dense_input(64, in_dtype=dtype.as_torch(), trans_A=False, trans_B=False)
    sp_tl, meta_tl = compress(A, meta_dtype=meta_dtype.as_torch())
    sp_ref, meta_ref = torch_compress(A, meta_dtype=meta_dtype.as_torch())
    # NOTE: in case that there are multiple zeros, the case might fail occasionally
    # if we directly compare the compressed sparse values
    program = matmul(
        64,
        64,
        64,
        64,
        64,
        64,
        False,
        False,
        dtype,
        dtype,
        T.int32 if dtype == T.int8 else T.float32,
        meta_dtype,
        get_e_factor(dtype, meta_dtype),
        0,
        128,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
    )

    C_tl = kernel(sp_tl, meta_tl, B)
    C_ref = kernel(sp_ref, meta_ref, B)
    torch_assert_close(C_tl, C_ref, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype, meta_dtype",
    [
        (T.int8, T.int8),
        (T.int8, T.int16),
        (T.int8, T.int32),
        (T.float16, T.int8),
        (T.float16, T.int16),
        (T.float32, T.int8),
        (T.float32, T.int16),
    ],
)
def test_compress(dtype, meta_dtype):
    _test_compress(dtype, meta_dtype)


if __name__ == "__main__":
    tilelang.testing.main()
