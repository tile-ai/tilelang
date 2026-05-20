import pytest
from tilelang.utils.sparse import compress, randn_semi_sparse, randint_semi_sparse, get_e_factor
from tilelang.utils.tensor import torch_assert_close

import tilelang.testing
import torch
import tilelang.language as T


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


def run_gemm_ss(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages,
    num_threads,
    meta_dtype,
):
    metadata_dtype = meta_dtype
    program = matmul(
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
        dtypeAccum,
        metadata_dtype,
        get_e_factor(in_dtype, metadata_dtype),
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    A, B = generate_dense_input(M, N, K, trans_A, trans_B, in_dtype)

    A_sparse, E = compress(A.t().contiguous() if trans_A else A, meta_dtype=meta_dtype.as_torch())
    if trans_A:
        A_sparse = A_sparse.t().contiguous()
        E = E.t().contiguous()
    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    torch_assert_close(
        C_sp.to(out_dtype.as_torch()).to(torch.float32),
        C.to(out_dtype.as_torch()).to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
        base_name="tilelang_sp",
        ref_name="ref_dense",
    )


def generate_dense_input(M, N, K, trans_A, trans_B, in_dtype, seed=0):
    torch.manual_seed(seed)
    is_8bit = "8" in in_dtype
    is_unsigned = "uint" in in_dtype
    is_int = "int" in in_dtype
    if is_int:
        if is_8bit:
            low, high = (0, 4) if is_unsigned else (-2, 2)
        else:
            low, high = (0, 128) if is_unsigned else (-64, 64)
        A = randint_semi_sparse(M, K, low=low, high=high, dtype=in_dtype.as_torch(), device="cuda", transposed=trans_A)
        B = torch.randint(size=(N, K) if trans_B else (K, N), low=low, high=high, dtype=in_dtype.as_torch(), device="cuda")
    else:
        A = randn_semi_sparse(M, K, dtype=in_dtype.as_torch(), device="cuda", transposed=trans_A)
        B = torch.randn((N, K) if trans_B else (K, N), device="cuda", dtype=torch.float32).to(in_dtype.as_torch())

    return A, B


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype",
    [
        (128, 128, 32, False, True, T.float16, T.float16, T.float, 128, 128, 32, 2, 128, T.int16),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int32),
        (128, 128, 32, False, False, T.float16, T.float16, T.float, 128, 128, 32, 2, 128, T.int16),
        (64, 128, 32, True, False, T.float16, T.float16, T.float, 64, 128, 32, 2, 128, T.int16),
        (64, 128, 32, True, True, T.float16, T.float16, T.float, 64, 128, 32, 2, 128, T.int16),
        (128, 8, 64, False, True, T.float16, T.float16, T.float, 128, 8, 32, 0, 128, T.int16),
        (128, 128, 32, False, True, T.bfloat16, T.bfloat16, T.float32, 128, 128, 32, 2, 128, T.int16),
        (64, 128, 128, True, True, T.int8, T.int8, T.int32, 64, 128, 128, 2, 128, T.int32),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int16),
        (128, 128, 64, False, True, T.float8_e5m2, T.float8_e5m2, T.float32, 128, 128, 64, 2, 128, T.int32),
    ],
)
def test_gemm_ss(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype
):
    run_gemm_ss(
        M,
        N,
        K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        block_M,
        block_N,
        block_K,
        num_stages,
        num_threads,
        meta_dtype=meta_dtype,
    )


def matmul_rs(
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
    A_frag_shape = A_shared_shape
    E_shape = (M, K // E_factor) if not trans_A else (K // E_factor, M)
    E_shared_shape = (block_M, block_K // E_factor) if not trans_A else (block_K // E_factor, block_M)

    import tilelang.language as T

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
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                }
            )
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
                T.copy(A_shared, A_frag)
                T.gemm_sp(A_frag, E_shared, B_shared, C_frag, trans_A, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages,
    num_threads,
    meta_dtype,
):
    metadata_dtype = meta_dtype
    program = matmul_rs(
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
        dtypeAccum,
        metadata_dtype,
        get_e_factor(in_dtype, metadata_dtype),
        num_stages,
        num_threads,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    A, B = generate_dense_input(M, N, K, trans_A, trans_B, in_dtype)
    A_sparse, E = compress(A.t().contiguous() if trans_A else A, meta_dtype=meta_dtype.as_torch())
    if trans_A:
        A_sparse = A_sparse.t().contiguous()
        E = E.t().contiguous()
    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    torch_assert_close(
        C_sp.to(out_dtype.as_torch()).to(torch.float32),
        C.to(out_dtype.as_torch()).to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
        base_name="tilelang_sp",
        ref_name="ref_dense",
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype",
    [
        (128, 256, 32, False, True, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int32),
        (128, 256, 32, False, False, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, False, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, True, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (128, 8, 64, False, True, T.float16, T.float16, T.float32, 128, 8, 32, 0, 128, T.int16),
        (128, 256, 32, False, True, T.bfloat16, T.bfloat16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (64, 128, 128, True, True, T.int8, T.int8, T.int32, 64, 128, 128, 2, 128, T.int32),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int16),
        (128, 128, 64, False, True, T.float8_e5m2, T.float8_e5m2, T.float32, 128, 128, 64, 2, 128, T.int32),
    ],
)
def test_gemm_rs(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype
):
    run_gemm_rs(
        M,
        N,
        K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        block_M,
        block_N,
        block_K,
        num_stages,
        num_threads,
        meta_dtype=meta_dtype,
    )


def matmul_sr(
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
    B_frag_shape = B_shared_shape
    E_shape = (M, K // E_factor) if not trans_A else (K // E_factor, M)
    E_shared_shape = (block_M, block_K // E_factor) if not trans_A else (block_K // E_factor, block_M)

    import tilelang.language as T

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
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                }
            )
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
                T.copy(B_shared, B_frag)
                T.gemm_sp(A_shared, E_shared, B_frag, C_frag, trans_A, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages,
    num_threads,
    meta_dtype,
):
    metadata_dtype = meta_dtype
    program = matmul_sr(
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
        dtypeAccum,
        metadata_dtype,
        get_e_factor(in_dtype, metadata_dtype),
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    A, B = generate_dense_input(M, N, K, trans_A, trans_B, in_dtype)
    A_sparse, E = compress(A.t().contiguous() if trans_A else A, meta_dtype=meta_dtype.as_torch())
    if trans_A:
        A_sparse = A_sparse.t().contiguous()
        E = E.t().contiguous()
    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    torch_assert_close(
        C_sp.to(out_dtype.as_torch()).to(torch.float32),
        C.to(out_dtype.as_torch()).to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
        base_name="tilelang_sp",
        ref_name="ref_dense",
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype",
    [
        (128, 256, 32, False, True, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (128, 128, 128, False, True, T.int8, T.int8, T.int32, 128, 128, 128, 2, 128, T.int32),
        (128, 256, 32, False, False, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, False, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, True, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (128, 8, 64, False, True, T.float16, T.float16, T.float32, 128, 8, 32, 0, 128, T.int16),
        (128, 256, 32, False, True, T.bfloat16, T.bfloat16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (64, 128, 128, True, True, T.int8, T.int8, T.int32, 64, 128, 128, 2, 128, T.int32),
        (128, 128, 128, False, True, T.int8, T.int8, T.int32, 128, 128, 128, 2, 128, T.int16),
        (128, 128, 64, False, True, T.float8_e5m2, T.float8_e5m2, T.float32, 128, 128, 64, 2, 128, T.int32),
    ],
)
def test_gemm_sr(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype
):
    run_gemm_sr(
        M,
        N,
        K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        block_M,
        block_N,
        block_K,
        num_stages,
        num_threads,
        meta_dtype=meta_dtype,
    )


def matmul_rr(
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
    A_frag_shape = A_shared_shape
    B_frag_shape = B_shared_shape
    E_shape = (M, K // E_factor) if not trans_A else (K // E_factor, M)
    E_shared_shape = (block_M, block_K // E_factor) if not trans_A else (block_K // E_factor, block_M)

    import tilelang.language as T

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
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout(
                {
                    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                }
            )
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
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm_sp(A_frag, E_shared, B_frag, C_frag, trans_A, trans_A, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
    meta_dtype=T.int16,
):
    metadata_dtype = meta_dtype
    program = matmul_rr(
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
        dtypeAccum,
        metadata_dtype,
        get_e_factor(in_dtype, metadata_dtype),
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    A, B = generate_dense_input(M, N, K, trans_A, trans_B, in_dtype)
    A_sparse, E = compress(A.t().contiguous() if trans_A else A, meta_dtype=meta_dtype.as_torch())
    if trans_A:
        A_sparse = A_sparse.t().contiguous()
        E = E.t().contiguous()
    C_sp = kernel(A_sparse, E, B)

    def _matmul(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return torch.matmul(A, B)

    C = _matmul(A, B)

    torch_assert_close(
        C_sp.to(out_dtype.as_torch()).to(torch.float32),
        C.to(out_dtype.as_torch()).to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
        base_name="tilelang_sp",
        ref_name="ref_dense",
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype",
    [
        (128, 256, 32, False, True, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int32),
        (128, 256, 32, False, False, T.float16, T.float16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, False, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (64, 256, 32, True, True, T.float16, T.float16, T.float32, 64, 256, 32, 2, 128, T.int16),
        (128, 256, 32, False, True, T.bfloat16, T.bfloat16, T.float32, 128, 256, 32, 2, 128, T.int16),
        (128, 8, 32, False, True, T.float16, T.float16, T.float32, 128, 8, 32, 2, 128, T.int16),
        (128, 8, 64, False, True, T.int8, T.int8, T.int32, 128, 8, 64, 2, 128, T.int32),
        (64, 128, 128, True, True, T.int8, T.int8, T.int32, 64, 128, 128, 2, 128, T.int32),
        (128, 128, 64, False, True, T.int8, T.int8, T.int32, 128, 128, 64, 2, 128, T.int16),
        (128, 128, 64, False, True, T.float8_e5m2, T.float8_e5m2, T.float32, 128, 128, 64, 2, 128, T.int32),
    ],
)
def test_gemm_rr(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages, num_threads, meta_dtype
):
    run_gemm_rr(
        M,
        N,
        K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        block_M,
        block_N,
        block_K,
        num_stages,
        num_threads,
        meta_dtype=meta_dtype,
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "in_dtype, out_dtype, dtypeAccum, meta_dtype",
    [
        (T.float16, T.float16, T.float32, T.int16),
        (T.int8, T.int8, T.int32, T.int32),
        (
            T.float16,
            T.float16,
            T.float32,
            T.int8,
        ),
        (
            T.bfloat16,
            T.bfloat16,
            T.float32,
            T.int8,
        ),
        (
            T.bfloat16,
            T.bfloat16,
            T.float32,
            T.int16,
        ),
        (
            T.int8,
            T.int8,
            T.int32,
            T.int8,
        ),
        (
            T.int8,
            T.int8,
            T.int32,
            T.int16,
        ),
        (
            T.float8_e5m2,
            T.float8_e5m2,
            T.float32,
            T.int8,
        ),
        (
            T.float8_e5m2,
            T.float8_e5m2,
            T.float32,
            T.int16,
        ),
        (
            T.float8_e5m2,
            T.float8_e5m2,
            T.float32,
            T.int32,
        ),
    ],
)
def test_compress_dtype_combinations(in_dtype, out_dtype, dtypeAccum, meta_dtype):
    run_gemm_ss(128, 128, 128, False, True, in_dtype, out_dtype, dtypeAccum, 128, 128, 64, 2, 128, meta_dtype=meta_dtype)


if __name__ == "__main__":
    tilelang.testing.main()
