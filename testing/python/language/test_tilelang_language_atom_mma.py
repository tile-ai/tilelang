import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.intrinsics import (
    TensorCoreIntrinEmitter,
    WGMMATensorCoreIntrinEmitter,
    TCGEN05TensorCoreIntrinEmitter,
)
from tilelang.cuda.intrinsics.layout.mma_layout import get_swizzle_layout
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    if shape[-1] * T.dtype(dtype).bits == 512:

        def transform_func(i, j):
            return get_swizzle_layout(i, j, shape[-1], dtype)

        return T.Layout(shape, transform_func)
    return T.Layout(shape, lambda *args: args)


def infer_wgmma_shared_layout(continuity, dtype):
    vectorized_size = 128 // T.dtype(dtype).bits
    if continuity % (vectorized_size * 8) == 0:
        return make_full_bank_swizzled_layout
    if continuity % (vectorized_size * 4) == 0:
        return make_half_bank_swizzled_layout
    if continuity % (vectorized_size * 2) == 0:
        return make_quarter_bank_swizzled_layout
    return make_linear_layout


# ---------------------------------------------------------------------------
# SM80+ MMA (atom-level)  --  correctness test
# ---------------------------------------------------------------------------


def make_mma_atom_kernel(M, N, K, in_dtype, out_dtype, accum_dtype):
    micro_size_x = micro_size_y = micro_size_k = 16
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32 if in_dtype == T.float16 else 64

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk
    threads = 32 * block_row_warps * block_col_warps

    emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )
    warp_rows = emitter.warp_rows
    warp_cols = emitter.warp_cols
    local_size_a = emitter.local_size_a
    local_size_b = emitter.local_size_b
    local_size_c = emitter.local_size_out
    num_inst_m = emitter.mma_num_inst_m
    num_inst_n = emitter.mma_num_inst_n

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_shared = T.alloc_shared((block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y), out_dtype)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                }
            )

            T.clear(C_local)

            for ko in T.serial(K // block_K):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(block_K // micro_size_k):
                    emitter.ldmatrix_a(A_local, A_shared, ki)
                    emitter.ldmatrix_b(B_local, B_shared, ki)
                    for i, j in T.grid(num_inst_m, num_inst_n):
                        emitter.mma_atom(A_local, B_local, C_local, i, j, ki)

            emitter.stmatrix(C_local, C_shared)
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y]

    return main


def _run_mma_atom(M, N, K, in_dtype, out_dtype, accum_dtype):
    kernel = tilelang.compile(make_mma_atom_kernel(M, N, K, in_dtype, out_dtype, accum_dtype), target="cuda", out_idx=[2])
    a = torch.randn(M, K, device="cuda", dtype=in_dtype.as_torch())
    b = torch.randn(N, K, device="cuda", dtype=in_dtype.as_torch())
    c = kernel(a, b)
    ref = (a.float() @ b.T.float()).to(out_dtype.as_torch())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=0.1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_mma_atom_gemm():
    _run_mma_atom(128, 128, 128, T.float16, T.float16, T.float16)
    _run_mma_atom(256, 256, 256, T.bfloat16, T.float32, T.float32)


# ---------------------------------------------------------------------------
# SM90 WGMMA (atom-level, SS variant)  --  codegen and correctness test
# ---------------------------------------------------------------------------
def make_wgmma_atom_kernel(M, N, K, in_dtype, out_dtype, accum_dtype):
    chunk = 32 if in_dtype == T.float16 else 64
    block_row_warps = 4
    block_col_warps = 1
    warp_row_tiles = M // block_row_warps
    warp_col_tiles = N // block_col_warps
    block_K = chunk

    emi = WGMMATensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )
    a_layout = infer_wgmma_shared_layout(K, in_dtype)
    b_layout = infer_wgmma_shared_layout(emi.wgmma_inst_n, in_dtype)
    num_inst_m = emi.wgmma_num_inst_m
    num_inst_n = emi.wgmma_num_inst_n
    num_k_atoms = emi.wgmma_num_k_atoms

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, N), in_dtype)
            C_local = T.alloc_fragment((M, N), accum_dtype)

            emi._assign_a_shared_layout(a_layout(A_shared))
            emi._assign_b_shared_layout(b_layout(B_shared))
            T.annotate_layout(
                {
                    A_shared: a_layout(A_shared),
                    B_shared: b_layout(B_shared),
                    C_local: emi.make_mma_store_layout(C_local),
                }
            )

            T.copy(A[0:M, 0:block_K], A_shared)
            T.copy(B[0:block_K, 0:N], B_shared)

            a_params = emi.compute_wgmma_a_desc_params(A_shared)
            b_params = emi.compute_wgmma_b_desc_params(B_shared)

            desc_a = T.alloc_wgmma_desc()
            desc_b = T.alloc_wgmma_desc()
            emi.init_wgmma_a_desc(desc_a, A_shared, a_params)
            emi.init_wgmma_b_desc(desc_b, B_shared, b_params)
            emi.wgmma_fence_c(C_local)
            emi.wgmma_arrive()

            for n in T.unroll(num_inst_n):
                for m in T.unroll(num_inst_m):
                    for ki in T.unroll(num_k_atoms):
                        emi.wgmma_ss_atom(desc_a, desc_b, C_local, m, n, ki, a_params, b_params, T.bool(True))

            emi.wgmma_commit()
            emi.wgmma_wait(0)
            emi.wgmma_fence_c(C_local)

            T.copy(C_local, C[0:M, 0:N])

    return main


def _run_wgmma_atom(M, N, K, in_dtype, out_dtype, accum_dtype):
    kernel = tilelang.compile(
        make_wgmma_atom_kernel(M, N, K, in_dtype, out_dtype, accum_dtype),
        target="cuda",
        out_idx=[2],
    )
    src = kernel.get_kernel_source()
    assert "wgmma_ss" in src
    assert "initialize_wgmma_descriptor" in src
    assert "warpgroup_arrive" in src
    assert "warpgroup_commit_batch" in src

    a = torch.randn(M, K, device="cuda", dtype=in_dtype.as_torch())
    b = torch.randn(K, N, device="cuda", dtype=in_dtype.as_torch())
    c = kernel(a, b)
    ref = (a.float() @ b.float()).to(out_dtype.as_torch())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=0.1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_atom_gemm():
    _run_wgmma_atom(64, 64, 32, T.float16, T.float16, T.float32)


# ---------------------------------------------------------------------------
# SM100 TCGEN05MMA (atom-level, SS variant)  --  codegen and correctness test
# ---------------------------------------------------------------------------


def make_tcgen05_atom_kernel(M, N, K, in_dtype, out_dtype, accum_dtype):
    chunk = K
    emi = TCGEN05TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=M,
        warp_col_tiles=N,
        chunk=chunk,
    )
    emi.get_tcgen5_mma_meta(M, N, K, True)
    a_layout = infer_wgmma_shared_layout(K, in_dtype)
    b_layout = infer_wgmma_shared_layout(K, in_dtype)
    num_inst_m = emi.tcgen05_num_inst_m
    num_inst_n = emi.tcgen05_num_inst_n
    num_k_atoms = emi.tcgen05_num_k_atoms
    instr_desc = emi.compute_tcgen05_instr_desc()

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        D: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), in_dtype)
            B_shared = T.alloc_shared((N, K), in_dtype)
            C_tmem = T.alloc_tmem((M, N), accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((M, N), accum_dtype)
            C_shared = T.alloc_shared((M, N), out_dtype)

            emi._assign_a_shared_layout(a_layout(A_shared))
            emi._assign_b_shared_layout(b_layout(B_shared))
            T.annotate_layout(
                {
                    A_shared: a_layout(A_shared),
                    B_shared: b_layout(B_shared),
                    C_tmem: emi.make_mma_store_layout(C_tmem),
                }
            )

            for i, k in T.Parallel(M, K):
                A_shared[i, k] = A[i, k]
            for j, k in T.Parallel(N, K):
                B_shared[j, k] = B[j, k]

            a_params = emi.compute_tcgen05_a_desc_params(A_shared)
            b_params = emi.compute_tcgen05_b_desc_params(B_shared)

            if T.get_thread_binding() // 32 == 0:
                desc_a = T.alloc_tcgen05_smem_desc()
                desc_b = T.alloc_tcgen05_smem_desc()
                emi.init_tcgen05_a_desc(desc_a, A_shared, a_params)
                emi.init_tcgen05_b_desc(desc_b, B_shared, b_params)

                for n in T.unroll(num_inst_n):
                    for m in T.unroll(num_inst_m):
                        for ki in T.unroll(0, num_k_atoms):
                            emi.tcgen05_ss_atom(desc_a, desc_b, C_tmem, m, n, ki, a_params, b_params, instr_desc, T.bool(True))
                emi.tcgen05_atom_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:M, 0:N])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_tcgen05_atom_gemm():
    M, N, K = 128, 128, 128
    in_dtype = T.bfloat16
    out_dtype = T.bfloat16
    kernel = tilelang.compile(
        make_tcgen05_atom_kernel(M, N, K, in_dtype, out_dtype, T.float32),
        target="cuda",
        out_idx=[2],
    )
    src = kernel.get_kernel_source()
    assert "tcgen05mma_ss" in src
    assert "threadIdx.x) >> 5) == 0" in src  # elect 1 thread to issue UMMA

    a = torch.randn(M, K, device="cuda", dtype=in_dtype.as_torch())
    b = torch.randn(N, K, device="cuda", dtype=in_dtype.as_torch())
    d = kernel(a, b)
    ref = (a.float() @ b.T.float()).to(out_dtype.as_torch())
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
