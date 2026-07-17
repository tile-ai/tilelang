"""Regression tests for issue #2549: copies from ptr-backed buffers
(``T.make_tensor``) must not be TMA-promoted by warp specialization.

A make_tensor base pointer is a Bind-bound handle var whose value is computed
inside the kernel body (here: indexed from a device-side pointer table), so no
host-side TMA descriptor can be encoded for it. Before the fix, the default
sm_90 pipeline promoted such copies to TMA and aborted in MakePackedAPI with
"variables (A, B) are used, but are not passed in as API arguments".
"""

import math

import tilelang
import tilelang.testing
from tilelang import language as T
import torch


def _compile(func, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="auto",
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
            **kwargs,
        )
    finally:
        tilelang.enable_cache()


def _make_ptr_table(tensors):
    return torch.tensor([t.data_ptr() for t in tensors], device=tensors[0].device, dtype=torch.int64)


def _grouped_gemm_ptr(batch_sizes, K, N, block_M, block_N, block_K, threads=128, dtype=T.float16):
    batch_count = len(batch_sizes)
    max_M = max(batch_sizes)
    tile_offsets = [0]
    for size in batch_sizes[:-1]:
        tile_offsets.append(tile_offsets[-1] + math.ceil(size / block_M))
    total_m_blocks = sum(math.ceil(size / block_M) for size in batch_sizes)

    @T.prim_func
    def kernel(
        A_ptrs: T.Tensor([batch_count], T.ptr),
        B_ptrs: T.Tensor([batch_count], T.ptr),
        C_ptrs: T.Tensor([batch_count], T.ptr),
        batch_tile_offsets: T.Tensor([batch_count], T.int32),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_tile_offset = T.alloc_var(dtype=T.int32)

            cur_batch_idx = 0
            cur_tile_offset = 0
            for i in range(batch_count):
                in_cur = bx >= batch_tile_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur, i, cur_batch_idx)
                cur_tile_offset = T.if_then_else(in_cur, batch_tile_offsets[i], cur_tile_offset)

            m_start = (bx - cur_tile_offset) * block_M
            A = T.make_tensor(A_ptrs[cur_batch_idx], (max_M, K), dtype)
            B = T.make_tensor(B_ptrs[cur_batch_idx], (K, N), dtype)
            C = T.make_tensor(C_ptrs[cur_batch_idx], (max_M, N), dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[m_start, ko * block_K], A_shared)
                T.copy(B[ko * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[m_start, by * block_N])

    return kernel


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_make_tensor_grouped_gemm_compiles_under_default_ws():
    """The ptr-table grouped GEMM must compile and run without disabling WS."""

    batch_sizes, K, N = [16, 33, 64], 128, 96
    block_M = block_N = block_K = 32
    program = _grouped_gemm_ptr(batch_sizes, K, N, block_M, block_N, block_K)

    # Aborted in MakePackedAPI before the fix.
    kernel = _compile(program)

    # Every pipelined copy is make_tensor-based, so nothing is TMA-eligible:
    # the function must take the plain (non-WS) path.
    assert "tl::tma_load" not in kernel.get_kernel_source()

    device = torch.device("cuda")
    dtype = torch.float16
    max_M = max(batch_sizes)
    a_list = [torch.randn(max_M, K, device=device, dtype=dtype) for _ in batch_sizes]
    b_list = [torch.randn(K, N, device=device, dtype=dtype) for _ in batch_sizes]
    c_list = [torch.empty(max_M, N, device=device, dtype=dtype) for _ in batch_sizes]
    tile_offsets = [0]
    for size in batch_sizes[:-1]:
        tile_offsets.append(tile_offsets[-1] + math.ceil(size / block_M))

    kernel(
        _make_ptr_table(a_list),
        _make_ptr_table(b_list),
        _make_ptr_table(c_list),
        torch.tensor(tile_offsets, device=device, dtype=torch.int32),
    )
    torch.cuda.synchronize()
    for a, b, c, size in zip(a_list, b_list, c_list, batch_sizes):
        torch.testing.assert_close(c[:size], a[:size] @ b, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_mixed_param_and_make_tensor_copies_keep_ws_for_param_copy():
    """Param-backed copies keep TMA/WS; the make_tensor copy stays a plain copy."""

    M = N = K = 128
    block_M, block_N, block_K = 64, 64, 32
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B_ptrs: T.Tensor([1], T.ptr),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            B = T.make_tensor(B_ptrs[0], (K, N), dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    kernel = _compile(main, out_idx=[2])

    src = kernel.get_kernel_source()
    # The param-backed A copy keeps TMA and the kernel is warp-specialized...
    assert "tl::tma_load(A_desc" in src
    assert "if (((int)threadIdx.x) < 128)" in src
    # ...while the make_tensor-backed B copy must not get a TMA descriptor.
    assert "B_desc" not in src

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, _make_ptr_table([b]))
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
