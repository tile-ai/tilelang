import re

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T


def _compile_tvm_ffi(func, pass_configs=None):
    with tilelang.testing.without_kernel_cache():
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs or {},
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("use_cp_async", [False, True])
def test_ws_keeps_full_producer_extent_for_lowered_simt_copy(use_cp_async):
    M, N, K = 128, 64, 64
    block_M, block_N, block_K = 64, 64, 32
    num_stages = 2
    threads = 256
    e_factor = 8

    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), T.int8),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), T.int8)
            C_frag = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared, prefer_instruction="cp_async" if use_cp_async else None)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, transpose_A=False, transpose_E=False, transpose_B=False)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    kernel = _compile_tvm_ffi(main)
    src = kernel.get_kernel_source()
    flat_src = " ".join(src.split())

    assert "__launch_bounds__(512, 1)" in src
    assert "if (((int)threadIdx.x) < 256) {" in flat_src
    assert "tl::tl_shuffle_elect<256>()" in src or "if (((int)threadIdx.x) == 0) {" in src
    assert re.search(r"tl::__sync_thread_partial\(\d+, 256\);", src), src
    producer = flat_src.split("if (((int)threadIdx.x) < 256) {", 1)[1].split("} else {", 1)[0]
    assert re.findall(r"mbarrier\[\d+\]\.init\((\d+)\);", src) == ["256"] * 4, src
    assert producer.count(".arrive();") == 1, src
    assert producer.rindex("E_shared") < producer.index(".arrive();"), src
    assert ("cp_async_gs" in producer) == use_cp_async, src
    if use_cp_async:
        assert producer.index("mbarrier_cp_async_arrive(") < producer.index(".arrive();"), src
        assert "mbarrier_cp_async_arrive_noinc" not in producer, src

    a_sparse = torch.randn(M, K // 2, device="cuda", dtype=torch.float16)
    a = torch.zeros(M, K, device="cuda", dtype=torch.float16)
    a.reshape(M, K // 4, 4)[:, :, :2] = a_sparse.reshape(M, K // 4, 2)
    # Each metadata nibble selects positions 0 and 1 in a group of four.
    e = torch.full((M, K // e_factor), 0x44, device="cuda", dtype=torch.int8)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel(a_sparse, e, b, c)
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("coordinate", ["direct", "indexed", "leading"])
def test_ws_tma_producer_participation(coordinate):
    M, N, K = 256, 256, 512
    block_M, block_N, block_K = 128, 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        I: T.Tensor((K // block_K,), T.int32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_frag = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                if coordinate == "leading":
                    b_k = I[k] * block_K
                T.copy(A[by * block_M, k * block_K], A_shared)
                if coordinate == "indexed":
                    b_k = I[k] * block_K
                T.copy(B[k * block_K if coordinate == "direct" else b_k, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_frag)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    kernel = _compile_tvm_ffi(main)
    src = kernel.get_kernel_source()
    flat_src = " ".join(src.split())

    producer = flat_src.split("if (((int)threadIdx.x) < 128) {", 1)[1].split("} else {", 1)[0]
    fast_path = coordinate == "direct"
    assert ("if (((int)threadIdx.x) < 32)" in producer) == fast_path, src
    assert ("arrive_and_expect_tx" in producer) == fast_path, src
    assert re.findall(r"mbarrier\[\d+\]\.init\((\d+)\);", src) == ["1" if fast_path else "128"] * 2 + ["128"] * 2, src
    assert producer.count(".arrive();") == (0 if fast_path else 1), src

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    indices = torch.arange(K // block_K - 1, -1, -1, device="cuda", dtype=torch.int32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, indices, c)
    reference_b = b.reshape(K // block_K, block_K, N)[indices.long()].reshape(K, N) if coordinate != "direct" else b
    torch.testing.assert_close(c, a.float() @ reference_b.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_ws_publishes_synchronous_fp8_conversion():
    """A TMA producer and synchronous conversion must both finish before GEMM."""
    M, N, K = 512, 1024, 3072

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, K), T.float8_e4m3fn),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(N // 16, M // 16, threads=64) as (bn, bm):
            a = T.alloc_shared((16, 128), T.bfloat16)
            w = T.alloc_shared((16, 128), T.bfloat16)
            acc = T.alloc_fragment((16, 16), T.float32)
            T.clear(acc)
            for bk in T.Pipelined(K // 128, num_stages=2):
                T.copy(A[bm * 16, bk * 128], a)
                for i, j in T.Parallel(16, 128):
                    w[i, j] = T.cast(W[bn * 16 + i, bk * 128 + j], T.bfloat16)
                T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, C[bm * 16, bn * 16])

    kernel = _compile_tvm_ffi(main)
    row_tags = torch.arange(M, device="cuda") // 16 % 4 + 1
    k_tags = torch.arange(K, device="cuda") // 128 + 1
    a = (row_tags[:, None] * k_tags[None, :]).bfloat16()
    w = ((k_tags % 7) + 1).expand(N, K).to(torch.float8_e4m3fn)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    expected = (a.float() @ w.to(torch.bfloat16).float().T).bfloat16()
    for _ in range(200):
        c.fill_(float("nan"))
        kernel(a, w, c)
        torch.testing.assert_close(c, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
