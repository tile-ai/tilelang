"""Tests for the warp-specialized producer/consumer pass."""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.layout import make_swizzled_layout
from tilelang.utils.target import determine_target


def matmul_pipelined(M, N, K, block_M, block_K, block_N, num_stages, dtype="float16", threads=128):
    """A simple pipelined GEMM using T.copy + T.gemm tile ops."""

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_windowed_pipelined(
    M,
    N,
    K,
    block_M,
    block_K,
    block_N,
    num_stages,
    window_tiles=2,
    dtype="float16",
    threads=128,
):
    """A pipelined GEMM whose K-loop has a dynamic lower bound."""

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)

            start = T.max(0, bx - (window_tiles - 1))
            end = T.min(T.ceildiv(K, block_K), bx + 1)
            for ko in T.Pipelined(start, end, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def prelude_tma_wait_sink(block=64, iters=2, dtype="float16", threads=128):
    """A tiled-WS kernel with pre-loop TMA loads consumed at different points."""

    @T.prim_func
    def main(
        Q: T.Buffer((iters * block, block), dtype),
        K_in: T.Buffer((block, block), dtype),
        V_in: T.Buffer((block, block), dtype),
        O: T.Buffer((block, block), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            K_shared = T.alloc_shared((block, block), dtype)
            V_shared = T.alloc_shared((block, block), dtype)
            q = T.alloc_shared((block, block), dtype)
            acc0 = T.alloc_fragment((block, block), "float32")
            acc1 = T.alloc_fragment((block, block), "float32")
            out = T.alloc_fragment((block, block), "float32")

            T.copy(K_in[0, 0], K_shared)
            T.copy(V_in[0, 0], V_shared)
            T.clear(out)
            for ko in T.Pipelined(iters, num_stages=2):
                T.copy(Q[ko * block, 0], q)
                T.clear(acc0)
                T.gemm(K_shared, q, acc0, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.clear(acc1)
                T.gemm(V_shared, q, acc1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block, block):
                    out[i, j] = acc0[i, j] + acc1[i, j]

            T.copy(out, O[0, 0])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_stage1_dynamic_loop_start():
    """Stage-1 tiled WS should handle dynamic pipeline loop bounds."""
    import torch

    M, N, K = 64, 128, 64
    block_M, block_K, block_N = 64, 32, 64
    func = matmul_windowed_pipelined(
        M,
        N,
        K,
        block_M,
        block_K,
        block_N,
        num_stages=1,
        window_tiles=2,
    )
    target = determine_target()
    kernel = tilelang.compile(func, target=target, out_idx=[2])
    source = kernel.get_kernel_source()

    assert "__launch_bounds__(256, 1)" in source

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)

    ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    num_k_tiles = (K + block_K - 1) // block_K
    num_n_tiles = (N + block_N - 1) // block_N
    for bx in range(num_n_tiles):
        start = max(0, bx - 1)
        end = min(num_k_tiles, bx + 1)
        n_slice = slice(bx * block_N, min((bx + 1) * block_N, N))
        acc = torch.zeros(M, n_slice.stop - n_slice.start, dtype=torch.float32, device="cuda")
        for ko in range(start, end):
            k_slice = slice(ko * block_K, min((ko + 1) * block_K, K))
            acc += A[:, k_slice].float() @ B[k_slice, n_slice].float()
        ref[:, n_slice] = acc

    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_correctness():
    """End-to-end correctness test: pipelined GEMM via tiled WS."""
    import torch

    M, N, K = 256, 256, 256
    func = matmul_pipelined(M, N, K, 64, 32, 64, num_stages=2)
    target = determine_target()
    kernel = tilelang.compile(func, target=target, out_idx=[2])

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)

    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_stage3():
    """Pipelined GEMM with 3 stages."""
    import torch

    M, N, K = 512, 512, 512
    func = matmul_pipelined(M, N, K, 128, 64, 128, num_stages=3)
    target = determine_target()
    kernel = tilelang.compile(func, target=target, out_idx=[2])

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)

    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


def _compile_tvm_ffi(func, pass_configs=None, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs or {},
            **kwargs,
        )
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_swizzled_layout_allows_ws():
    """Swizzled layout on a TMA copy target should NOT block warp specialization.

    Swizzled layouts are valid TMA layouts (TMA supports 32B/64B/128B swizzle).
    Layout::Expand correctly handles MVB expansion for swizzled layouts.
    """
    import torch

    M, N, K = 256, 256, 256
    block_M, block_K, block_N = 64, 64, 64

    @T.prim_func
    def gemm_swizzled(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.annotate_layout({A_shared: make_swizzled_layout(A_shared), B_shared: make_swizzled_layout(B_shared)})

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(gemm_swizzled, pass_configs, out_idx=[2])
    src = kernel.get_kernel_source()

    # WS should be applied: launch bounds should include producer warp group
    assert "__launch_bounds__(256, 1)" in src
    # TMA loads should be present
    assert "tl::tma_load" in src

    # Correctness check
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_incompatible_layout_blocks_ws():
    """A non-swizzle, non-linear layout on ALL TMA copy targets should block WS.

    If every copy that could be a TMA producer has an incompatible layout,
    there are no real TMA candidates and WS should not apply.
    """
    from tilelang.layout import Layout

    M, K = 16, 128
    block_m, block_k = 16, 128

    # A padded layout: (i, j) -> i * (block_k + 8) + j
    # This is neither a swizzle layout nor a linear layout (output shape != input shape).
    padded_continuous = block_k + 8
    padded_layout = Layout([block_m, block_k], lambda i, j: i * padded_continuous + j)

    @T.prim_func
    def copy_with_padded_layout(
        x: T.Tensor((M, K), "float16"),
        y: T.Tensor((M, K), "float16"),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=128) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), "float16")

            T.annotate_layout({x_shared: padded_layout})

            for _ in T.Pipelined(1, num_stages=1):
                T.copy(x[pid_m * block_m, 0], x_shared)
                T.copy(x_shared, y[pid_m * block_m, 0])

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(copy_with_padded_layout, pass_configs, out_idx=[1])
    src = kernel.get_kernel_source()

    # WS should NOT be applied: no producer/consumer split
    assert "__launch_bounds__(256, 1)" not in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_sinks_preloop_tma_waits_into_consumer():
    """Pre-loop TMA loads should not emit immediate waits in the common prelude."""

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(prelude_tma_wait_sink(), pass_configs, out_idx=[3])
    src = kernel.get_kernel_source()

    k_load = src.find("tl::tma_load(K_in_desc")
    v_load = src.find("tl::tma_load(V_in_desc")
    branch = src.find("if (128 <= ((int)threadIdx.x))")
    first_wait = src.find(".wait(0)")

    assert min(k_load, v_load, branch, first_wait) >= 0
    assert k_load < v_load < branch < first_wait


if __name__ == "__main__":
    test_tiled_ws_stage1_dynamic_loop_start()
    test_tiled_ws_correctness()
    test_tiled_ws_stage3()
    test_tiled_ws_swizzled_layout_allows_ws()
    test_tiled_ws_incompatible_layout_blocks_ws()
    test_tiled_ws_sinks_preloop_tma_waits_into_consumer()
