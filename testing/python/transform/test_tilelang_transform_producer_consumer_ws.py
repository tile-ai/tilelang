"""Tests for the warp-specialized producer/consumer pass."""

import re

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.layout import make_swizzled_layout
from tilelang.backend.target import determine_target


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


def device_bound_copy_pipelined(size=16, dtype="float16", threads=128):
    """A TMA-shaped copy whose global base comes from a pointer table."""

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out: T.Tensor((size, size), dtype),
    ):
        with T.Kernel(1, threads=threads):
            src = T.make_tensor(src_ptrs[0], (size, size), dtype)
            shared = T.alloc_shared((size, size), dtype)
            for _ in T.Pipelined(1, num_stages=1):
                T.copy(src, shared)
                T.copy(shared, out)

    return main


def dual_gemm_shared_accumulator(
    M,
    N,
    K0,
    K1,
    block_M,
    block_N,
    block_K,
    num_stages,
    dtype="float16",
    threads=128,
):
    """A prelude GEMM and a pipelined-loop GEMM accumulating into the same
    fragment, mirroring the #2547 `example_mamba_chunk_scan.py` pattern:
    `T.gemm(A0, B0, acc)` runs once before the loop, then
    `T.gemm(A1_k, B1_k, acc)` accumulates into the same `acc` inside a
    `T.Pipelined` loop. Under warp specialization, the prelude GEMM must be
    classified consumer-only so both GEMMs share one thread_range for `acc`.
    """

    @T.prim_func
    def main(
        A0: T.Tensor((M, K0), dtype),
        B0: T.Tensor((K0, N), dtype),
        A1: T.Tensor((M, K1), dtype),
        B1: T.Tensor((K1, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            acc = T.alloc_fragment((block_M, block_N), "float32")
            A0_shared = T.alloc_shared((block_M, K0), dtype)
            B0_shared = T.alloc_shared((K0, block_N), dtype)
            A1_shared = T.alloc_shared((block_M, block_K), dtype)
            B1_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            T.clear(acc)
            T.copy(A0[by * block_M, 0], A0_shared)
            T.copy(B0[0, bx * block_N], B0_shared)
            T.gemm(A0_shared, B0_shared, acc)  # prelude GEMM -> acc

            for ko in T.Pipelined(T.ceildiv(K1, block_K), num_stages=num_stages):
                T.copy(A1[by * block_M, ko * block_K], A1_shared)
                T.copy(B1[ko * block_K, bx * block_N], B1_shared)
                T.gemm(A1_shared, B1_shared, acc)  # loop GEMM -> same acc

            T.copy(acc, C_local)
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


def prelude_tma_bound_index(block=64, iters=2, dtype="float16", threads=128):
    """Pre-loop TMA load uses a scalar bind that is also consumed in the WS branch."""

    @T.prim_func
    def main(
        Q: T.Buffer((iters * block, block), dtype),
        K_in: T.Buffer((iters * block, block), dtype),
        idx: T.Buffer((1,), "int32"),
        O: T.Buffer((block, block), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            K_shared = T.alloc_shared((block, block), dtype)
            q = T.alloc_shared((block, block), dtype)
            acc = T.alloc_fragment((block, block), "float32")
            out = T.alloc_fragment((block, block), "float32")

            start = idx[0]
            T.copy(K_in[start, 0], K_shared)
            T.clear(out)
            for ko in T.Pipelined(iters, num_stages=2):
                T.copy(Q[ko * block, 0], q)
                T.clear(acc)
                T.gemm(K_shared, q, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block, block):
                    out[i, j] += acc[i, j] + T.cast(start, "float32")

            T.copy(out, O[0, 0])

    return main


def guarded_prelude_tma_postloop_scalar(block=64, iters=2, dtype="float16", threads=128):
    """Nested guarded pipeline whose post-loop consumer uses a prelude scalar."""

    @T.prim_func
    def main(
        Q: T.Buffer((iters * block * 2, block), dtype),
        K: T.Buffer((iters * block * 2, block), dtype),
        LSE: T.Buffer((block * 2, block), dtype),
        idx: T.Buffer((1,), "int32"),
        O: T.Buffer((block, block), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            Q_shared = T.alloc_shared((block, block), dtype)
            K_shared = T.alloc_shared((block, block), dtype)
            LSE_shared = T.alloc_shared((block, block), dtype)
            acc = T.alloc_fragment((block, block), "float32")

            base = idx[0]
            T.clear(acc)
            if base < block:
                T.copy(LSE[base, 0], LSE_shared)
                for ko in T.Pipelined(iters, num_stages=2):
                    T.copy(Q[ko * block + base, 0], Q_shared)
                    T.copy(K[ko * block + base, 0], K_shared)
                    T.gemm(Q_shared, K_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block, block):
                    O[i, j] = acc[i, j] + LSE_shared[i, j] + T.cast(base, "float32")

    return main


def explicit_cp_async_wait_position(iters=4, block=16, cp_elems=8, dtype="float16", threads=128):
    """A mixed TMA + explicit cp.async pipeline with cp.async consumed first."""

    @T.prim_func
    def main(
        A: T.Buffer((iters, block), dtype),
        B: T.Buffer((iters, cp_elems), dtype),
        B_out: T.Buffer((iters,), dtype),
        A_out: T.Buffer((iters, block), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((block,), dtype)
            B_shared = T.alloc_shared((cp_elems,), dtype)

            for ko in T.Pipelined(iters, num_stages=2):
                T.ptx_cp_async(
                    T.access_ptr(B_shared[0], "w", cp_elems),
                    T.access_ptr(B[ko, 0], "r", cp_elems),
                    cp_elems,
                )
                T.copy(A[ko, 0], A_shared)
                B_out[ko] = B_shared[0]
                for i in T.Parallel(block):
                    A_out[ko, i] = A_shared[i]

    return main


def grouped_gemm_padded_pipelined(
    batch_sizes,
    K,
    N,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    threads=256,
    dtype="float16",
):
    """Grouped GEMM with padded M tiles to exercise WS shared-prelude local vars."""

    batch_sizes = tuple(batch_sizes)
    batch_count = len(batch_sizes)
    batch_sum = sum(batch_sizes)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes)

    @T.prim_func
    def main(
        A: T.Buffer((batch_sum, K), dtype),
        B: T.Buffer((batch_count, K, N), dtype),
        C: T.Buffer((batch_sum, N), dtype),
        batch_sizes_buf: T.Buffer((batch_count,), "int32"),
        batch_offsets: T.Buffer((batch_count,), "int32"),
        batch_padded_offsets: T.Buffer((batch_count,), "int32"),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            cur_batch_idx = T.alloc_var("int32")
            cur_batch_size = T.alloc_var("int32")

            m_start_padded = bx * block_M
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes_buf[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(
                0,
                T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded),
            )

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start, ko * block_K], A_shared)
                T.copy(B[cur_batch_idx, ko * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return main


def grouped_gemm_reference(A, B, batch_sizes):
    import torch

    outputs = []
    start = 0
    for idx, size in enumerate(batch_sizes):
        end = start + size
        outputs.append(torch.mm(A[start:end], B[idx]))
        start = end
    return torch.cat(outputs, dim=0)


def grouped_gemm_inputs(batch_sizes, K, N, block_M, dtype="float16"):
    import math
    import torch

    batch_sizes = list(batch_sizes)
    batch_offsets = [0]
    batch_padded_offsets = [0]
    for i in range(len(batch_sizes) - 1):
        batch_offsets.append(batch_offsets[-1] + batch_sizes[i])
        batch_padded_offsets.append(batch_padded_offsets[-1] + math.ceil(batch_sizes[i] / block_M) * block_M)

    A = torch.randn(sum(batch_sizes), K, dtype=getattr(torch, dtype), device="cuda")
    B = torch.randn(len(batch_sizes), K, N, dtype=getattr(torch, dtype), device="cuda")
    batch_sizes_t = torch.tensor(batch_sizes, dtype=torch.int32, device="cuda")
    batch_offsets_t = torch.tensor(batch_offsets, dtype=torch.int32, device="cuda")
    batch_padded_offsets_t = torch.tensor(batch_padded_offsets, dtype=torch.int32, device="cuda")
    return A, B, batch_sizes_t, batch_offsets_t, batch_padded_offsets_t


def _find_after(src, needle, start=0):
    pos = src.find(needle, start)
    assert pos >= 0, f"missing substring: {needle}"
    return pos


def _count_device_bound_copy_annotations(func):
    annotated = 0
    total = 0

    def _visit(node):
        nonlocal annotated, total
        if not isinstance(node, tvm.tirx.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if str(node.op.name) not in {
            "tl.tileop.copy",
            "tl.tileop.async_copy",
            "tl.tileop.tma_copy",
        }:
            return
        total += 1
        value = node.annotations.get("tma_descriptor_base_is_device_bound") if node.annotations else None
        if isinstance(value, tvm.tirx.IntImm) and int(value.value) != 0:
            annotated += 1

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return annotated, total


def _compile_grouped_gemm_ws(batch_sizes=(63, 77), K=128, N=128, block_M=64, block_N=64, block_K=32):
    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    func = grouped_gemm_padded_pipelined(batch_sizes, K, N, block_M, block_N, block_K)
    kernel = _compile_tvm_ffi(func, pass_configs, out_idx=[2])
    return kernel, batch_sizes


@tilelang.testing.requires_cuda
def test_tiled_ws_skips_device_bound_tma_descriptor_base():
    func = device_bound_copy_pipelined().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.AnnotateDeviceBoundTmaCopies()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)

    annotated, total = _count_device_bound_copy_annotations(mod["main"])
    assert (annotated, total) == (1, 2)

    script = mod["main"].script()
    assert "tl_tiled_ws_applied" not in script
    assert "T.tma_copy" not in script


@tilelang.testing.requires_cuda
def test_tiled_ws_places_producer_in_first_warp_group():
    """Auto-WS should put the producer in the low threadIdx.x partition."""

    func = matmul_pipelined(64, 64, 64, 64, 32, 64, num_stages=2).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    assert "tl_tiled_ws_applied" in script
    branch = _find_after(script, "if tx < 128:")
    producer_tma = _find_after(script, "T.tma_copy", branch)
    consumer_branch = _find_after(script, "else:", producer_tma)
    consumer_gemm = _find_after(script, "T.gemm", consumer_branch)

    assert branch < producer_tma < consumer_branch < consumer_gemm
    assert "if 128 <= tx:" not in script


@tilelang.testing.requires_cuda
def test_tiled_ws_accepts_int64_pipeline_indices():
    """Inductor-style TIR may use int64 shapes and loop extents."""

    func = matmul_pipelined(
        T.int64(32),
        T.int64(1024),
        T.int64(256),
        T.int64(32),
        T.int64(64),
        T.int64(128),
        num_stages=4,
        dtype="bfloat16",
    ).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    assert "tl_tiled_ws_applied" in script
    assert "T.tma_copy" in script


def _run_grouped_gemm_ws(kernel, batch_sizes, K=128, N=128, block_M=64, dtype="float16"):
    import torch

    A, B, batch_sizes_t, batch_offsets_t, batch_padded_offsets_t = grouped_gemm_inputs(batch_sizes, K, N, block_M, dtype)
    out = kernel(A, B, batch_sizes_t, batch_offsets_t, batch_padded_offsets_t)
    ref = grouped_gemm_reference(A.float(), B.float(), batch_sizes)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    return out


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


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_dual_gemm_shared_accumulator_correctness():
    """Regression test for #2547.

    A prelude GEMM and a pipelined-loop GEMM accumulate into the same
    fragment. Before the fix, `LocalAccessCollector` never recorded the
    prelude GEMM's write to `acc`, so it was misclassified as
    `kKeepSharedPrelude` and stayed in the shared `[0, 2*threads)` prelude,
    while the loop GEMM ran consumer-only in `[threads, 2*threads)`.
    `LayoutInference` then aborted with "Get different layout for acc"
    because the two GEMMs disagreed on `acc`'s thread_range.
    """
    import torch

    M, N, K0, K1 = 64, 64, 64, 128
    block_M, block_N, block_K = 64, 64, 64
    func = dual_gemm_shared_accumulator(M, N, K0, K1, block_M, block_N, block_K, num_stages=2)
    target = determine_target()
    kernel = tilelang.compile(func, target=target, out_idx=[4])

    A0 = torch.randn(M, K0, dtype=torch.float16, device="cuda")
    B0 = torch.randn(K0, N, dtype=torch.float16, device="cuda")
    A1 = torch.randn(M, K1, dtype=torch.float16, device="cuda")
    B1 = torch.randn(K1, N, dtype=torch.float16, device="cuda")
    C = kernel(A0, B0, A1, B1)

    ref = A0.float() @ B0.float() + A1.float() @ B1.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_dual_gemm_shared_accumulator_disable_wgmma():
    """Same dual-GEMM-into-one-accumulator pattern, with WGMMA disabled.

    #2547 showed the crash was gated purely by warp specialization, not by
    WGMMA instruction selection: `tl.disable_wgmma=True` (WS still on) hit
    the identical thread_range conflict on an MMA-shaped fragment, so this
    is not an "accumulator unsupported" guard but a thread-range bookkeeping
    defect in the WS pass itself.
    """
    import torch

    M, N, K0, K1 = 64, 64, 64, 128
    block_M, block_N, block_K = 64, 64, 64
    func = dual_gemm_shared_accumulator(M, N, K0, K1, block_M, block_N, block_K, num_stages=2)
    target = determine_target()
    kernel = tilelang.compile(
        func,
        target=target,
        out_idx=[4],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WGMMA: True},
    )

    A0 = torch.randn(M, K0, dtype=torch.float16, device="cuda")
    B0 = torch.randn(K0, N, dtype=torch.float16, device="cuda")
    A1 = torch.randn(M, K1, dtype=torch.float16, device="cuda")
    B1 = torch.randn(K1, N, dtype=torch.float16, device="cuda")
    C = kernel(A0, B0, A1, B1)

    ref = A0.float() @ B0.float() + A1.float() @ B1.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


def _compile_tvm_ffi(func, pass_configs=None, **kwargs):
    with tilelang.testing.without_kernel_cache():
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs or {},
            **kwargs,
        )


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
    branch = src.find("if (((int)threadIdx.x) < 128)")
    first_wait = src.find(".wait(0)")

    assert min(k_load, v_load, branch, first_wait) >= 0
    assert k_load < v_load < branch < first_wait


@tilelang.testing.requires_cuda
def test_tiled_ws_explicit_cp_async_wait_precedes_first_consumer_read():
    """Explicit cp.async destinations must pull the consumer wait earlier."""

    func = explicit_cp_async_wait_position().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    assert "tl_tiled_ws_applied" in script
    assert "T.ptx_cp_async" in script
    assert "T.tma_copy" in script

    consumer_branch = _find_after(script, "else:")
    wait = _find_after(script, "T.mbarrier_wait_parity", consumer_branch)
    cp_async_read = _find_after(script, "B_out[ko] = B_shared[0]", consumer_branch)
    tma_read = _find_after(script, "A_out[ko, i] = A_shared", consumer_branch)

    assert wait < cp_async_read < tma_read


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_keeps_preloop_tma_scalar_bind_shared():
    """Scalar binds used by common pre-loop TMA copies must stay before WS."""

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(prelude_tma_bound_index(), pass_configs, out_idx=[3])
    src = kernel.get_kernel_source()

    start_bind = _find_after(src, "int start =")
    k_load = _find_after(src, "tl::tma_load(K_in_desc")
    branch = _find_after(src, "if (((int)threadIdx.x) < 128)")

    assert start_bind < k_load < branch


@tilelang.testing.requires_cuda
def test_tiled_ws_propagates_nested_postloop_liveness_to_outer_prelude():
    """Outer scalar binds used by nested post-loop consumers must stay shared."""

    func = guarded_prelude_tma_postloop_scalar().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    assert "tl_tiled_ws_applied" in script
    shared_base = _find_after(script, "base: T.int32 = idx[0]")
    guard = _find_after(script, "if base < 64:")
    branch = _find_after(script, 'T.attr([128, 128], "kWarpSpecializationScope", 0)')
    producer_base = script.find("base = idx[0]", branch)

    assert shared_base < guard < branch
    assert producer_base < 0


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_keeps_shared_prelude_local_vars_for_grouped_gemm():
    """Shared-prelude grouped-gemm indices must stay outside WS branches."""
    kernel, batch_sizes = _compile_grouped_gemm_ws()
    src = kernel.get_kernel_source()

    branch = _find_after(src, "if (((int)threadIdx.x) < 128)")
    cur_batch_idx_loop = _find_after(src, "for (int i = 0; i < 2; ++i)")
    m_start = _find_after(src, "int m_start =")
    actual_rows = _find_after(src, "int actual_rows =")

    assert cur_batch_idx_loop < m_start < actual_rows < branch
    _run_grouped_gemm_ws(kernel, batch_sizes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_does_not_clone_local_var_into_producer_branch():
    """Producer branch should reuse shared local.var state instead of cloning it."""
    kernel, batch_sizes = _compile_grouped_gemm_ws()
    src = kernel.get_kernel_source()

    assert "cur_batch_idx_producer_ws" not in src
    assert "cur_batch_size_producer_ws" not in src
    assert "tl::tma_load(B_desc" in src
    assert "cur_batch_idx);" in src
    _run_grouped_gemm_ws(kernel, batch_sizes)


def _wide_linear_reduce_kernel(rows, mids, width, batches, rows_per_batch, dtype, num_stages=2):
    """A grad_w-style batched reduction: each pipelined iteration TMA-loads a
    (rows_per_batch, width) slice (fixed middle index, strided rows) into a
    linear-layout shared buffer and accumulates it into a fragment."""

    @T.prim_func
    def main(
        src: T.Tensor((rows, mids, width), dtype),
        out: T.Tensor((mids, width), "float32"),
    ):
        with T.Kernel(mids, threads=128) as bx:
            smem = T.alloc_shared((rows_per_batch, width), dtype)
            acc = T.alloc_fragment((width,), "float32")
            T.clear(acc)
            for k in T.Pipelined(batches, num_stages=num_stages):
                T.copy(src[k * rows_per_batch : (k + 1) * rows_per_batch, bx, :], smem)
                for i in T.Serial(rows_per_batch):
                    for j in T.Parallel(width):
                        acc[j] += smem[i, j]
            T.copy(acc, out[bx, :])

    return main


def _run_wide_linear_reduce(kernel, rows, mids, width, dtype):
    import torch

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    src = torch.randn(rows, mids, width, device="cuda", dtype=torch_dtype)
    out = torch.zeros(mids, width, device="cuda", dtype=torch.float32)
    kernel(src, out)
    ref = src.float().sum(dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_multi_versioned_wide_linear_smem_keeps_tma():
    """A multi-versioned linear shared buffer with a >256-element row must
    stay a TMA producer: the version axis stays outside the per-copy TMA tile,
    while the width quotient stays outside its 256-element box. Pins the fix
    for the gapped-slice bijection ICHECK."""
    func = _wide_linear_reduce_kernel(rows=128, mids=4, width=512, batches=4, rows_per_batch=32, dtype="float32")
    kernel = tilelang.compile(func, target="cuda")
    src = kernel.get_kernel_source()
    assert "tl::tma_load(" in src
    assert "cp_async" not in src
    assert re.search(r"for \(int \w+ = 0; \w+ < 2; \+\+\w+\) \{\n\s*tl::tma_load\(", src)
    _run_wide_linear_reduce(kernel, 128, 4, 512, "float32")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_wide_linear_smem_multi_way_box_split():
    """1024 bf16 columns force a (256, 4) split of the contiguous box mode."""
    func = _wide_linear_reduce_kernel(rows=64, mids=2, width=1024, batches=4, rows_per_batch=16, dtype="bfloat16")
    kernel = tilelang.compile(func, target="cuda")
    src = kernel.get_kernel_source()
    assert "tl::tma_load(" in src
    _run_wide_linear_reduce(kernel, 64, 2, 1024, "bfloat16")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_wide_linear_smem_three_stages():
    """Three versions interleave the ring dimension above the 256-splits."""
    func = _wide_linear_reduce_kernel(rows=48, mids=2, width=768, batches=3, rows_per_batch=16, dtype="float32", num_stages=3)
    kernel = tilelang.compile(func, target="cuda")
    src = kernel.get_kernel_source()
    assert "cp_async" not in src
    assert re.search(r"for \(int \w+ = 0; \w+ < 3; \+\+\w+\) \{\n\s*tl::tma_load\(", src)
    _run_wide_linear_reduce(kernel, 48, 2, 768, "float32")


def make_im2col_kernel(channels=1, binding="none"):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, channels), T.float16),
        weights: T.Tensor((128,), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, channels), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, channels), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(weights, weight_shared)
                if binding == "pure":
                    step_base = T.int32(step % 2)
                    nhw_step = T.int32(step_base + 1)
                    T.im2col(data, data_shared, nhw_step, 0, 3, 1, 1, 1)
                elif binding == "stateful":
                    nhw_step = T.cast(weight_shared[0], T.int32)
                    T.im2col(data, data_shared, nhw_step, 0, 3, 1, 1, 1)
                elif binding == "global":
                    nhw_step = T.cast(weights[step], T.int32)
                    T.im2col(data, data_shared, nhw_step, 0, 3, 1, 1, 1)
                else:
                    T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                for i, j in T.Parallel(16, channels):
                    data_out[step, i, j] = data_shared[i, j]

    return main


def make_overwrite_kernel(mode, destination):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, 32), T.float16),
        weights: T.Tensor((128,), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, 32), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, 32), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(weights, weight_shared)
                T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                if mode == "fill":
                    if destination == "simt":
                        T.fill(data_shared, 0)
                    else:
                        T.fill(weight_shared, 0)
                else:
                    if destination == "simt":
                        for i, j in T.Parallel(16, 32):
                            data_shared[i, j] = T.float16(0)
                    else:
                        for i in T.Parallel(128):
                            weight_shared[i] = T.float32(0)
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                for i, j in T.Parallel(16, 32):
                    data_out[step, i, j] = data_shared[i, j]

    return main


def make_dual_kernel(binding="none", prelude=False, include_simt=True):
    @T.prim_func
    def main(
        data: T.Tensor((1, 8, 8, 32), T.float16),
        weights: T.Tensor((128,), T.float32),
        other_weights: T.Tensor((3, 128), T.float32),
        weights_out: T.Tensor((4, 128), T.float32),
        other_out: T.Tensor((4, 128), T.float32),
        data_out: T.Tensor((4, 16, 32), T.float16),
    ):
        with T.Kernel(1, threads=128):
            weight_shared = T.alloc_shared((128,), T.float32)
            other_shared = T.alloc_shared((128,), T.float32)
            data_shared = T.alloc_shared((16, 32), T.float16)
            if prelude:
                T.copy(weights, weight_shared)
            for step in T.Pipelined(4, num_stages=2):
                if not prelude:
                    T.copy(weights, weight_shared)
                if include_simt:
                    T.im2col(data, data_shared, step % 2, 0, 3, 1, 1, 1)
                if binding == "pure":
                    base_offset = T.int32(step % 2)
                    other_offset = T.int32(base_offset + 1)
                    T.copy(other_weights[other_offset, 0], other_shared)
                elif binding == "stateful":
                    other_offset = T.cast(weight_shared[0], T.int32) % 2
                    T.copy(other_weights[other_offset, 0], other_shared)
                else:
                    T.copy(other_weights[step % 2, 0], other_shared)
                for i in T.Parallel(128):
                    weight_shared[i] = T.float32(0)
                for i in T.Parallel(128):
                    other_out[step, i] = other_shared[i]
                for i in T.Parallel(128):
                    weights_out[step, i] = weight_shared[i]
                if include_simt:
                    for i, j in T.Parallel(16, 32):
                        data_out[step, i, j] = data_shared[i, j]

    return main


def make_shared_producers(order):
    @T.prim_func
    def main(data: T.Tensor((1, 8, 8, 32), T.float16), seed: T.Tensor((16, 32), T.float16), out: T.Tensor((4, 16, 32), T.float16)):
        with T.Kernel(1, threads=128):
            target_shared = T.alloc_shared((16, 32), T.float16)
            for step in T.Pipelined(4, num_stages=2):
                if order == "tma_first":
                    T.copy(seed, target_shared)
                    T.im2col(data, target_shared, step % 2, 0, 3, 1, 1, 1)
                else:
                    T.im2col(data, target_shared, step % 2, 0, 3, 1, 1, 1)
                    T.copy(seed, target_shared)
                for i, j in T.Parallel(16, 32):
                    out[step, i, j] = target_shared[i, j]

    return main


def make_preceding_consumer():
    @T.prim_func
    def main(
        a: T.Tensor((128,), T.float32),
        b: T.Tensor((128,), T.float32),
        out_a: T.Tensor((4, 128), T.float32),
        out_b: T.Tensor((4, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((128,), T.float32)
            target_shared = T.alloc_shared((128,), T.float32)
            for step in T.Pipelined(4, num_stages=2):
                T.copy(a, a_shared)
                for i in T.Parallel(128):
                    target_shared[i] = T.float32(0)
                T.copy(b, target_shared)
                for i in T.Parallel(128):
                    out_a[step, i] = a_shared[i]
                    out_b[step, i] = target_shared[i]

    return main


def touches(expr, buffer_name):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(
        expr,
        lambda node: found.append(node) if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name.startswith(buffer_name) else None,
    )
    return bool(found)


def same_barrier(left, right):
    if not isinstance(left, tvm.tirx.BufferLoad) or not isinstance(right, tvm.tirx.BufferLoad):
        raise RuntimeError("expected a buffer-load barrier reference")
    return left.buffer.data.same_as(right.buffer.data) and tvm.ir.structural_equal(left.indices, right.indices, map_free_vars=True)


def target_barriers(body, producer, buffer_name, destination):
    forward = []
    phases = []
    reuse = []
    last_wait = []

    def collect_forward(node):
        if not isinstance(node, tvm.tirx.Call):
            return
        name = getattr(node.op, "name", "")
        if destination == "simt":
            if name.endswith("ptx_cp_async_barrier"):
                forward.append(node.args[0])
            if name == "tl.tileop.im2col" and touches(node.args[1], buffer_name):
                phases.append(node.annotations["tl.pipeline_mbar_phase_expr"])
        elif name == "tl.tileop.tma_copy" and touches(node.args[1], buffer_name):
            if "barrier" not in node.annotations:
                raise RuntimeError("target TMA copy has no completion barrier")
            forward.append(node.annotations["barrier"])
            phases.append(
                node.annotations["tl.pipeline_mbar_phase_expr"]
                if "tl.pipeline_mbar_phase_expr" in node.annotations
                else tvm.tirx.IntImm("int32", 0)
            )

    tvm.tirx.stmt_functor.post_order_visit(producer if destination == "simt" else body, collect_forward)

    def collect_reuse(node):
        if not isinstance(node, tvm.tirx.Call):
            return
        name = getattr(node.op, "name", "")
        if name.endswith("mbarrier_wait_parity"):
            last_wait[:] = [node.args[0]]
        matches = (destination == "tma" and name == "tl.tileop.tma_copy") or (destination == "simt" and name == "tl.tileop.im2col")
        if matches and touches(node.args[1], buffer_name):
            if not last_wait:
                raise RuntimeError("target producer has no preceding reuse wait")
            reuse.append(last_wait[0])

    tvm.tirx.stmt_functor.post_order_visit(producer, collect_reuse)
    if len(forward) != 1 or len(phases) != 1 or len(reuse) > 1:
        raise RuntimeError(f"expected one target completion barrier, got {len(forward)} forward/{len(reuse)} reuse")
    return forward[0], phases[0], reuse[0] if reuse else None


def events(stmt, buffer_name, forward, phase, reuse):
    result = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name.startswith(buffer_name):
            result.append("read")
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.name.startswith(buffer_name):
            result.append("overwrite")
        if isinstance(node, tvm.tirx.Call):
            name = getattr(node.op, "name", "")
            if name.endswith("mbarrier_wait_parity"):
                matches = same_barrier(node.args[0], forward) and tvm.ir.structural_equal(node.args[1], phase, map_free_vars=True)
                result.append("target_wait" if matches else "other_wait")
            if name.endswith("ptx_arrive_barrier") and reuse is not None and same_barrier(node.args[0], reuse):
                result.append("target_release")
            if name == "tl.tileop.fill" and touches(node.args[0], buffer_name):
                result.append("overwrite")

    tvm.tirx.stmt_functor.post_order_visit(stmt, visit)
    return result


def ordering_violations(producer, consumer, reuse):
    violations = []
    if "overwrite" in producer:
        violations.append("consumer overwrite moved into producer issue sequence")
    access_positions = [i for i, event in enumerate(consumer) if event in ("read", "overwrite")]
    if access_positions and ("target_wait" not in consumer or access_positions[0] < consumer.index("target_wait")):
        violations.append("consumer access precedes its target completion wait")
    if (
        reuse is not None
        and access_positions
        and ("target_release" not in consumer or consumer.index("target_release") < access_positions[-1])
    ):
        violations.append("target stage released before its final consumer access")
    return violations


def is_specialized(func):
    return "tl_tiled_ws_applied" in func.attrs


def producer_consumer_split(func):
    def has_tma(stmt):
        calls = []
        tvm.tirx.stmt_functor.post_order_visit(
            stmt,
            lambda node: (
                calls.append(node) if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.tma_copy" else None
            ),
        )
        return bool(calls)

    splits = []

    def visit(node):
        if isinstance(node, tvm.tirx.IfThenElse) and node.else_case is not None and has_tma(node.then_case):
            splits.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    if len(splits) != 1:
        raise RuntimeError(f"expected one producer/consumer split, got {len(splits)}")
    return splits[0]


def check_fallback(before, after):
    if is_specialized(after):
        raise RuntimeError("dependent producer work was warp-specialized")

    def remove_pipeline_hint(node):
        if isinstance(node, tvm.tirx.For) and "num_stages" in node.annotations:
            annotations = {key: value for key, value in node.annotations.items() if key != "num_stages"}
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                thread_binding=node.thread_binding,
                annotations=annotations,
                step=node.step,
            )
        return None

    expected_body = tvm.tirx.stmt_functor.ir_transform(before.body, None, remove_pipeline_hint, ["tirx.For"])
    if not tvm.ir.structural_equal(before.with_body(expected_body), after):
        raise RuntimeError("fallback changed the original loop beyond its num_stages hint")


def lower_producer_consumer_scenario(func):
    target = determine_target({"kind": "cuda", "arch": "sm_100"}, return_object=True)
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    before = tilelang.transform.MaterializeKernelLaunch()(mod)
    after = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(before)
    scoped = tvm.tirx.transform.ConvertSSA()(after)["main"]
    bound = [*scoped.params, *(buf.data for buf in scoped.buffer_map.values())]
    undefined = tvm.tirx.analysis.undefined_vars(scoped.body, bound)
    if undefined:
        pytest.fail(f"transformed kernel references undefined variables: {undefined}")
    with tvm.transform.PassContext(), target:
        artifact = tilelang.lower(func, target=target)
    if not artifact.kernel_source:
        pytest.fail("compiler returned no CUDA source")
    return before["main"], after["main"]


def check_consumer_accesses(func, buffer_name, destination, overwrite=False):
    if not is_specialized(func):
        pytest.fail("independent producers lost warp specialization")
    split = producer_consumer_split(func)
    forward, phase, reuse = target_barriers(func.body, split.then_case, buffer_name, destination)
    producer = events(split.then_case, buffer_name, forward, phase, reuse)
    consumer = events(split.else_case, buffer_name, forward, phase, reuse)
    violations = ordering_violations(producer, consumer, reuse)
    if violations:
        pytest.fail(str(violations))
    if "read" not in consumer:
        pytest.fail("consumer read was lost from the transformed IR")
    if overwrite and "overwrite" not in consumer:
        pytest.fail("consumer overwrite was lost from the transformed IR")


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("channels,binding", [(1, "none"), (32, "none"), (32, "pure"), (32, "stateful"), (32, "global")])
def test_tiled_ws_simt_im2col(channels, binding):
    before, after = lower_producer_consumer_scenario(make_im2col_kernel(channels, binding))
    if binding in ("stateful", "global"):
        check_fallback(before, after)
        return
    check_consumer_accesses(after, "data_shared", "simt")
    arrivals = []
    tvm.tirx.stmt_functor.post_order_visit(
        producer_consumer_split(after).then_case,
        lambda node: (
            arrivals.append(node)
            if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "").endswith("ptx_arrive_barrier")
            else None
        ),
    )
    if not arrivals:
        pytest.fail("SIMT producer has no release arrival for synchronous stores")


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("mode", ["fill", "parallel"])
@pytest.mark.parametrize("destination", ["simt", "tma"])
def test_tiled_ws_waits_before_consumer_overwrite(mode, destination):
    _, after = lower_producer_consumer_scenario(make_overwrite_kernel(mode, destination))
    check_consumer_accesses(after, "data_shared" if destination == "simt" else "weight_shared", destination, overwrite=True)


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize(
    "binding,prelude,include_simt",
    [("none", False, True), ("pure", False, True), ("pure", False, False), ("none", True, True), ("stateful", False, False)],
)
def test_tiled_ws_multiple_producer_dependencies(binding, prelude, include_simt):
    before, after = lower_producer_consumer_scenario(make_dual_kernel(binding, prelude, include_simt))
    if binding == "stateful":
        check_fallback(before, after)
    else:
        check_consumer_accesses(after, "weight_shared", "tma", overwrite=True)


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("order", ["tma_first", "im2col_first", "consumer_before_tma"])
def test_tiled_ws_preserves_conflicting_producer_order(order):
    func = make_preceding_consumer() if order == "consumer_before_tma" else make_shared_producers(order)
    before, after = lower_producer_consumer_scenario(func)
    check_fallback(before, after)


if __name__ == "__main__":
    test_tiled_ws_skips_device_bound_tma_descriptor_base()
    test_tiled_ws_places_producer_in_first_warp_group()
    test_tiled_ws_accepts_int64_pipeline_indices()
    test_tiled_ws_stage1_dynamic_loop_start()
    test_tiled_ws_correctness()
    test_tiled_ws_stage3()
    test_tiled_ws_dual_gemm_shared_accumulator_correctness()
    test_tiled_ws_dual_gemm_shared_accumulator_disable_wgmma()
    test_tiled_ws_swizzled_layout_allows_ws()
    test_tiled_ws_incompatible_layout_blocks_ws()
    test_tiled_ws_sinks_preloop_tma_waits_into_consumer()
    test_tiled_ws_explicit_cp_async_wait_precedes_first_consumer_read()
    test_tiled_ws_keeps_shared_prelude_local_vars_for_grouped_gemm()
    test_tiled_ws_does_not_clone_local_var_into_producer_branch()
    test_tiled_ws_multi_versioned_wide_linear_smem_keeps_tma()
    test_tiled_ws_wide_linear_smem_multi_way_box_split()
    test_tiled_ws_wide_linear_smem_three_stages()
