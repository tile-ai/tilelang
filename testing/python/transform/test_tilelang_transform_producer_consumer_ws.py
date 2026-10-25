"""Tests for the warp-specialized producer/consumer pass."""

import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tvm_tirx import ir, tirx, sym
from tilelang.testing.ir import collect_calls, collect_nodes
from tilelang._compat import stmt_functor
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


def _tma_copies(body):
    return [call for call in collect_calls(body, op="tl.tileop.copy") if call.attrs and bool(call.attrs.get("is_tma_copy", False))]


def _automatic_ws(function):
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    module = tvm.IRModule.from_expr(function.with_attr("global_symbol", "main"))
    module = tirx.transform.BindTarget(target)(module)
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    return tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(module)["main"]


def _ws_roles(function):
    assert bool(function.attrs.get("tl_tiled_ws_applied", False))
    assert tirx.analysis.check_well_formed(function)
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)))
    (concurrent,) = [region for region in collect_nodes(function.body, ir.RegionStmt) if bool(region.attrs.get("tl.ws_concurrent", False))]
    branches = list(concurrent.body.seq)
    assert len(branches) >= 2 and all(isinstance(branch, ir.If) for branch in branches)
    (launch,) = [
        region
        for region in collect_nodes(function.body, ir.RegionStmt)
        if region.op.name == "tirx.launch_thread" and region.args[0].value == "threadIdx.x"
    ]
    thread = launch.body_params[0]
    analyzer = sym.Analyzer()
    used = set()
    for branch in branches:
        active = {
            index
            for index in range(int(launch.args[1]))
            if bool(int(analyzer.simplify(stmt_functor.substitute(branch.condition, {thread: ir.const(index)}))))
        }
        assert active and not used.intersection(active)
        used.update(active)
    assert used == set(range(int(launch.args[1])))
    worker, producer = branches[:2]
    assert not _tma_copies(worker.then_case)
    assert _tma_copies(producer.then_case)
    assert not collect_calls(producer.then_case, op="tl.tileop.gemm")
    return worker.then_case, producer.then_case


def _role_scalar(role, name):
    (binding,) = [binding for binding in collect_nodes(role, ir.Bind) if re.fullmatch(rf"{re.escape(name)}(?:_\d+)?", binding.var.name)]
    return binding


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
        value = node.attrs.get("tma_descriptor_base_is_device_bound") if node.attrs else None
        if isinstance(value, (bool, int, tvm.tirx.IntImm)) and int(value) != 0:
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
def test_tiled_ws_uses_disjoint_worker_and_producer_partitions():
    """Native automatic scheduling separates consumer and TMA producer warps."""

    func = matmul_pipelined(64, 64, 64, 64, 32, 64, num_stages=2).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    worker, producer = _ws_roles(mod["main"])
    assert len(_tma_copies(producer)) == 2
    assert len(collect_calls(worker, op="tl.tileop.gemm")) == 1
    assert len(collect_calls(worker, op="tl.mbarrier_wait_parity")) == 2


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
    assert _tma_copies(mod["main"].body)


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

    assert re.search(r"__launch_bounds__\(256(?:, 1)?\)", source)

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
@pytest.mark.parametrize("warp_specialized", [False, True])
def test_tiled_ws_dual_gemm_shared_accumulator_correctness(warp_specialized):
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
    kernel = tilelang.compile(func, target=target, out_idx=[4], pass_configs={"tl.disable_warp_specialized": not warp_specialized})

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
    assert re.search(r"__launch_bounds__\(256(?:, 1)?\)", src)
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
    assert re.search(r"__launch_bounds__\(128(?:, 1)?\)", src)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_sinks_preloop_tma_waits_into_consumer():
    """Pre-loop TMA loads should not emit immediate waits in the common prelude."""

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(prelude_tma_wait_sink(), pass_configs, out_idx=[3])
    src = kernel.get_kernel_source()

    worker, producer = _ws_roles(_automatic_ws(prelude_tma_wait_sink()))
    sources = [copy.args[0].args[0].source.name for copy in _tma_copies(producer)]
    assert sources[:2] == ["K_in", "V_in"]
    operations = collect_nodes(worker, ir.Call)
    gemm_position = next(index for index, call in enumerate(operations) if call.op.name == "tl.tileop.gemm")
    waits = [index for index, call in enumerate(operations) if call.op.name == "tl.mbarrier_wait_parity"]
    assert len(waits) >= 2 and waits[0] < waits[1] < gemm_position
    assert "tl::tma_load" in src and "tl::mbarrier_wait" in src


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
    worker, producer = _ws_roles(mod["main"])
    assert collect_calls(producer, op="tl.ptx_cp_async")
    assert collect_calls(producer, op="tirx.ptx.cp_async_commit_group")
    events = []
    stmt_functor.post_order_visit(worker, lambda node: events.append(node) if isinstance(node, (ir.Evaluate, ir.TensorStore)) else None)
    positions = {}
    for index, statement in enumerate(events):
        if isinstance(statement, ir.TensorStore) and statement.dest.name in ("A_out", "B_out"):
            positions[statement.dest.name] = index
        if (
            isinstance(statement, ir.Evaluate)
            and isinstance(statement.value, ir.Call)
            and statement.value.op.name == "tl.mbarrier_wait_parity"
        ):
            positions[statement.value.args[0].source.name] = index
    assert positions["ws_barrier_B_shared"] < positions["B_out"] < positions["ws_barrier_A_shared"] < positions["A_out"]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_keeps_preloop_tma_scalar_bind_shared():
    """Role-local scalar replay must dominate pre-loop TMA and consumer uses."""

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(prelude_tma_bound_index(), pass_configs, out_idx=[3])
    src = kernel.get_kernel_source()

    worker, producer = _ws_roles(_automatic_ws(prelude_tma_bound_index()))
    worker_start = _role_scalar(worker, "start")
    producer_start = _role_scalar(producer, "start")
    tvm.ir.assert_structural_equal(worker_start.value, producer_start.value)
    assert any(variable.same_as(producer_start.var) for variable in tirx.analysis.undefined_vars(_tma_copies(producer)[0]))
    assert "tl::tma_load" in src


@tilelang.testing.requires_cuda
def test_tiled_ws_propagates_nested_postloop_liveness_to_outer_prelude():
    """Guarded outer scalars must remain available in both scheduled roles."""

    func = guarded_prelude_tma_postloop_scalar().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)
    script = mod["main"].script()

    assert "tl_tiled_ws_applied" in script
    worker, producer = _ws_roles(mod["main"])
    bases = [_role_scalar(role, "base") for role in (worker, producer)]
    tvm.ir.assert_structural_equal(bases[0].value, bases[1].value)
    for role, binding in zip((worker, producer), bases):
        assert any(
            any(variable.same_as(binding.var) for variable in tirx.analysis.undefined_vars(guard.condition))
            for guard in collect_nodes(role, ir.If)
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_keeps_shared_prelude_local_vars_for_grouped_gemm():
    """Grouped GEMM replays role-local indices without duplicating storage."""
    kernel, batch_sizes = _compile_grouped_gemm_ws()
    src = kernel.get_kernel_source()

    function = _automatic_ws(grouped_gemm_padded_pipelined(batch_sizes, 128, 128, 64, 64, 32))
    worker, producer = _ws_roles(function)
    for role in (worker, producer):
        _role_scalar(role, "m_start")
    _role_scalar(worker, "actual_rows")
    assert len(re.findall(r"\bint\s+cur_batch_idx_ptr\[1\];", src)) == 1
    _run_grouped_gemm_ws(kernel, batch_sizes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tiled_ws_does_not_clone_local_var_into_producer_branch():
    """Producer and worker roles must reference one private-state allocation."""
    kernel, batch_sizes = _compile_grouped_gemm_ws()
    src = kernel.get_kernel_source()

    assert "cur_batch_idx_producer_ws" not in src
    assert "cur_batch_size_producer_ws" not in src
    assert "tl::tma_load" in src
    assert len(re.findall(r"\bint\s+cur_batch_idx_ptr\[1\];", src)) == 1
    assert "cur_batch_idx_ptr[0]" in src
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


if __name__ == "__main__":
    test_tiled_ws_skips_device_bound_tma_descriptor_base()
    test_tiled_ws_uses_disjoint_worker_and_producer_partitions()
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
