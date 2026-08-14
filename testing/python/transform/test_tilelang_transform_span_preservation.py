"""Tests that source spans survive the lowering pass chain.

Span injection is validated by test_tilelang_language_span.py on freshly
parsed IR. These tests cover the other half of the pipeline: lowering passes
must not drop spans (StmtMutator copy-on-write preserves them; only explicit
node reconstruction can lose them), so that codegen-time source mapping
(`#line` emission) has something to work with.
"""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ir import get_stmt_span, span_to_location
from tvm.ir.instrument import pass_instrument
from tvm.tirx.stmt_functor import post_order_visit


def _count_spans(mod) -> tuple[int, int]:
    """Return (stmts_with_span, total_stmts) over all PrimFuncs in the module."""
    with_span = 0
    total = 0
    for _, func in mod.functions.items():
        if not isinstance(func, tvm.tirx.PrimFunc):
            continue

        def visit(node):
            nonlocal with_span, total
            if isinstance(node, tvm.tirx.Stmt):
                total += 1
                if get_stmt_span(node) is not None:
                    with_span += 1

        post_order_visit(func.body, visit)
    return with_span, total


@pass_instrument
class _SpanCoverageRecorder:
    """Records (pass_name, with_span, total) after every pass."""

    def __init__(self):
        self.rows: list[tuple[str, int, int]] = []

    def run_after_pass(self, mod, info):
        self.rows.append((info.name, *_count_spans(mod)))


def _marker_line(marker: str) -> int:
    with open(__file__) as f:
        for i, line in enumerate(f, 1):
            if marker in line:
                return i
    raise ValueError(f"marker not found: {marker}")


def _make_vector_add():
    @T.prim_func
    def main(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        with T.Kernel(1024):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0  # span_marker_vadd_store

    return main


def _make_gemm():
    M = N = K = 128
    block_M = block_N = block_K = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16", scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), "float16", scope="shared")
            C_local = T.alloc_shared((block_M, block_N), "float32", scope="shared")
            T.clear(C_local)  # span_marker_clear
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)  # span_marker_copy_a
                T.copy(B[ko * block_K, bx * block_N], B_shared)  # span_marker_copy_b
                T.gemm(A_shared, B_shared, C_local)  # span_marker_gemm
            T.copy(C_local, C[by * block_M, bx * block_N])  # span_marker_copy_out

    return main


def _lower_with_recorder(func, target: str) -> _SpanCoverageRecorder:
    recorder = _SpanCoverageRecorder()
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3, instruments=[recorder]):
        tilelang.lower(func, target=target)
    return recorder


# Passes whose span propagation was fixed; they must never drop a span
# (statement *deletion* is fine — it also reduces the total).
_SPAN_SAFE_PASSES = {
    "tl.MaterializeKernelLaunch",
    "tl.AddWrapperForSingleBufStore",
    "tl.IfStmtBinding",
    "tl.InjectSoftwarePipeline",
    "tl.LowerTileOp",
    "tl.DecoupleTypeCast",
    "tl.LegalizeSafeMemoryAccess",
    "tl.LoopUnswitching",
    "tl.MergeIfStmt",
}


def _assert_no_span_loss(recorder: _SpanCoverageRecorder):
    prev_w = None
    for name, w, _t in recorder.rows:
        if prev_w is not None and name in _SPAN_SAFE_PASSES:
            assert w >= prev_w, f"pass {name} dropped spans: {prev_w} -> {w}"
        prev_w = w


def test_span_survives_lowering_cpu():
    """Vector add on the CPU backend: fixed passes must not drop spans."""
    func = _make_vector_add()
    store_line = _marker_line("span_marker_vadd_store")

    recorder = _lower_with_recorder(func, target="c")
    assert recorder.rows, "no passes recorded"
    _assert_no_span_loss(recorder)

    # The final IR still contains a BufferStore pointing at the user store line.
    # (Checked on the last pipeline row's module indirectly: re-lower and walk.)
    with tvm.target.Target("c"):
        artifact = tilelang.lower(_make_vector_add(), target="c")

    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tvm.tirx.BufferStore):
            loc = span_to_location(get_stmt_span(node))
            if loc is not None and loc[0] == __file__ and loc[1] == store_line:
                found = True

    for mod in (artifact.device_mod, artifact.host_mod):
        if mod is not None:
            for _, f in mod.functions.items():
                if isinstance(f, tvm.tirx.PrimFunc):
                    post_order_visit(f.body, visit)
    assert found, "lowered IR lost the span of the user's BufferStore line"


def test_span_survives_lowering_gemm_metal():
    """GEMM on Metal: tile-op lowering must stamp the expanded subtree."""
    pytest.importorskip("tilelang.metal")
    copy_a_line = _marker_line("span_marker_copy_a")
    copy_out_line = _marker_line("span_marker_copy_out")
    clear_line = _marker_line("span_marker_clear")

    recorder = _lower_with_recorder(_make_gemm(), target="metal")
    _assert_no_span_loss(recorder)

    with tvm.target.Target("metal"):
        artifact = tilelang.lower(_make_gemm(), target="metal")

    lines = set()

    def visit(node):
        if isinstance(node, tvm.tirx.Stmt):
            loc = span_to_location(get_stmt_span(node))
            if loc is not None and loc[0] == __file__:
                lines.add(loc[1])

    for mod in (artifact.device_mod, artifact.host_mod):
        if mod is not None:
            for _, f in mod.functions.items():
                if isinstance(f, tvm.tirx.PrimFunc):
                    post_order_visit(f.body, visit)

    # The copy/clear user statements are lowered into loops; their source lines
    # must still be present somewhere in the lowered IR.
    # NOTE: T.gemm on Metal is expanded by a Python macro, so its leaves carry
    # library-definition lines (eager-builder macro policy) rather than the
    # user call-site line — that is intentional and asserted by the language
    # span tests, so the gemm line itself is not checked here.
    assert copy_a_line in lines, "T.copy(A) line lost during lowering"
    assert copy_out_line in lines, "T.copy(C) line lost during lowering"
    assert clear_line in lines, "T.clear line lost during lowering"


if __name__ == "__main__":
    tilelang.testing.main()
