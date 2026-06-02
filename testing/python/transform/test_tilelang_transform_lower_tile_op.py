"""Tests for TileLang `LowerTileOp` copy annotations affecting cp.async sync."""

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _count_calls(func: tvm.tirx.PrimFunc):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(func.body, _visit)
    return counts


def test_lower_tile_op_respects_copy_annotation_for_pipeline_managed_cp_async():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_lower_tile_op_respects_copy_annotation_for_explicit_async_copy():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.async_copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_lower_tile_op_respects_parallel_loop_async_annotation_without_pipeline_context():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(
            16,
            annotations={"parallel_async_without_async_commit_wait": T.bool(True)},
        ):
            S[i] = A[i]
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_lower_tile_op_preserves_ragged_parallel_padding_guard():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def before(B: T.Tensor((8, 384), T.int32)):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", 256)
        for row, col in T.Parallel(8, 384):
            B[row, col] = T.if_then_else(col < 264, col, 2147483647)

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        assert "parallel_loop_requires_padding_guard" in mod.script(show_meta=True)
        tl.transform.LowerTileOp()(mod)


if __name__ == "__main__":
    tilelang.testing.main()
