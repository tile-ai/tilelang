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


def _cpu_while_kernel_module():
    """The while + fragment kernel from issue #2202, on a CPU `c` target."""

    @T.prim_func
    def main(flag: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]

    mod = tvm.IRModule.from_expr(main)
    target = tvm.target.Target("c")
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=False)(mod)
    return mod


def _collect_var_names(func: tvm.tirx.PrimFunc):
    names = set()

    def _visit(node):
        if isinstance(node, tvm.tirx.Var):
            names.add(_var_name(node))

    post_order_visit(func.body, _visit)
    return names


def _var_name(var) -> str:
    if hasattr(var, "name_hint"):
        return var.name_hint
    if hasattr(var, "name"):
        return var.name
    return str(var).split(":")[0].strip()


def _assert_no_unexpected_free_vars(func: tvm.tirx.PrimFunc):
    """Every free var must come from the buffer_map; nothing synthetic may leak.

    Right after LowerTileOp/SplitHostDevice the buffer data vars still appear
    "undefined" to var-use analysis (they are declared via match_buffer and
    lowered later), so the assertion filters them out and checks that no
    *other* free variable — e.g. a synthetic thread placeholder — survives.
    """
    buffer_data_names = {_var_name(buffer.data) for buffer in func.buffer_map.values()}
    unexpected = [var for var in tvm.tirx.analysis.undefined_vars(func.body, func.params) if _var_name(var) not in buffer_data_names]
    assert not unexpected, f"unexpected free variables: {unexpected}"


def test_lower_tile_op_cpu_no_synthetic_thread_var():
    """CPU lowering must not materialize a synthetic thread placeholder.

    Regression test for https://github.com/tile-ai/tilelang/issues/2226.
    LayoutInference/LowerTileOp used to share a synthetic `v_thread` fallback
    Var that could escape into later passes as a free variable. Thread-oriented
    helpers must now receive constant 0 on CPU, so no synthetic thread Var may
    appear in the lowered function or its annotations.
    """
    mod = _cpu_while_kernel_module()
    mod = tl.transform.LayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    assert "v_thread" not in _collect_var_names(func)
    _assert_no_unexpected_free_vars(func)


def test_lower_tile_op_cpu_split_host_device_abi_is_clean():
    """SplitHostDevice must never see a synthetic CPU thread placeholder.

    See https://github.com/tile-ai/tilelang/issues/2226: a leaked `v_thread`
    would be picked up by SplitHostDevice's use-def analysis and become an
    unexpected device ABI parameter.
    """
    mod = _cpu_while_kernel_module()
    mod = tl.transform.LayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tl.transform.AnnotateDeviceRegions()(mod)
    mod = tl.transform.SplitHostDevice()(mod)

    assert len(mod.functions) >= 1
    for gvar, func in mod.functions.items():
        param_names = {_var_name(param) for param in func.params}
        assert "v_thread" not in param_names, f"{gvar} gained a synthetic thread parameter"
        assert "v_thread" not in _collect_var_names(func)
        _assert_no_unexpected_free_vars(func)


if __name__ == "__main__":
    tilelang.testing.main()
