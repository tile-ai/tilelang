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


def _var_name(var) -> str:
    if hasattr(var, "name_hint"):
        return var.name_hint
    if hasattr(var, "name"):
        return var.name
    return str(var).split(":")[0].strip()


def _vars_of(node) -> list:
    """Collect all Var objects referenced by an expression or statement."""
    vars_found = []

    def _visit(n):
        if isinstance(n, tvm.tirx.Var):
            vars_found.append(n)

    post_order_visit(node, _visit)
    return vars_found


def _collect_var_names(func: tvm.tirx.PrimFunc):
    return {_var_name(var) for var in _vars_of(func.body)}


def _for_predicate_annotations(func: tvm.tirx.PrimFunc) -> list:
    """Every parallel_loop_predicate annotation expression in the function.

    Generic statement visitors do not traverse annotation payloads, so the
    annotation expressions are read out explicitly here.
    """
    predicates = []

    def _visit(node):
        if isinstance(node, tvm.tirx.For):
            predicate = node.annotations.get("parallel_loop_predicate", None)
            if predicate is not None:
                predicates.append(predicate)

    post_order_visit(func.body, _visit)
    return predicates


def _thread_extent_vars(func: tvm.tirx.PrimFunc) -> list:
    """Vars bound by threadIdx.x thread_extent AttrStmts (real thread bindings)."""
    vars_found = []

    def _visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x":
            vars_found.append(node.node.var)

    post_order_visit(func.body, _visit)
    return vars_found


def _assert_no_unexpected_free_vars(func: tvm.tirx.PrimFunc):
    """Every free var must be a buffer_map data var; nothing synthetic may leak.

    Right after LowerTileOp/SplitHostDevice the buffer data vars still appear
    "undefined" to var-use analysis (they are declared via match_buffer and
    lowered later), so the assertion filters them out — by object identity,
    not by name — and checks that no *other* free variable (e.g. a synthetic
    thread placeholder) survives.
    """
    buffer_data_vars = [buffer.data for buffer in func.buffer_map.values()]
    unexpected = [
        var
        for var in tvm.tirx.analysis.undefined_vars(func.body, func.params)
        if not any(var.same_as(data_var) for data_var in buffer_data_vars)
    ]
    assert not unexpected, f"unexpected free variables: {unexpected}"


def _cpu_target(with_host: bool = False) -> tvm.target.Target:
    host = tvm.target.Target("llvm") if with_host else None
    return tvm.target.Target("c", host) if with_host else tvm.target.Target("c")


def _cpu_while_kernel_module(with_host: bool = False):
    """The while + fragment kernel from issue #2202, on a CPU `c` target.

    Returns (module, target); run subsequent passes inside `with target:` —
    parallel-loop lowering queries Target::Current().
    """

    @T.prim_func
    def main(flag: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]

    mod = tvm.IRModule.from_expr(main)
    target = _cpu_target(with_host)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=False)(mod)
    return mod, target


def _cpu_parallel_kernel_module():
    """A CPU `c` target module with T.Parallel loops and a fragment buffer.

    Unlike the while repro, this exercises the thread-oriented helper paths
    directly: `ParallelOpNode::GetPredicate` during LayoutInference and
    `LowerParallelLoop`/`LowerArgs.thread_index` during LowerTileOp.

    Returns (module, target); run subsequent passes inside `with target:`.
    """

    @T.prim_func
    def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((16,), "int32")
            for i in T.Parallel(16):
                frag[i] = A[i]
            for i in T.Parallel(16):
                B[i] = frag[i]

    mod = tvm.IRModule.from_expr(main)
    target = _cpu_target()
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=False)(mod)
    return mod, target


def test_layout_inference_cpu_parallel_predicate_is_ground():
    """Producer-level regression for issue #2226 at the LayoutInference boundary.

    With a real thread binding absent, LayoutInference must materialize
    parallel-loop predicates with the constant 0 thread index. The historical
    LowerTileOp canonicalizer would rewrite a leaked `v_thread` only at the
    end of LowerTileOp, so checking right after LayoutInference is what
    distinguishes a producer-level fix from the old compatibility cleanup:
    the old code would leave `v_thread >= 0 && v_thread < 1` in the
    annotation here.
    """
    mod, target = _cpu_parallel_kernel_module()
    with target:
        mod = tl.transform.LayoutInference()(mod)
    func = mod["main"]

    # CPU predicates substitute the constant 0 thread index, so they simplify
    # away; any surviving annotation expression must be ground (Var-free).
    for predicate in _for_predicate_annotations(func):
        assert _vars_of(predicate) == [], f"CPU predicate must be ground, got: {predicate}"
    assert "v_thread" not in _collect_var_names(func)


def test_layout_inference_metal_parallel_predicate_uses_real_thread_var():
    """Targets with a real thread binding keep the bound threadIdx.x in predicates.

    Companion to the CPU boundary test: the logical thread index must remain
    the real threadIdx.x Var (same object as the thread_extent binding) when
    one exists.
    """

    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(8):
                B[i] = A[i] * 2.0

    mod = tvm.IRModule.from_expr(main)
    target = tvm.target.Target("metal")
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    with target:
        mod = tl.transform.LayoutInference()(mod)
    func = mod["main"]

    thread_vars = _thread_extent_vars(func)
    assert len(thread_vars) == 1, "expected exactly one threadIdx.x binding"
    predicates = _for_predicate_annotations(func)
    assert predicates, "expected a parallel_loop_predicate annotation"
    for predicate in predicates:
        predicate_vars = _vars_of(predicate)
        assert predicate_vars, f"predicate should reference threadIdx.x: {predicate}"
        for var in predicate_vars:
            assert any(var.same_as(tv) for tv in thread_vars), f"predicate variable {var} is not the bound threadIdx.x"


def test_lower_tile_op_cpu_no_synthetic_thread_var():
    """CPU lowering must not materialize a synthetic thread placeholder.

    Historical end-to-end regression for https://github.com/tile-ai/tilelang/issues/2226
    using the issue #2202 while + fragment kernel.
    """
    mod, target = _cpu_while_kernel_module()
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    assert "v_thread" not in _collect_var_names(func)
    _assert_no_unexpected_free_vars(func)


def test_lower_tile_op_cpu_parallel_loop_no_synthetic_thread_var():
    """CPU parallel-loop lowering receives constant 0 as the thread index (#2226)."""
    mod, target = _cpu_parallel_kernel_module()
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    assert "v_thread" not in _collect_var_names(func)
    _assert_no_unexpected_free_vars(func)


def test_split_host_device_cpu_device_abi_is_clean():
    """SplitHostDevice must never see a synthetic CPU thread placeholder.

    See https://github.com/tile-ai/tilelang/issues/2226: a leaked `v_thread`
    would be picked up by SplitHostDevice's use-def analysis and become an
    unexpected device ABI parameter. Uses a device+host target pair (as the
    production lowering does) so AnnotateDeviceRegions/SplitHostDevice really
    extract a device function, then compares the exact device signature.
    """
    mod, target = _cpu_while_kernel_module(with_host=True)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        mod = tl.transform.AnnotateDeviceRegions()(mod)
        mod = tl.transform.SplitHostDevice()(mod)

    host_func = None
    device_func = None
    for gvar, func in mod.functions.items():
        if "kernel" in gvar.name_hint:
            device_func = func
        else:
            host_func = func

    assert host_func is not None, "expected the original host function"
    assert device_func is not None, "expected SplitHostDevice to extract a device function"

    # The device signature is exactly the two buffer parameters the kernel
    # reads/writes — any leaked synthetic scalar would show up as an extra
    # parameter here.
    device_param_names = {_var_name(param) for param in device_func.params}
    assert device_param_names == {"flag", "out"}, f"unexpected device ABI: {device_param_names}"
    _assert_no_unexpected_free_vars(device_func)


def test_cpu_legitimate_v_thread_param_is_preserved():
    """A user variable named `v_thread` must survive lowering untouched.

    The old name-based canonicalizer rewrote *any* same-named variable to 0
    (see issue #2226). A legitimate scalar parameter with that name must keep
    its identity and its uses through LayoutInference and LowerTileOp.
    """

    @T.prim_func
    def main(v_thread: T.int32, out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = v_thread
            out[0] = state[0]

    original_param = main.params[0]
    mod = tvm.IRModule.from_expr(main)
    target = _cpu_target()
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=False)(mod)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    matching_params = [p for p in func.params if _var_name(p) == "v_thread"]
    assert len(matching_params) == 1, "the legitimate v_thread parameter must remain"
    assert matching_params[0].same_as(original_param)
    body_refs = [var for var in _vars_of(func.body) if _var_name(var) == "v_thread"]
    assert body_refs, "the body must still read the v_thread parameter"
    assert all(var.same_as(original_param) for var in body_refs)


if __name__ == "__main__":
    tilelang.testing.main()
