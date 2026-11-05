"""Tests for TileLang `LowerTileOp` synchronization and copy lowering."""

from collections import Counter

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
import tvm_tirx as tvm
import tvm_ffi
from tvm_tirx import ir, tirx


def _evaluate_copy_index(expr, values):
    """Evaluate this test's bounded, nonnegative index grammar in Python."""
    if isinstance(expr, ir.IntImm):
        return int(expr.value)
    if isinstance(expr, ir.Var):
        return values[expr]
    left = _evaluate_copy_index(expr.a, values)
    right = _evaluate_copy_index(expr.b, values)
    if isinstance(expr, ir.prim.Add):
        return left + right
    if isinstance(expr, ir.prim.Mul):
        return left * right
    if isinstance(expr, ir.prim.FloorDiv):
        return left // right
    if isinstance(expr, ir.prim.FloorMod):
        return left % right
    raise TypeError(f"Unexpected copy index expression: {type(expr).__name__}")


def _call_annotations(call):
    assert call.attrs["schema_version"] == 1
    assert len(call.args) == call.attrs["operand_count"] + 1
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"])


def _attribute_fields(region):
    assert region.op.name == "tl.attr"
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])


def _count_calls(func: tirx.Function):
    return Counter(str(call.op.name) for call in collect_nodes(func, ir.Call) if isinstance(call.op, tvm.ir.Op))


def _find_loop_with_annotation(func: tirx.Function, annotation: str):
    loops = [loop for loop in collect_nodes(func, ir.For) if annotation in loop.annotations]
    assert len(loops) == 1, f"Expected one loop with {annotation}, got {len(loops)}"
    return loops[0]


@tilelang.testing.requires_cuda
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
    assert calls.get("tl.ptx_commit_group", 0) == 0
    assert calls.get("tl.ptx_wait_group", 0) == 0


@tilelang.testing.requires_cuda
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
    assert calls.get("tl.ptx_commit_group", 0) == 0
    assert calls.get("tl.ptx_wait_group", 0) == 0


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("enable_let_inline", [False, True])
@pytest.mark.parametrize(
    "cols,vector_size,dtype",
    [(1024, 8, "bfloat16"), (1024, 4, "bfloat16"), (1024, 4, "float32"), (3072, 8, "bfloat16")],
)
def test_lower_tile_op_async_copy_with_partitioned_layout(cols, vector_size, dtype, enable_let_inline):
    """Simplify inverse-layout indices enough to prove cp.async contiguity."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    layout = T.Fragment(
        (4, cols),
        forward_fn=lambda i, j: (
            i * 64 + (j // vector_size) % 64,
            j % vector_size + (j // (64 * vector_size)) * vector_size,
        ),
    )

    @T.prim_func
    def main(A: T.Tensor((4, cols), dtype), B: T.Tensor((4, cols), dtype)):
        with T.Kernel(1, threads=256):
            shared = T.alloc_shared((4, cols), dtype)
            T.async_copy(A, shared, loop_layout=layout)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.copy(shared, B)

    config = {tl.PassConfigKey.TL_SIMPLIFY: {tl.PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE: enable_let_inline}}
    with target, tvm.transform.PassContext(config=config):
        mod = tvm.IRModule.from_expr(main)
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    _assert_no_unexpected_free_vars(func)
    assert_call_count(func, op="tl.ptx_cp_async", count=1)
    # The native injector consumes a complete contiguous vector loop and emits
    # one legal-width transaction. Check its width and full ownership/address
    # mapping, including the synchronous branch for unaligned input pointers.
    call = collect_calls(func, op="tl.ptx_cp_async")[0]
    assert int(call.args[2]) == vector_size
    source_ptr, dest_ptr = call.args[1], call.args[0]
    for pointer, access in ((source_ptr, 1), (dest_ptr, 2)):
        assert pointer.op.name == "tl.access_ptr"
        assert isinstance(pointer.args[0], ir.TensorLoad)
        assert int(pointer.args[1]) == vector_size
        assert int(pointer.args[2]) == access
    source_load, dest_load = source_ptr.args[0], dest_ptr.args[0]
    assert source_load.source.same_as(func.params[0])
    assert str(dest_load.source.ty.scope) == "shared.dyn"
    assert source_load.ty == dest_load.ty
    assert vector_size * source_load.ty.dtype.bits // 8 in (4, 8, 16)

    copy_loops = [loop for loop in collect_nodes(func, ir.For) if collect_calls(loop, op="tl.ptx_cp_async")]
    assert len(copy_loops) == 1
    copy_loop = copy_loops[0]
    assert int(copy_loop.begin) == 0 and int(copy_loop.step) == 1
    assert int(copy_loop.end) == cols // (64 * vector_size)
    launches = [
        region
        for region in collect_nodes(func, ir.RegionStmt)
        if region.op.name == "tirx.launch_thread" and region.args[0].value == "threadIdx.x"
    ]
    assert len(launches) == 1 and int(launches[0].args[1]) == 256
    thread = launches[0].body_params[0]

    guards = [
        branch
        for branch in collect_nodes(copy_loop, ir.If)
        if branch.else_case is not None and collect_calls(branch.then_case, op="tl.ptx_cp_async")
    ]
    assert len(guards) == 1
    guard = guards[0]
    assert not collect_nodes(guard.then_case, ir.For)
    assert not collect_calls(guard.else_case, op="tl.ptx_cp_async")
    fallback_loops = collect_nodes(guard.else_case, ir.For)
    assert len(fallback_loops) == 1
    fallback = fallback_loops[0]
    assert fallback.kind == ir.ForKind.VECTORIZED
    assert int(fallback.begin) == 0 and int(fallback.end) == vector_size and int(fallback.step) == 1
    stores = collect_nodes(fallback, ir.TensorStore)
    assert len(stores) == 1 and isinstance(stores[0].value, ir.TensorLoad)
    assert stores[0].dest.same_as(dest_load.source)
    assert stores[0].value.source.same_as(source_load.source)

    covered = set()
    for thread_value in range(256):
        for iteration in range(int(copy_loop.end)):
            values = {thread: thread_value, copy_loop.var: iteration}
            expected_base = (thread_value // 64, iteration * 64 * vector_size + thread_value % 64 * vector_size)
            assert tuple(_evaluate_copy_index(index, values) for index in source_load.indices) == expected_base
            assert tuple(_evaluate_copy_index(index, values) for index in dest_load.indices) == expected_base
            for lane in range(vector_size):
                logical = (expected_base[0], expected_base[1] + lane)
                assert logical not in covered
                covered.add(logical)
                i, j = logical
                assert i * 64 + (j // vector_size) % 64 == thread_value
                assert j % vector_size + (j // (64 * vector_size)) * vector_size == iteration * vector_size + lane
                fallback_values = {**values, fallback.var: lane}
                assert tuple(_evaluate_copy_index(index, fallback_values) for index in stores[0].indices) == logical
                assert tuple(_evaluate_copy_index(index, fallback_values) for index in stores[0].value.indices) == logical
    assert covered == {(i, j) for i in range(4) for j in range(cols)}


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipelined_tma_copy_compiler_generated_barrier_uses_emitted_loop_epoch(num_stages):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})

    @T.prim_func
    def before(
        A: T.Tensor((9, 16), T.float16),
        B: T.Tensor((9, 16), T.float16),
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16,), T.float16)
            for i in T.serial(
                3,
                9,
                annotations={
                    "software_pipeline_stage": [0, num_stages - 1],
                    "software_pipeline_order": [0, 1],
                    "tl_pipelined_num_stages": T.int32(num_stages),
                },
            ):
                T.copy(A[i, 0:16], shared, prefer_instruction="tma")
                T.copy(shared, B[i, 0:16], prefer_instruction="sync")

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    with target:
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        copy_calls = collect_calls(mod["main"], op="tl.tileop.copy")
        assert all("tl.pipeline_mbar_phase_expr" not in _call_annotations(call) for call in copy_calls)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    waits = collect_calls(func, op="tl.mbarrier_wait_parity")
    loop = _find_loop_with_annotation(func, "tl_pipelined_num_stages")
    loop_waits = collect_calls(loop, op="tl.mbarrier_wait_parity")

    assert len(waits) == num_stages
    assert len(loop_waits) == 1
    barrier_indices = [int(wait.args[0].indices[0]) for wait in waits]
    assert sorted(barrier_indices) == list(range(num_stages))
    simplified_phases = [tvm.sym.Analyzer().simplify(wait.args[1]) for wait in waits]
    constant_phases = [int(phase) for phase in simplified_phases if isinstance(phase, ir.IntImm)]
    assert constant_phases == [0] * (num_stages - 1)
    expected = (loop.var - loop.begin) % 2
    assert tvm.sym.Analyzer().can_prove_equal(loop_waits[0].args[1], expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipelined_tma_copy_explicit_mbar_uses_logical_epoch(num_stages):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})

    @T.prim_func
    def before(
        A: T.Tensor((9, 16), T.float16),
        B: T.Tensor((9, 16), T.float16),
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.serial(
                3,
                9,
                annotations={
                    "software_pipeline_stage": [0, num_stages - 1],
                    "software_pipeline_order": [0, 1],
                    "tl_pipelined_num_stages": T.int32(num_stages),
                },
            ):
                T.copy(
                    A[i, 0:16],
                    shared,
                    prefer_instruction="tma",
                    annotations={"barrier": mbar[0]},
                )
                T.copy(shared, B[i, 0:16], prefer_instruction="sync")

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    with target:
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    func = mod["main"]
    waits = collect_calls(func, op="tl.mbarrier_wait_parity")
    loop = _find_loop_with_annotation(func, "tl_pipelined_num_stages")
    loop_waits = collect_calls(loop, op="tl.mbarrier_wait_parity")

    assert len(waits) == num_stages
    assert len(loop_waits) == 1
    barrier_indices = [int(wait.args[0].indices[0]) for wait in waits]
    assert barrier_indices == [0] * num_stages
    simplified_phases = [tvm.sym.Analyzer().simplify(wait.args[1]) for wait in waits]
    constant_phases = [int(phase) for phase in simplified_phases if isinstance(phase, ir.IntImm)]
    assert constant_phases == [i % 2 for i in range(num_stages - 1)]
    logical_epoch = loop.var - loop.begin + num_stages - 1
    assert tvm.sym.Analyzer().can_prove_equal(loop_waits[0].args[1], logical_epoch % 2)


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
    assert calls.get("tl.ptx_commit_group", 0) == 0
    assert calls.get("tl.ptx_wait_group", 0) == 0


@tilelang.testing.requires_cuda
def test_lower_tile_op_rejects_shifted_modulo_fragment_index():
    """Reject #2948 instead of silently dropping a fragment index rotation."""
    size = 128
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def before(
        A: T.Tensor((size,), T.int32),
        B: T.Tensor((size,), T.int32),
    ):
        with T.Kernel(1, threads=size):
            fragment = T.alloc_fragment((size,), T.int32)
            T.copy(A, fragment)
            for i in T.Parallel(size):
                B[i] = fragment[(i + 1) % size]

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        # Native iter-map validation may reject this mapping before the
        # explicit round-trip check. Either path must refuse to lower it.
        with pytest.raises(Exception, match="non-round-tripping inverse|Mapped indices are not independent"):
            tl.transform.LowerTileOp()(mod)


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
    return collect_nodes(node, ir.Var)


def _collect_var_names(func: tirx.Function):
    return {_var_name(var) for var in _vars_of(func.body)}


def _for_predicate_annotations(func: tirx.Function) -> list:
    """Read static and typed operand carriers; keep actual variable identities."""
    predicates = [
        loop.annotations["parallel_loop_predicate"] for loop in collect_nodes(func, ir.For) if "parallel_loop_predicate" in loop.annotations
    ]
    for region in collect_nodes(func, ir.RegionStmt):
        if region.op.name != "tl.attr":
            continue
        fields = _attribute_fields(region)
        if fields["key"] == "tl.loop.annotations" and "parallel_loop_predicate" in fields["value"]:
            predicates.append(fields["value"]["parallel_loop_predicate"])
    return predicates


def _thread_extent_vars(func: tirx.Function) -> list:
    """Native launch_thread body parameters define real thread identities."""
    return [
        region.body_params[0]
        for region in collect_nodes(func, ir.RegionStmt)
        if region.op.name == "tirx.launch_thread" and region.args[0].value == "threadIdx.x"
    ]


def _assert_no_unexpected_free_vars(func: tirx.Function):
    # Native TensorVar parameters own their layout/shape, and data_ptr is an
    # explicit expression. No old buffer_map data-variable exclusion is needed.
    unexpected = tirx.analysis.undefined_vars(func.body, func.params)
    assert not unexpected, f"unexpected free variables: {unexpected}"


def _cpu_target(with_host: bool = False) -> tvm.target.Target:
    host = tvm.target.Target("llvm") if with_host else None
    return tvm.target.Target("c", host) if with_host else tvm.target.Target("c")


@pytest.mark.parametrize("target_kind", ["cuda", "hip"])
@pytest.mark.parametrize("rows,cols,threads,thread_start", [(1, 2, 2, 0), (2, 8, 4, 0), (3, 5, 8, 0), (2, 8, 8, 4)])
def test_lower_tile_op_parallel_let_indices(target_kind, rows, cols, threads, thread_start):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"} if target_kind == "cuda" else {"kind": "hip", "mcpu": "gfx1100"})
    size = 2 * rows * cols + 8

    @T.prim_func
    def main(A: T.Tensor((size,), T.int32), B: T.Tensor((size,), T.int32)):
        with T.Kernel(2, threads=threads) as bx:
            if T.get_thread_binding() >= thread_start:
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = A[idx] * 3 + idx
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = B[idx] + 5

    config = {tl.PassConfigKey.TL_SIMPLIFY: {tl.PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE: False}}
    with target, tvm.transform.PassContext(config=config):
        mod = tvm.IRModule.from_expr(main)
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    _assert_no_unexpected_free_vars(mod["main"])


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
    assert device_param_names == {"flag_ptr", "out_ptr"}, f"unexpected device ABI: {device_param_names}"
    assert all(isinstance(param.ty, ir.PtrType) for param in device_func.params)
    declarations = [
        bind
        for bind in collect_nodes(device_func, ir.Bind)
        if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.decl_tensor" and bind.var.name in {"flag", "out"}
    ]
    assert len(declarations) == 2
    for bind in declarations:
        pointer = next(param for param in device_func.params if param.name == bind.var.name + "_ptr")
        assert bind.value.args[0].same_as(pointer)
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
