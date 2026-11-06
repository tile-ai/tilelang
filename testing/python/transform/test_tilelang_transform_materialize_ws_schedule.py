"""Unit tests for the MaterializeWSSchedule transform.

Each test builds its input in the form the pass consumes directly: a
static-shape ``@T.prim_func`` kernel (traced at decoration — no torch, no
GPU) prepared with exactly the two passes that define the input contract,
BindTarget and MaterializeKernelLaunch. The target is pinned so the tests
are independent of the machine and of the production pipeline.
"""

from tilelang.language import _tensor_spec

import pytest

import tilelang
import tilelang.testing
import tilelang.cuda.op  # noqa: F401  # Register the implementations used by direct lowering below.
from tilelang import language as T

from tilelang.tileop import Gemm, GemmBlockScaled
import tvm_tirx
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

_TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _materialize(func):
    """Prepare `func` (BindTarget + MaterializeKernelLaunch — the pass's
    input contract) and apply MaterializeWSSchedule; returns the
    transformed native Function."""
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm_tirx.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    transformed = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    _assert_no_ws_annotations(transformed)
    _assert_no_undefined_vars(transformed)
    return transformed


def _annotations(node):
    if isinstance(node, ir.For):
        return node.annotations
    if not isinstance(node, (ir.Call, ir.RegionStmt)) or not isinstance(node.attrs, ir.DictAttrs):
        return {}
    if "annotation_schema" not in node.attrs:
        return node.attrs.dict
    count = int(node.attrs.get("operand_count", 0))
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[count], node.attrs["annotation_schema"])


def _operands(call):
    count = int(call.attrs.get("operand_count", len(call.args))) if call.attrs else len(call.args)
    return call.args[:count]


def _attr_key(region):
    return _annotations(region).get("key")


def _only(body):
    while isinstance(body, ir.SeqStmt) and len(body.seq) == 1:
        body = body.seq[0]
    return body


def _extent(loop):
    return tvm_tirx.sym.Analyzer().simplify(loop.end - loop.begin)


def _assert_no_ws_annotations(func):
    markers = {"tl.ws_schedule", "tl.ws_op_id", "tl.ws_next_op_id"}
    assert markers.isdisjoint(func.attrs.keys() if func.attrs else ())
    for node in collect_nodes(func, tvm_ffi.Object):
        assert markers.isdisjoint(_annotations(node))
        if isinstance(node, ir.RegionStmt):
            assert _attr_key(node) not in markers


def _buffer(root, name):
    buffers = [
        bind.var
        for bind in collect_nodes(root, ir.Bind)
        if isinstance(bind.var.ty, tirx.TensorType)
        and isinstance(bind.value, ir.Call)
        and bind.value.op.same_as(ir.Op.get("tirx.alloc_tensor"))
    ]
    if isinstance(root, tirx.Function):
        buffers.extend(param for param in root.params if isinstance(param.ty, tirx.TensorType))
    matches = [buf for buf in buffers if buf.name == name]
    assert len(matches) == 1, f"Expected one tensor {name}, got {len(matches)}"
    return matches[0]


def _barrier(func, name, counts=None):
    tensor = _buffer(func, name)
    assert str(tensor.ty.scope) == "shared.barrier"
    initializers = []
    for node in collect_nodes(func, ir.Call):
        annotations = _annotations(node)
        if "barrier_init" in annotations and tensor in annotations["barrier_init"]:
            initializers.append(annotations["barrier_init"][tensor])
    assert len(initializers) == 1, f"Missing or duplicated barrier initialization for {name}"
    values = initializers[0]
    assert len(tensor.ty.shape) == 1 and int(tensor.ty.shape[0]) == len(values)
    assert all(isinstance(value, ir.IntImm) and str(value.ty) == "int32" for value in values)
    if counts is not None:
        assert [int(value) for value in values] == counts
    return tensor


def _barrier_calls(root, op, tensor):
    return [
        call
        for call in collect_calls(root, op=op)
        if call.args and isinstance(call.args[0], ir.TensorLoad) and call.args[0].source.same_as(tensor)
    ]


def _nreg_args(root):
    calls = collect_calls(root, op="tl.set_max_nreg")
    assert all(len(call.args) == 2 and all(isinstance(arg, ir.IntImm) for arg in call.args) for call in calls)
    return [tuple(int(arg) for arg in call.args) for call in calls]


def _role_branches(func):
    (launch,) = [
        region
        for region in collect_nodes(func, ir.RegionStmt)
        if region.op.name == "tirx.launch_thread" and region.args[0].value == "threadIdx.x"
    ]
    (custom,) = [region for region in collect_nodes(launch.body, ir.RegionStmt) if _attr_key(region) == "kCustomWarpSpecialization"]
    assert int(_annotations(custom)["value"]) == 1
    branches = []
    lo = 0
    body = _only(custom.body)
    while isinstance(body, ir.If):
        condition = body.condition
        assert isinstance(condition, prim.LT) and condition.a.same_as(launch.body_params[0])
        assert isinstance(condition.b, ir.IntImm)
        hi = int(condition.b)
        assert lo < hi <= int(launch.args[1])
        branches.append((lo, hi, body.then_case))
        lo, body = hi, body.else_case
        assert body is not None
        body = _only(body)
    branches.append((lo, int(launch.args[1]), body))
    return branches


def _role_body(func, first_thread):
    (body,) = [body for lo, _, body in _role_branches(func) if lo == first_thread]
    return body


def _assert_stmt_order(root, *statements):
    """Require direct siblings in one lexical scope, in execution order."""
    for seq in collect_nodes(root, ir.SeqStmt):
        positions = [next((i for i, child in enumerate(seq.seq) if child.same_as(stmt)), -1) for stmt in statements]
        if positions and positions[0] >= 0 and all(a < b for a, b in zip(positions, positions[1:])):
            return
    raise AssertionError("Expected ordered siblings in one SeqStmt")


def _call_stmt(root, call):
    (stmt,) = [stmt for stmt in collect_nodes(root, ir.Evaluate) if stmt.value.same_as(call)]
    return stmt


def _assert_versioned_access(root, tensor, version):
    # A native TensorRegion holds its start coordinates directly; load the
    # start only for these index assertions, without inventing old Buffer IR.
    loads = [load for load in collect_nodes(root, ir.TensorLoad) if load.source.same_as(tensor)]
    loads.extend(
        ir.TensorLoad(region.source, [dim.begin for dim in region.region])
        for region in collect_nodes(root, ir.TensorRegion)
        if region.source.same_as(tensor)
    )
    assert loads, f"Missing accesses to {tensor.name}"
    for load in loads:
        assert len(load.indices) == len(load.source.ty.shape)
        tvm_tirx.ir.assert_structural_equal(load.indices[0], version)
    return loads


def _copy_region_loads(call):
    operands = _operands(call)
    assert len(operands) == 2
    assert all(isinstance(region, ir.TensorRegion) for region in operands)
    return [ir.TensorLoad(region.source, [dim.begin for dim in region.region]) for region in operands]


def _assert_counter_updates(root, tensor, updates):
    assert isinstance(root, ir.SeqStmt)
    stores = [store for store in collect_nodes(root, ir.TensorStore) if store.dest.same_as(tensor)]
    initializers = [store for store in stores if isinstance(store.value, ir.IntImm) and int(store.value) == 0]
    assert len(initializers) == 1 and len(stores) == updates + 1
    initializer = initializers[0]
    assert initializer.value.ty == tensor.ty.dtype
    (position,) = [i for i, stmt in enumerate(root.seq) if stmt.same_as(initializer)]
    for stmt in root.seq[:position]:
        assert not any(load.source.same_as(tensor) for load in collect_nodes(stmt, ir.TensorLoad))
    expected = ir.TensorLoad(tensor, [0]) + 1
    for store in stores:
        assert len(store.indices) == 1 and int(store.indices[0]) == 0
        if not store.same_as(initializer):
            tvm_tirx.ir.assert_structural_equal(store.value, expected)
    return initializer


def _assert_phase_counter(func, role, *, pipeline="buf", updates=1):
    counter = _buffer(role, pipeline + "_phase")
    assert str(counter.ty.scope) == "local" and str(counter.ty.dtype) == "int32" and [int(dim) for dim in counter.ty.shape] == [1]
    _assert_counter_updates(role, counter, updates)
    barriers = [_barrier(func, pipeline + suffix) for suffix in ("_full", "_empty")]
    waits = [call for barrier in barriers for call in _barrier_calls(role, "tl.mbarrier_wait_parity", barrier)]
    assert waits
    for call in waits:
        tvm_tirx.ir.assert_structural_equal(call.args[0].indices[0], ir.TensorLoad(counter, [0]) % int(call.args[0].source.ty.shape[0]))


def _assert_no_undefined_vars(func):
    assert not tirx.analysis.undefined_vars(func)
    assert tirx.analysis.check_well_formed(func)


def _smem_pipeline_kernel(
    *,
    depth=2,
    producer_stage=0,
    consumer_wait_stage=0,
    consumer_release_stage=0,
    schedule_edit=None,
    kernel_edit=None,
):
    """A minimal two-role smem pipeline: one TMA warp copies A tile-by-tile
    into a multi-versioned shared buffer; one consumer warpgroup copies it
    back out to global memory through a fragment.

    ``schedule_edit(bodies)`` may rewrite the per-role instruction lists and
    ``kernel_edit`` selects alternative kernel-body variants used by the
    error-path tests.
    """

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            if kernel_edit == "metadata_attrs":
                T.use_swizzle(10)
                T.annotate_min_blocks_per_sm(2)

            bodies = {
                "Producer": [
                    T.WSSync.producer_acquire("buf", stage=producer_stage),
                    "copy_in",
                    T.WSSync.producer_commit("buf", stage=producer_stage),
                ],
                "Consumer": [
                    T.WSSync.consumer_wait("buf", stage=consumer_wait_stage),
                    "copy_frag",
                    T.WSSync.consumer_release("buf", stage=consumer_release_stage),
                    "copy_out",
                ],
            }
            if schedule_edit is not None:
                schedule_edit(bodies)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf", [A_shared], depth=depth),
                    ],
                    scopes=[
                        T.WSScope("loop_k", bodies),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=depth, annotations={T.WSID: "loop_k"}):
                if kernel_edit == "unsupported_tma_preference":
                    T.copy(
                        A[i * 64, 0],
                        A_shared,
                        disable_tma=True,
                        prefer_instruction="tma",
                        annotations={T.WSID: "copy_in"},
                    )
                else:
                    T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                if kernel_edit == "unannotated_consumer":
                    T.copy(A_shared, A_frag)
                else:
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})
                if kernel_edit == "unannotated_global_store":
                    B[0, 0] = T.float16(0.0)

    return kernel


@tilelang.testing.requires_cuda
def test_basic_two_role_transform():
    func = _materialize(_smem_pipeline_kernel())
    # Check the actual thread partition and each branch's register budget.
    branches = _role_branches(func)
    assert [(lo, hi) for lo, hi, _ in branches] == [(0, 32), (32, 128), (128, 256)]
    assert [_nreg_args(body) for _, _, body in branches] == [[(40, 0)], [(40, 0)], [(224, 1)]]


@tilelang.testing.requires_cuda
def test_scalar_op_annotations_preserved():
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=224)],
                    pipelines=[],
                    scopes=[
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Worker": [
                                    "copy",
                                    "atomic_max",
                                    "atomic_min",
                                    "atomic_add",
                                ]
                            },
                        )
                    ],
                )
            )
            T.copy(A[0], B[0], annotations={T.WSID: "copy"})
            T.atomic_max(B[0], A[0], annotations={T.WSID: "atomic_max"})
            T.atomic_min(B[0], A[0], annotations={T.WSID: "atomic_min"})
            T.atomic_add(B[0], A[0], annotations={T.WSID: "atomic_add"})

    func = _materialize(kernel)
    for op in ("tl.tileop.copy", "tl.atomic_max_elem_op", "tl.atomic_min_elem_op", "tl.atomic_add_elem_op"):
        assert_call_count(func, op=op, count=1)


@tilelang.testing.requires_cuda
def test_buffer_multi_versioning():
    func = _materialize(_smem_pipeline_kernel(depth=3))
    shared = _buffer(func, "A_shared")
    assert [int(dim) for dim in shared.ty.shape] == [3, 64, 64]
    for first_thread in (0, 128):
        (loop,) = collect_nodes(_role_body(func, first_thread), ir.For)
        loads = _assert_versioned_access(loop.body, shared, loop.var % 3)
        for load in loads:
            assert [int(index) for index in load.indices[1:]] == [0, 0]


@tilelang.testing.requires_cuda
def test_barrier_counts_tma_and_threads():
    func = _materialize(_smem_pipeline_kernel())
    # TMA's elected arrive and all 128 synchronous consumer threads.
    _barrier(func, "buf_full", [1, 1])
    _barrier(func, "buf_empty", [128, 128])


@tilelang.testing.requires_cuda
def test_tma_copy_conversion_carries_barrier():
    func = _materialize(_smem_pipeline_kernel())
    producer = _role_body(func, 0)
    (loop,) = collect_nodes(producer, ir.For)
    (copy,) = collect_calls(producer, op="tl.tileop.tma_copy")
    full = _barrier(func, "buf_full")
    tvm_tirx.ir.assert_structural_equal(_annotations(copy)["barrier"], ir.TensorLoad(full, [loop.var % 2]))
    assert type(_annotations(copy)["emit_arrive"]) is int and _annotations(copy)["emit_arrive"] == 1
    assert not _barrier_calls(func, "tl.ptx_arrive_barrier", full)


@tilelang.testing.requires_cuda
def test_guarded_tma_copy_keeps_unconditional_syncs():
    """A source-guarded TMA copy skips only the copy itself: the
    acquire/commit and the per-thread arrives stay unconditional, so the
    pipeline cycles every iteration (a skipped copy simply contributes no
    transaction bytes)."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                if i < 3:
                    T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    full = _barrier(func, "buf_full", [32, 32])
    producer = _role_body(func, 0)
    (loop,) = collect_nodes(producer, ir.For)
    (copy,) = collect_calls(loop.body, op="tl.tileop.tma_copy")
    assert "emit_arrive" not in _annotations(copy)
    (guard,) = collect_nodes(loop.body, ir.If)
    tvm_tirx.ir.assert_structural_equal(guard.condition, loop.var < 3)
    assert_call_count(guard.then_case, op="tl.tileop.tma_copy", count=1)
    (wait,) = collect_calls(loop.body, op="tl.mbarrier_wait_parity")
    (arrive,) = _barrier_calls(loop.body, "tl.ptx_arrive_barrier", full)
    _assert_stmt_order(loop.body, _call_stmt(loop.body, wait), guard, _call_stmt(loop.body, arrive))


@tilelang.testing.requires_cuda
def test_cp_async_with_sync_work_keeps_both_arrives():
    """The deferred cp.async arrive publishes only the thread's own
    cp.async data, so synchronous work on the same side keeps its plain
    release arrive — both are emitted and both are counted."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "pad_in",
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.fill(A_shared, 0, annotations={T.WSID: "pad_in"})
                T.async_copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    # One deferred and one plain arrive per producer thread (32 + 32).
    full = _barrier(func, "buf_full", [64, 64])
    producer = _role_body(func, 0)
    assert len(_barrier_calls(producer, "tl.ptx_cp_async_barrier_noinc", full)) == 1
    assert len(_barrier_calls(producer, "tl.ptx_arrive_barrier", full)) == 1


@tilelang.testing.requires_cuda
def test_prefer_instruction_cp_async_selects_cp_async_atom():
    """``T.copy(..., prefer_instruction="cp_async")`` classifies as a
    cp.async op like ``T.async_copy``: the copy loses its implicit
    commit/wait and the commit entry carries the deferred arrive."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(
                    A[i * 64, 0],
                    A_shared,
                    prefer_instruction="cp_async",
                    annotations={T.WSID: "copy_in"},
                )
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    producer = _role_body(func, 0)
    (copy,) = collect_calls(producer, op="tl.tileop.copy")
    assert int(_annotations(copy)["no_implicit_async_commit_wait"]) == 1
    full = _barrier(func, "buf_full", [32, 32])
    assert len(_barrier_calls(producer, "tl.ptx_cp_async_barrier_noinc", full)) == 1
    assert not _barrier_calls(func, "tl.ptx_arrive_barrier", full)


@tilelang.testing.requires_cuda
def test_unsupported_preferred_copy_rejected():
    with pytest.raises(Exception, match='prefer_instruction="tma" conflicts with disable_tma=true'):
        _materialize(_smem_pipeline_kernel(kernel_edit="unsupported_tma_preference"))


@tilelang.testing.requires_cuda
def test_consumer_wait_parity():
    func = _materialize(_smem_pipeline_kernel(depth=2))
    for first_thread, name in ((0, "buf_empty"), (128, "buf_full")):
        role = _role_body(func, first_thread)
        (loop,) = collect_nodes(role, ir.For)
        barrier = _barrier(func, name)
        (wait,) = _barrier_calls(loop.body, "tl.mbarrier_wait_parity", barrier)
        parity = prim.bitwise_and(loop.var // 2, 1)
        if first_thread == 0:
            parity = prim.bitwise_xor(parity, 1)
        tvm_tirx.ir.assert_structural_equal(wait.args, tvm_ffi.convert([ir.TensorLoad(barrier, [loop.var % 2]), parity]))


@tilelang.testing.requires_cuda
def test_uniform_stages_cancel():
    # When ALL of a role's sync stages agree, the offsets cancel and the
    # loop is emitted unshifted (the documented loop-equivalence).
    func = _materialize(_smem_pipeline_kernel(consumer_wait_stage=1, consumer_release_stage=1))
    for first_thread in (0, 128):
        (loop,) = collect_nodes(_role_body(func, first_thread), ir.For)
        assert loop.kind == ir.ForKind.DEFAULT and int(loop.begin) == 0 and int(_extent(loop)) == 4
        assert not collect_nodes(loop.body, ir.If)


def _stage_offset_kernel(extent):
    """Two pipelines synced at different stages within one role: the
    entries at the later stage run one logical iteration behind."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((32, 64), T.float16)
            C_shared = T.alloc_shared((32, 64), T.float16)
            A_frag = T.alloc_fragment((32, 64), T.float16)
            C_frag = T.alloc_fragment((32, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("a", [A_shared], depth=2),
                        T.WSPipeline("c", [C_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("a", stage=0),
                                    "copy_a",
                                    T.WSSync.producer_commit("a", stage=0),
                                    # The second copy runs one iteration
                                    # behind the first.
                                    T.WSSync.producer_acquire("c", stage=1),
                                    "copy_c",
                                    T.WSSync.producer_commit("c", stage=1),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("a", stage=0),
                                    "read_a",
                                    T.WSSync.consumer_release("a", stage=0),
                                    T.WSSync.consumer_wait("c", stage=0),
                                    "read_c",
                                    T.WSSync.consumer_release("c", stage=0),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(extent, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 32, 0], A_shared, annotations={T.WSID: "copy_a"})
                T.copy(A[i * 32, 0], C_shared, annotations={T.WSID: "copy_c"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "read_a"})
                T.copy(C_shared, C_frag, annotations={T.WSID: "read_c"})

    return kernel


@tilelang.testing.requires_cuda
def test_stage_offset_unrolls_prologue_epilogue():
    """The shifted body unrolls into an explicit prologue step, a
    steady-state loop with no per-iteration boundary checks, and an
    explicit epilogue step."""
    func = _materialize(_stage_offset_kernel(4))
    producer = _role_body(func, 0)
    (loop,) = collect_nodes(producer, ir.For)
    assert loop.kind == ir.ForKind.DEFAULT and int(loop.begin) == 1 and int(_extent(loop)) == 3
    assert not collect_nodes(loop.body, ir.If)
    _assert_versioned_access(loop.body, _buffer(func, "C_shared"), (loop.var - 1) % 2)
    prologue, epilogue = [stmt for stmt in producer.seq if isinstance(stmt, ir.If)]
    _assert_stmt_order(producer, prologue, loop, epilogue)
    for step, name, index, parity in ((prologue, "a_empty", 0, 1), (epilogue, "c_empty", 1, 0)):
        assert isinstance(step.condition, ir.IntImm) and int(step.condition) == 1
        barrier = _barrier(func, name)
        (wait,) = _barrier_calls(step.then_case, "tl.mbarrier_wait_parity", barrier)
        tvm_tirx.ir.assert_structural_equal(wait.args, tvm_ffi.convert([ir.TensorLoad(barrier, [index]), ir.IntImm("int32", parity)]))


@tilelang.testing.requires_cuda
def test_short_static_loop_folds_to_prologue_epilogue():
    """A loop shorter than its stage shift: the unrolled prologue and
    epilogue steps cover every iteration under constant bounds, and the
    pipeline's Simplify pass folds them away together with the empty
    steady-state loop."""
    func = _materialize(_stage_offset_kernel(1))
    mod = tilelang.transform.Simplify()(tvm_tirx.IRModule.from_expr(func))
    producer = _role_body(mod["main"], 0)
    for name in ("a_empty", "c_empty"):
        assert len(_barrier_calls(producer, "tl.mbarrier_wait_parity", _barrier(mod["main"], name))) == 1
    assert not collect_nodes(producer, ir.For)
    assert not collect_nodes(producer, ir.If)


@tilelang.testing.requires_cuda
def test_mismatched_pair_stages_rejected():
    with pytest.raises(Exception, match="pairs must share one stage"):
        _materialize(_smem_pipeline_kernel(consumer_wait_stage=0, consumer_release_stage=1))


@tilelang.testing.requires_cuda
def test_missing_commit_rejected():
    # A pipeline whose producer never commits has no full-side signal.
    def drop_commit(bodies):
        bodies["Producer"] = [e for e in bodies["Producer"] if not isinstance(e, T.WSSync) or e.kind.same_as(T.WSSyncKind.PRODUCER_ACQUIRE)]

    with pytest.raises(Exception, match="has no producer"):
        _materialize(_smem_pipeline_kernel(schedule_edit=drop_commit))


@tilelang.testing.requires_cuda
def test_unclosed_span_rejected():
    # An extra trailing acquire leaves the span open at the end of the body.
    def extra_acquire(bodies):
        bodies["Producer"].append(T.WSSync.producer_acquire("buf"))

    with pytest.raises(Exception, match="acquired at the end of a scope body"):
        _materialize(_smem_pipeline_kernel(schedule_edit=extra_acquire))


@tilelang.testing.requires_cuda
def test_double_acquire_rejected():
    # Two acquires without an intervening commit are a double-open.
    def double_acquire(bodies):
        bodies["Producer"].insert(1, T.WSSync.producer_acquire("buf"))
        bodies["Producer"].append(T.WSSync.producer_commit("buf"))

    with pytest.raises(Exception, match="acquired twice"):
        _materialize(_smem_pipeline_kernel(schedule_edit=double_acquire))


@tilelang.testing.requires_cuda
def test_role_both_producer_and_consumer_rejected():
    # The flavors are the two parties of the handshake: a role handing data
    # to itself needs no pipeline, so one role must not hold both flavors.
    def self_handshake(bodies):
        bodies["Producer"] = [
            T.WSSync.producer_acquire("buf"),
            "copy_in",
            T.WSSync.producer_commit("buf"),
            T.WSSync.consumer_wait("buf"),
            "copy_frag",
            T.WSSync.consumer_release("buf"),
        ]
        bodies["Consumer"] = ["copy_out"]

    with pytest.raises(Exception, match="both a producer and a consumer"):
        _materialize(_smem_pipeline_kernel(schedule_edit=self_handshake))


@tilelang.testing.requires_cuda
def test_unknown_pipeline_rejected():
    def rename_pipeline(bodies):
        bodies["Producer"][0] = T.WSSync.producer_acquire("nonexistent")

    with pytest.raises(Exception, match="unknown pipeline"):
        _materialize(_smem_pipeline_kernel(schedule_edit=rename_pipeline))


@tilelang.testing.requires_cuda
def test_missing_op_rejected():
    def add_ghost_op(bodies):
        bodies["Producer"].append("ghost_op")

    with pytest.raises(Exception, match="no statement in the kernel carries this id"):
        _materialize(_smem_pipeline_kernel(schedule_edit=add_ghost_op))


@tilelang.testing.requires_cuda
def test_unscheduled_statement_rejected():
    def drop_consumer_op(bodies):
        bodies["Consumer"] = [e for e in bodies["Consumer"] if e != "copy_frag"]

    with pytest.raises(Exception, match="carries no ws op id|never places it"):
        _materialize(_smem_pipeline_kernel(schedule_edit=drop_consumer_op, kernel_edit="unannotated_consumer"))


@tilelang.testing.requires_cuda
def test_op_with_pipeline_operand_duplicated_across_roles_rejected():
    # copy_frag reads A_shared, a pipeline buffer: duplicating it across
    # roles would have both parties of the handshake access the buffer.
    def share_op(bodies):
        bodies["Producer"].append("copy_frag")

    with pytest.raises(Exception, match="duplicated across roles"):
        _materialize(_smem_pipeline_kernel(schedule_edit=share_op))


@tilelang.testing.requires_cuda
def test_empty_defs_op_duplicated_across_roles():
    """An op touching no pipeline buffers may be placed by several roles;
    each role branch runs its own copy of the statement."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            probe = T.alloc_fragment((64,), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                    "step_frag",
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "step_frag",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["init_frag", "loop_k"],
                                "Consumer": ["init_frag", "loop_k", "copy_out"],
                            },
                        ),
                    ],
                )
            )

            T.fill(probe, 0, annotations={T.WSID: "init_frag"})
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                for r in T.Parallel(64, annotations={T.WSID: "step_frag"}):
                    probe[r] = probe[r] + T.float16(1.0)
            T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    probe = _buffer(func, "probe")
    # The same fill node is shared by both role bodies; query each role's
    # lexical occurrence rather than counting unique nodes in the function.
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        assert_call_count(role, op="tl.tileop.fill", count=1)
        (store,) = [store for store in collect_nodes(role, ir.TensorStore) if store.dest.same_as(probe)]
        (loop,) = [loop for loop in collect_nodes(role, ir.For) if loop.kind == ir.ForKind.PARALLEL]
        assert int(_extent(loop)) == 64
        tvm_tirx.ir.assert_structural_equal(store.indices, tvm_ffi.convert([loop.var]))
        tvm_tirx.ir.assert_structural_equal(store.value, ir.TensorLoad(probe, store.indices) + T.float16(1.0))


@tilelang.testing.requires_cuda
def test_raw_ptx_cp_async_is_opaque():
    """A raw ptx_cp_async gets no special handling: with its own
    commit_group + wait_group(0) it is a synchronous unit like any other
    op — its access_ptr operands bind it to the pipeline (the write-side
    base load), the version dimension joins the base load's indices, and
    the producer commit is the plain per-thread arrive."""

    @T.prim_func
    def kernel(B: T.Tensor((4, 8), T.float16), B_out: T.Tensor((4, 8), T.float16)):
        with T.Kernel(1, threads=128) as _:
            B_shared = T.alloc_shared((8,), T.float16)
            B_frag = T.alloc_fragment((8,), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [B_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "cp_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                with T.ws_op("cp_in"):
                    T.ptx_cp_async(
                        T.access_ptr(B_shared[0], "w", 8),
                        T.access_ptr(B[i, 0], "r", 8),
                        8,
                    )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                T.copy(B_shared, B_frag, annotations={T.WSID: "copy_frag"})
                T.copy(B_frag, B_out[i, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    producer = _role_body(func, 0)
    assert_call_count(func, op="tl.ptx_cp_async_barrier_noinc", count=0)
    assert_call_count(producer, op="tl.ptx_commit_group", count=1)
    (wait,) = collect_calls(producer, op="tl.ptx_wait_group")
    assert len(wait.args) == 1 and int(wait.args[0]) == 0
    full = _barrier(func, "buf_full", [32, 32])
    _barrier(func, "buf_empty", [128, 128])
    assert len(_barrier_calls(producer, "tl.ptx_arrive_barrier", full)) == 1
    shared = _buffer(func, "B_shared")
    for first_thread in (0, 128):
        (loop,) = collect_nodes(_role_body(func, first_thread), ir.For)
        for load in _assert_versioned_access(loop.body, shared, loop.var % 2):
            tvm_tirx.ir.assert_structural_equal(load.indices, tvm_ffi.convert([loop.var % 2, ir.IntImm("int32", 0)]))


def _guarded_scope_kernel(guard_on_pipeline_buffer=False):
    @T.prim_func
    def kernel(
        A: T.Tensor((256, 64), T.float16),
        B: T.Tensor((256, 64), T.float16),
        flag: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            if guard_on_pipeline_buffer:
                if A_shared[0, 0] > T.float16(0):
                    for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                        T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                        T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                        T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})
            else:
                if flag[0] > 0:
                    for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                        T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                        T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                        T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_guarded_scope_loop():
    """A source guard around a scope loop is emitted in every role: all
    roles evaluate the same uniform condition, so a false guard skips the
    scope's sync entries together and the pipeline phases stay aligned."""
    func = _materialize(_guarded_scope_kernel())
    flag = _buffer(func, "flag")
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        (guard,) = collect_nodes(role, ir.If)
        tvm_tirx.ir.assert_structural_equal(guard.condition, ir.TensorLoad(flag, [0]) > 0)
        assert len(collect_nodes(guard.then_case, ir.For)) == 1


@tilelang.testing.requires_cuda
def test_guarded_scope_loop_pipeline_buffer_rejected():
    # A guard reading a pipeline buffer cannot be uniform across roles.
    with pytest.raises(Exception, match="uniform across roles"):
        _materialize(_guarded_scope_kernel(guard_on_pipeline_buffer=True))


def _while_scope_kernel(shifted=False):
    """A persistent-scheduler-style while scope: per-role duplicated
    scheduler state and a smem pipeline synced under the while (forcing a
    runtime phase counter). With ``shifted``, a second producer pipeline
    is consumed at stage 1, so the consumer's while body carries a stage
    shift (forcing an unrolled prologue/epilogue)."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(4, threads=128) as block_id:
            A_shared = T.alloc_shared((32, 64), T.float16)
            C_shared = T.alloc_shared((32, 64), T.float16)
            A_frag = T.alloc_fragment((32, 64), T.float16)
            C_frag = T.alloc_fragment((32, 64), T.float16)
            sched = T.PersistentTileScheduler(4, 1, num_workers=4, name="sched")

            producer = [
                T.WSSync.producer_acquire("buf"),
                "copy_in",
                T.WSSync.producer_commit("buf"),
            ]
            consumer = [
                T.WSSync.consumer_wait("buf", stage=0),
                "copy_frag",
                T.WSSync.consumer_release("buf", stage=0),
            ]
            pipelines = [T.WSPipeline("buf", [A_shared], depth=2)]
            if shifted:
                pipelines.append(T.WSPipeline("cbuf", [C_shared], depth=2))
                producer += [
                    T.WSSync.producer_acquire("cbuf"),
                    "copy_in_c",
                    T.WSSync.producer_commit("cbuf"),
                ]
                consumer += [
                    T.WSSync.consumer_wait("cbuf", stage=1),
                    "copy_frag_c",
                    T.WSSync.consumer_release("cbuf", stage=1),
                ]
            consumer.append("copy_out")

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=pipelines,
                    scopes=[
                        T.WSScope(
                            "loop_wave",
                            {
                                "Producer": producer + ["sched_next"],
                                "Consumer": consumer + ["sched_next"],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["sched_init", "loop_wave"],
                                "Consumer": ["sched_init", "loop_wave"],
                            },
                        ),
                    ],
                )
            )

            with T.ws_op("sched_init"):
                sched.init(block_id)
            with T.ws_op("loop_wave"):
                while sched.valid():
                    T.copy(A[sched.m_idx * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                    if shifted:
                        T.copy(A[sched.m_idx * 64 + 32, 0], C_shared, annotations={T.WSID: "copy_in_c"})
                        T.copy(C_shared, C_frag, annotations={T.WSID: "copy_frag_c"})
                        T.copy(C_frag, B[sched.m_idx * 64 + 32, 0], annotations={T.WSID: "copy_out"})
                    else:
                        T.copy(A_frag, B[sched.m_idx * 64, 0], annotations={T.WSID: "copy_out"})
                    with T.ws_op("sched_next"):
                        sched.next_tile()

    return kernel


@tilelang.testing.requires_cuda
def test_while_scope_counter_phases():
    """A while scope has no iteration expression: the pipeline synced
    under it takes a runtime phase counter in both roles, the scheduler
    state ops are duplicated per role, and each role re-evaluates the
    uniform condition."""
    func = _materialize(_while_scope_kernel())
    linear = _buffer(func, "sched_linear_idx")
    current = _buffer(func, "sched_current_iter")
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        (loop,) = collect_nodes(role, ir.While)
        tvm_tirx.ir.assert_structural_equal(loop.condition, ir.TensorLoad(linear, [0]) < 4)
        _assert_counter_updates(role, current, 1)
        _assert_phase_counter(func, role)


@tilelang.testing.requires_cuda
def test_while_scope_stage_offset_prologue_epilogue():
    """A stage shift under a while scope unrolls into an explicit
    prologue step (the early-stage entries under the loop condition) and
    an epilogue step (the late-stage entries, guarded by the completed
    trip count) instead of per-iteration boundary guards."""
    func = _materialize(_while_scope_kernel(shifted=True))
    consumer = _role_body(func, 128)
    (loop,) = collect_nodes(consumer, ir.While)
    prologue, drain = [stmt for stmt in consumer.seq if isinstance(stmt, ir.If)]
    _assert_stmt_order(consumer, prologue, loop, drain)
    tvm_tirx.ir.assert_structural_equal(prologue.condition, loop.condition)
    trips = _buffer(consumer, "loop_wave_trips")
    initializer = _assert_counter_updates(consumer, trips, 2)
    _assert_stmt_order(consumer, initializer, prologue, loop, drain)
    tvm_tirx.ir.assert_structural_equal(drain.condition, prim.LE(ir.IntImm("int32", 1), ir.TensorLoad(trips, [0])))
    for part in (prologue.then_case, loop.body):
        assert len([store for store in collect_nodes(part, ir.TensorStore) if store.dest.same_as(trips)]) == 1


@tilelang.testing.requires_cuda
def test_while_condition_pipeline_buffer_rejected():
    # A while condition reading a pipeline buffer cannot be uniform.
    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_wave",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_wave"], "Consumer": ["loop_wave"]},
                        ),
                    ],
                )
            )
            with T.ws_op("loop_wave"):
                while A_shared[0, 0] > T.float16(0):
                    T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_in"})
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                    T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="uniform across roles"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_distinct_guards_stay_distinct():
    """Consecutive guarded ops with structurally different guards keep
    their own conditions."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag_lo",
                                    "copy_frag_hi",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                if i < 1:
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag_lo"})
                if i < 2:
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag_hi"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    consumer = _role_body(func, 128)
    (loop,) = collect_nodes(consumer, ir.For)
    guards = collect_nodes(loop.body, ir.If)
    assert len(guards) == 2
    for guard, bound in zip(guards, (1, 2)):
        tvm_tirx.ir.assert_structural_equal(guard.condition, loop.var < bound)
        assert_call_count(guard.then_case, op="tl.tileop.copy", count=1)
    _assert_stmt_order(loop.body, *guards)


@tilelang.testing.requires_cuda
def test_ws_op_wrapper_groups_statements():
    """T.ws_op wrapping several statements (an inlined scheduler method,
    say) forms ONE opaque op: the accesses union, and the whole group is
    cloned into its role."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag_out",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                with T.ws_op("copy_frag_out"):
                    T.copy(A_shared, A_frag)
                    T.copy(A_frag, B[i * 64, 0])

    func = _materialize(kernel)
    consumer = _role_body(func, 128)
    (loop,) = collect_nodes(consumer, ir.For)
    read, write = collect_calls(loop.body, op="tl.tileop.copy")
    read_src, read_dst = _copy_region_loads(read)
    write_src, write_dst = _copy_region_loads(write)
    shared, frag, output = (_buffer(func, name) for name in ("A_shared", "A_frag", "B"))
    assert read_src.source.same_as(shared) and read_dst.source.same_as(frag)
    assert write_src.source.same_as(frag) and write_dst.source.same_as(output)
    tvm_tirx.ir.assert_structural_equal(read_src.indices[0], loop.var % 2)
    _assert_stmt_order(loop.body, _call_stmt(loop.body, read), _call_stmt(loop.body, write))


@tilelang.testing.requires_cuda
def test_unannotated_global_store_rejected():
    """A statement whose only effect is a global write must be placed by
    the schedule like any other op; unannotated it would silently vanish
    from every role body."""
    with pytest.raises(Exception, match="carries no ws op id"):
        _materialize(_smem_pipeline_kernel(kernel_edit="unannotated_global_store"))


def _bind_guard_kernel(bind_in_consumer):
    """Producer binds a scalar read from global memory; the consumer guards
    its pipeline read with that scalar."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            consumer_body = [
                T.WSSync.consumer_wait("buf"),
                "copy_frag",
                T.WSSync.consumer_release("buf"),
                "copy_out",
            ]
            if bind_in_consumer:
                consumer_body.insert(0, "bind_lim")
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    "bind_lim",
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": consumer_body,
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                with T.ws_op("bind_lim"):
                    lim = A[i, 0]
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                if lim > T.float16(0):
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_var_use_requires_defining_op_in_role():
    """A guard using a var whose defining Bind op runs only in the other
    role would reference an undefined variable in the emitted branch."""
    with pytest.raises(Exception, match="does not run earlier in that role"):
        _materialize(_bind_guard_kernel(bind_in_consumer=False))
    # Placing the Bind in both roles (it has no pipeline operands) is legal.
    func = _materialize(_bind_guard_kernel(bind_in_consumer=True))
    _assert_no_undefined_vars(func)
    for first_thread in (0, 128):
        role = _role_body(func, first_thread)
        (loop,) = collect_nodes(role, ir.For)
        (bind,) = collect_nodes(loop.body, ir.Bind)
        assert isinstance(bind.value, ir.TensorLoad) and bind.value.source.same_as(_buffer(func, "A"))
        if first_thread == 128:
            (guard,) = collect_nodes(loop.body, ir.If)
            assert isinstance(guard.condition, prim.GT) and guard.condition.a.same_as(bind.var)
            _assert_stmt_order(loop.body, bind, guard)


@tilelang.testing.requires_cuda
def test_bind_under_stage_shift_freshens_per_slice():
    """A Bind op inside a stage-shifted scope is emitted once per unrolled
    prologue/epilogue step and once in the steady-state loop; every emitted
    copy must define a fresh variable (SSA)."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((32, 64), T.float16)
            C_shared = T.alloc_shared((32, 64), T.float16)
            A_frag = T.alloc_fragment((32, 64), T.float16)
            C_frag = T.alloc_fragment((32, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("a", [A_shared], depth=2),
                        T.WSPipeline("c", [C_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    "bind_v",
                                    T.WSSync.producer_acquire("a", stage=0),
                                    "copy_a",
                                    T.WSSync.producer_commit("a", stage=0),
                                    T.WSSync.producer_acquire("c", stage=1),
                                    "copy_c",
                                    T.WSSync.producer_commit("c", stage=1),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("a", stage=0),
                                    "read_a",
                                    T.WSSync.consumer_release("a", stage=0),
                                    T.WSSync.consumer_wait("c", stage=0),
                                    "read_c",
                                    T.WSSync.consumer_release("c", stage=0),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                with T.ws_op("bind_v"):
                    v = A[i, 0]
                if v > T.float16(0):
                    T.copy(A[i * 32, 0], A_shared, annotations={T.WSID: "copy_a"})
                T.copy(A[i * 32, 0], C_shared, annotations={T.WSID: "copy_c"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "read_a"})
                T.copy(C_shared, C_frag, annotations={T.WSID: "read_c"})

    func = _materialize(kernel)
    assert tirx.analysis.check_well_formed(func)
    _assert_no_undefined_vars(func)
    producer = _role_body(func, 0)
    binds = collect_nodes(producer, ir.Bind)
    assert len(binds) == 2 and not binds[0].var.same_as(binds[1].var)
    for bind in binds:
        assert isinstance(bind.value, ir.TensorLoad) and bind.value.source.same_as(_buffer(func, "A"))
    guards = [guard for guard in collect_nodes(producer, ir.If) if isinstance(guard.condition, prim.GT)]
    assert len(guards) == 2
    for bind, guard in zip(binds, guards):
        assert guard.condition.a.same_as(bind.var)
    # Native If scopes do not export Bind definitions. A statically valid
    # prologue emits its Bind next to the source guard consuming it.
    _assert_stmt_order(producer, binds[0], guards[0])
    (loop,) = collect_nodes(producer, ir.For)
    _assert_stmt_order(loop.body, binds[1], guards[1])


@tilelang.testing.requires_cuda
def test_var_def_use_stage_mismatch_rejected():
    """Under a stage shift, a definition and its use at different deltas
    would be split apart by the unrolled prologue/epilogue steps."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((32, 64), T.float16)
            C_shared = T.alloc_shared((32, 64), T.float16)
            A_frag = T.alloc_fragment((32, 64), T.float16)
            C_frag = T.alloc_fragment((32, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("a", [A_shared], depth=2),
                        T.WSPipeline("c", [C_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    "bind_v",
                                    T.WSSync.producer_acquire("a", stage=0),
                                    "copy_a",
                                    T.WSSync.producer_commit("a", stage=0),
                                    T.WSSync.producer_acquire("c", stage=1),
                                    "copy_c",
                                    T.WSSync.producer_commit("c", stage=1),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("a", stage=0),
                                    "read_a",
                                    T.WSSync.consumer_release("a", stage=0),
                                    T.WSSync.consumer_wait("c", stage=0),
                                    "read_c",
                                    T.WSSync.consumer_release("c", stage=0),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                with T.ws_op("bind_v"):
                    v = A[i, 0]
                T.copy(A[i * 32, 0], A_shared, annotations={T.WSID: "copy_a"})
                if v > T.float16(0):
                    T.copy(A[i * 32, 0], C_shared, annotations={T.WSID: "copy_c"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "read_a"})
                T.copy(C_shared, C_frag, annotations={T.WSID: "read_c"})

    with pytest.raises(Exception, match="at a different stage"):
        _materialize(kernel)


def _alias_pipeline_kernel(use_view):
    @T.prim_func
    def kernel(
        A: T.Tensor((4, 64), T.float16),
        B: T.Tensor((4, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64,), T.float16)
            alias = T.view(shared, (8, 8)) if use_view else T.reshape(shared, (8, 8))
            fragment = T.alloc_fragment((8, 8), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=32),
                        T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=0),
                    ],
                    pipelines=[T.WSPipeline("shared_pipe", [shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop",
                            {
                                "Load": [
                                    T.WSSync.producer_acquire("shared_pipe"),
                                    "load",
                                    T.WSSync.producer_commit("shared_pipe"),
                                ],
                                "Worker": [
                                    T.WSSync.consumer_wait("shared_pipe", stage=1),
                                    "consume_alias",
                                    T.WSSync.consumer_release("shared_pipe", stage=1),
                                    "store",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Load": ["loop"], "Worker": ["loop"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop"}):
                T.copy(A[i, 0], shared, annotations={T.WSID: "load"})
                T.copy(alias, fragment, annotations={T.WSID: "consume_alias"})
                T.copy(fragment, B[i, 0], annotations={T.WSID: "store"})

    return kernel


@tilelang.testing.requires_cuda
def test_versions_reshape_and_view_aliases_by_storage():
    """T.view / T.reshape aliases share their allocation's storage, so they
    join its pipeline and index the same acquired version."""
    for use_view in (False, True):
        func = _materialize(_alias_pipeline_kernel(use_view))
        _barrier(func, "shared_pipe_full")
        shared = _buffer(func, "shared")
        assert [int(dim) for dim in shared.ty.shape] == [2, 64]
        (producer_loop,) = collect_nodes(_role_body(func, 128), ir.For)
        _assert_versioned_access(producer_loop.body, shared, producer_loop.var % 2)
        (consumer_loop,) = collect_nodes(_role_body(func, 0), ir.For)
        (view_bind,) = [bind for bind in collect_nodes(consumer_loop.body, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)]
        assert view_bind.value.op.name == "tirx.decl_tensor"
        shifted = view_bind.value.args[0]
        assert shifted.op.name == "tirx.ptr_byte_offset"
        assert shifted.args[0].op.name == "tirx.tensor_data_ptr"
        alias = shifted.args[0].args[0]
        (alias_bind,) = [bind for bind in collect_nodes(func, ir.Bind) if bind.var.same_as(alias)]
        assert alias_bind.value.op.name == "tirx.decl_tensor"
        assert alias_bind.value.args[0].op.name == "tirx.tensor_data_ptr"
        assert alias_bind.value.args[0].args[0].same_as(shared)
        assert tvm_ffi.structural_equal(view_bind.var.ty, alias.ty)
        assert [int(value) for value in _tensor_spec.tensor_strides(alias)] == [8, 1]
        regions = [region for region in collect_nodes(consumer_loop.body, ir.TensorRegion) if region.source.same_as(view_bind.var)]
        assert len(regions) == 1 and [int(dim.extent) for dim in regions[0].region] == [8, 8]
        # Enumerate the physical byte address, including the version offset,
        # instead of expecting a legacy BufferLoad on the allocation itself.
        for iteration in range(4):
            analyzer = tvm_tirx.sym.Analyzer()
            constraints = tvm_tirx.sym.ConstraintContext()
            constraints.append(tvm_tirx.sym.BindingConstraint(consumer_loop.var, ir.const(iteration, "int32")))
            analyzer.sync_constraint_context(constraints)
            byte_offset = int(analyzer.simplify(shifted.args[1]))
            for row in range(8):
                for column in range(8):
                    assert byte_offset + 2 * (row * 8 + column) == 2 * ((iteration % 2) * 64 + row * 8 + column)


@tilelang.testing.requires_cuda
def test_duplicated_op_global_write_rejected():
    """A multi-role op runs once per role; a non-local write would repeat
    the side effect."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                    "mark_global",
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                    "mark_global",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})
                with T.ws_op("mark_global"):
                    B[0, 0] = T.float16(1)

    with pytest.raises(Exception, match="every role would repeat the write"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_async_copy_in_consumer_role_rejected():
    """A TMA copy writing a pipeline buffer must run in the role that
    commits the pipeline's full side, or its transaction bytes are never
    accounted."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "fill_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_in",
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.fill(A_shared, T.float16(0), annotations={T.WSID: "fill_in"})
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="never observes it finishing"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_kernel_metadata_attrs_preserved():
    """Kernel-level metadata AttrStmts (T.use_swizzle,
    T.annotate_min_blocks_per_sm) wrap the statements that follow them:
    the pass schedules through them and re-wraps the rebuilt body."""
    func = _materialize(_smem_pipeline_kernel(kernel_edit="metadata_attrs"))
    (swizzle,) = [attr for attr in collect_nodes(func, ir.RegionStmt) if _attr_key(attr) == "threadblock_swizzle_pattern"]
    (min_blocks,) = [attr for attr in collect_nodes(func, ir.RegionStmt) if _attr_key(attr) == "tl.min_blocks_per_sm"]
    (custom,) = [attr for attr in collect_nodes(func, ir.RegionStmt) if _attr_key(attr) == "kCustomWarpSpecialization"]
    assert _only(swizzle.body).same_as(min_blocks) and _only(min_blocks.body).same_as(custom)
    assert int(_annotations(min_blocks)["value"]) == 2
    assert_call_count(func, op="tl.tileop.tma_copy", count=1)


@tilelang.testing.requires_cuda
def test_nonzero_loop_base_phase():
    """A scope loop starting at a non-zero base counts pipeline phases from
    the base: iteration `base` is phase 0, so barrier indices and parities
    come from (i - base), not the raw loop var."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(3, 7, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[(i - 3) * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[(i - 3) * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    shared = _buffer(func, "A_shared")
    for first_thread in (0, 128):
        (loop,) = collect_nodes(_role_body(func, first_thread), ir.For)
        assert int(loop.begin) == 3 and int(_extent(loop)) == 4
        version = (loop.var - 3) % 2
        _assert_versioned_access(loop.body, shared, version)
        for wait in collect_calls(loop.body, op="tl.mbarrier_wait_parity"):
            tvm_tirx.ir.assert_structural_equal(wait.args[0].indices[0], version)


@tilelang.testing.requires_cuda
def test_release_before_use_rejected():
    """Work on a pipeline's buffers after the span closed races with the
    next producer of the stage; the span-coverage check rejects it."""

    def release_early(bodies):
        # consumer: wait, release, then read -- copy_frag lands outside.
        bodies["Consumer"] = [
            T.WSSync.consumer_wait("buf"),
            T.WSSync.consumer_release("buf"),
            "copy_frag",
            "copy_out",
        ]

    with pytest.raises(Exception, match="outside an open span"):
        _materialize(_smem_pipeline_kernel(schedule_edit=release_early))


@tilelang.testing.requires_cuda
def test_unbracketed_producer_rejected():
    """A producer op scheduled outside its pipeline's acquire/commit
    bracket is rejected by the span-coverage check even though the bracket
    itself is present."""

    def op_outside_bracket(bodies):
        bodies["Producer"] = [
            "copy_in",
            T.WSSync.producer_acquire("buf"),
            T.WSSync.producer_commit("buf"),
        ]

    with pytest.raises(Exception, match="outside an open span"):
        _materialize(_smem_pipeline_kernel(schedule_edit=op_outside_bracket))


@tilelang.testing.requires_cuda
def test_cycle_imbalance_rejected():
    """A pipeline cycling twice on the producer side but once on the
    consumer side per loop iteration diverges parity every trip."""

    def double_producer_cycle(bodies):
        bodies["Producer"] = [
            T.WSSync.producer_acquire("buf"),
            "copy_in",
            T.WSSync.producer_commit("buf"),
            T.WSSync.producer_acquire("buf"),
            T.WSSync.producer_commit("buf"),
        ]

    with pytest.raises(Exception, match="parity diverges"):
        _materialize(_smem_pipeline_kernel(schedule_edit=double_producer_cycle))


@tilelang.testing.requires_cuda
def test_cross_pipeline_deadlock_rejected():
    """Two roles waiting on each other's pipelines before producing their
    own deadlock immediately; the parity-model execution catches it."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            X_shared = T.alloc_shared((64, 64), T.float16)
            Y_shared = T.alloc_shared((64, 64), T.float16)
            X_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("RoleX", warps_lo=0, warps_hi=4, max_nreg=224),
                        T.WSRole("RoleY", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("x", [X_shared], depth=1),
                        T.WSPipeline("y", [Y_shared], depth=1),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                # RoleX produces x but first waits on y;
                                # RoleY produces y but first waits on x.
                                "RoleX": [
                                    T.WSSync.consumer_wait("y"),
                                    T.WSSync.producer_acquire("x"),
                                    "copy_x",
                                    T.WSSync.producer_commit("x"),
                                    "read_y",
                                    T.WSSync.consumer_release("y"),
                                    "copy_out_x",
                                ],
                                "RoleY": [
                                    T.WSSync.consumer_wait("x"),
                                    T.WSSync.producer_acquire("y"),
                                    "copy_y",
                                    T.WSSync.producer_commit("y"),
                                    T.WSSync.consumer_release("x"),
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"RoleX": ["loop_k"], "RoleY": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=1, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], X_shared, annotations={T.WSID: "copy_x"})
                T.copy(X_shared, Y_shared, annotations={T.WSID: "copy_y"})
                T.copy(Y_shared, X_frag, annotations={T.WSID: "read_y"})
                T.copy(X_frag, B[i * 64, 0], annotations={T.WSID: "copy_out_x"})

    with pytest.raises(Exception, match="deadlock"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_op_writing_two_pipelines_rejected():
    """An op's synchronization can only trigger one pipeline: a single op
    statement (here a T.Parallel op node) writing buffers of two pipelines
    is rejected."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            X_shared = T.alloc_shared((64, 64), T.float16)
            Y_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("x", [X_shared], depth=2),
                        T.WSPipeline("y", [Y_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("x"),
                                    T.WSSync.producer_acquire("y"),
                                    "copy_both",
                                    T.WSSync.producer_commit("x"),
                                    T.WSSync.producer_commit("y"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("x"),
                                    T.WSSync.consumer_wait("y"),
                                    "copy_frag_x",
                                    "copy_frag_y",
                                    T.WSSync.consumer_release("x"),
                                    T.WSSync.consumer_release("y"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_k"], "Consumer": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                # One statement (a T.Parallel op node) writing buffers of
                # two different pipelines.
                for u, v in T.Parallel(64, 64, annotations={T.WSID: "copy_both"}):
                    X_shared[u, v] = A[i * 64 + u, v]
                    Y_shared[u, v] = A[i * 64 + u, v]
                T.copy(X_shared, A_frag, annotations={T.WSID: "copy_frag_x"})
                T.copy(Y_shared, A_frag, annotations={T.WSID: "copy_frag_y"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="can only trigger one pipeline"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_unknown_role_body_rejected():
    def add_role_body(bodies):
        bodies["Ghost"] = []

    with pytest.raises(Exception, match="unknown role"):
        _materialize(_smem_pipeline_kernel(schedule_edit=add_role_body))


@tilelang.testing.requires_cuda
def test_num_warps_must_be_warpgroup_aligned():
    # Warps are managed in warpgroups of 4.
    with pytest.raises(AssertionError, match="multiple of 4"):
        T.WSSchedule(
            num_warps=6,
            roles=[T.WSRole("R", warps_lo=0, warps_hi=6)],
            pipelines=[],
            scopes=[],
        )


@tilelang.testing.requires_cuda
def test_ws_scope_root_constant():
    assert T.WSScope.ROOT == "tl.ws_scope_root"


@tilelang.testing.requires_cuda
def test_ws_sync_kind_enum_round_trip():
    sync = T.WSSync.consumer_release("p", stage=3)
    assert sync.kind.same_as(T.WSSyncKind.CONSUMER_RELEASE)
    assert str(sync.pipeline) == "p"
    assert int(sync.stage) == 3


@tilelang.testing.requires_cuda
def test_ws_role_requires_keyword_range():
    role = T.WSRole("R", warps_lo=4, warps_hi=8, max_nreg=224)
    assert int(role.warp_lo) == 4 and int(role.warp_hi) == 8
    with pytest.raises(TypeError):
        T.WSRole("R", (4, 8))  # positional tuples are not accepted


def _scope_kind_kernel(loop_ctor):
    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in loop_ctor(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_sequential_scope_loop_kinds():
    """Any sequential loop can be a schedule scope (T.Pipelined is a serial
    loop whose num_stages this pass consumes; a T.unroll scope keeps its
    unroll kind); T.Parallel loops cannot."""
    # A plain T.serial scope materializes like a T.Pipelined one.
    for ctor, kind in ((T.serial, ir.ForKind.DEFAULT), (T.unroll, ir.ForKind.UNROLLED)):
        func = _materialize(_scope_kind_kernel(ctor))
        for first_thread in (0, 128):
            role = _role_body(func, first_thread)
            (loop,) = collect_nodes(role, ir.For)
            assert loop.kind == kind and int(_extent(loop)) == 4
            assert "num_stages" not in loop.annotations
            assert_call_count(loop.body, op="tl.mbarrier_wait_parity", count=1)
    with pytest.raises(Exception, match="must be a sequential loop"):
        _materialize(_scope_kind_kernel(T.Parallel))


@tilelang.testing.requires_cuda
def test_op_node_loop():
    """A T.unroll loop with an id is one op node: inner statements need no
    ids, the loop keeps its kind, and its pipeline reads rebind to the
    acquired version."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "store_loop",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                # The unrolled loop is one op node; its statements are
                # unannotated and its A_shared read binds it to the pipeline.
                for s in T.unroll(2, annotations={T.WSID: "store_loop"}):
                    T.copy(A_shared[:, s * 32 : (s + 1) * 32], A_frag[:, s * 32 : (s + 1) * 32])
                    T.copy(A_frag[:, s * 32 : (s + 1) * 32], B[i * 64, s * 32])

    func = _materialize(kernel)
    consumer = _role_body(func, 128)
    (outer,) = [loop for loop in collect_nodes(consumer, ir.For) if loop.kind == ir.ForKind.DEFAULT]
    (inner,) = [loop for loop in collect_nodes(outer.body, ir.For) if loop.kind == ir.ForKind.UNROLLED]
    assert int(_extent(inner)) == 2
    shared = _buffer(func, "A_shared")
    for load in _assert_versioned_access(inner.body, shared, outer.var % 2):
        tvm_tirx.ir.assert_structural_equal(load.indices, tvm_ffi.convert([outer.var % 2, ir.IntImm("int32", 0), inner.var * 32]))


@tilelang.testing.requires_cuda
def test_idle_donor_branch():
    # Roles cover warps [0,1) and [4,8) of 8: warps 1-4 form a gap branch
    # and warps beyond the last role donate registers too.
    func = _materialize(_smem_pipeline_kernel())
    assert _nreg_args(_role_body(func, 32)) == [(40, 0)]
    assert _nreg_args(func).count((40, 0)) >= 2


@tilelang.testing.requires_cuda
def test_unscheduled_kernel_untouched():
    """Schedule and op ids come together (both or neither): a kernel
    without the schedule annotation is returned exactly as it came in —
    no transformation, no stripping."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_in"})
            T.copy(A_shared, B[0, 0], annotations={T.WSID: "copy_out"})

    mod = tvm_tirx.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    mod = tvm_tirx.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    out = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    assert tvm_ffi.structural_equal(out, mod["main"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("blockscaled", [False, True])
def test_tcgen05_async_arrive_count(blockscaled):
    """A gemm accumulating into TMEM signals with tcgen05.commit: the full
    barrier of the accumulator pipeline gets arrive count 1, and the smem
    release below the gemm becomes a tcgen05_mma_arrive."""
    tile = 128 if blockscaled else 64
    dtype = T.float8_e4m3fn if blockscaled else T.float16

    @T.prim_func
    def kernel(A: T.Tensor((4 * tile, tile), dtype), B: T.Tensor((tile, tile), T.float32)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((tile, tile), dtype)
            C_tmem = T.alloc_tmem((tile, tile), T.float32)
            C_frag = T.alloc_fragment((tile, tile), T.float32)
            if blockscaled:
                SFA = T.alloc_tmem((tile, 4), T.uint32)
                SFB = T.alloc_tmem((tile, 4), T.uint32)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("MMA", warps_lo=1, warps_hi=2, max_nreg=40),
                        T.WSRole("Epilogue", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("smem", [A_shared], depth=2),
                        T.WSPipeline("acc", [C_tmem], depth=1),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("smem"),
                                    "copy_in",
                                    T.WSSync.producer_commit("smem"),
                                ],
                                "MMA": [
                                    T.WSSync.consumer_wait("smem"),
                                    "gemm_C",
                                    T.WSSync.consumer_release("smem"),
                                ],
                                "Epilogue": [],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["loop_k"],
                                "MMA": [
                                    T.WSSync.producer_acquire("acc"),
                                    "loop_k",
                                    T.WSSync.producer_commit("acc"),
                                ],
                                "Epilogue": [
                                    "loop_k",
                                    T.WSSync.consumer_wait("acc"),
                                    "copy_C",
                                    T.WSSync.consumer_release("acc"),
                                    "copy_out",
                                ],
                            },
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * tile, 0], A_shared, annotations={T.WSID: "copy_in"})
                if blockscaled:
                    T.gemm_blockscaled(
                        A_shared,
                        A_shared,
                        C_tmem,
                        SFA,
                        SFB,
                        transpose_B=True,
                        clear_accum=i == 0,
                        k_start=i * tile,
                        sf_a_granularity_k=128,
                        sf_b_granularity_k=128,
                        annotations={T.WSID: "gemm_C"},
                    )
                else:
                    T.gemm(
                        A_shared,
                        A_shared,
                        C_tmem,
                        transpose_B=True,
                        clear_accum=i == 0,
                        annotations={T.WSID: "gemm_C"},
                    )
            T.copy(C_tmem, C_frag, annotations={T.WSID: "copy_C"})
            T.copy(C_frag, B[0, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    calls = [
        call
        for op in ("gemm", "gemm_blockscaled", "tcgen05_gemm", "tcgen05_gemm_blockscaled")
        for call in collect_calls(func, op="tl.tileop." + op)
    ]
    (call,) = calls
    # Scheduling converts each GEMM to its explicit asynchronous CUDA op.
    explicit_op = "tcgen05_gemm_blockscaled" if blockscaled else "tcgen05_gemm"
    assert call.op.name == "tl.tileop." + explicit_op
    gemm = call.op.get_attr("TLOpBuilder")(_operands(call), _annotations(call))
    assert isinstance(gemm, GemmBlockScaled if blockscaled else Gemm)
    assert gemm.isTcgen05
    assert gemm.mbar is None
    if blockscaled:
        assert gemm.sfaRegion.source.name == "SFA"
        assert gemm.sfbRegion.source.name == "SFB"

    # The scheduled op must lower without a per-op barrier or an implicit
    # wait/arrival: the schedule's synchronization points publish completion.
    with _TARGET:
        layouts = gemm.infer_layout(_TARGET, 32)
        lowered = gemm.lower(layouts, _TARGET, tvm_tirx.ir.Range(32, 64), ir.Var("tx", "int32"), ir.const(0, "int32"))
    assert_call_count(lowered, op="tl.mbarrier_wait_parity", count=0)
    assert_call_count(lowered, op="tl.tcgen05_mma_arrive", count=0)
    mma_op = "ptx_tcgen05_mma_blockscaled_ss" if blockscaled else "ptx_tcgen05_mma_ss"
    assert collect_calls(lowered, op="tl." + mma_op)
    # smem_empty and acc_full are signaled by the tensor core: count 1 each.
    smem_empty = _barrier(func, "smem_empty", [1, 1])
    acc_full = _barrier(func, "acc_full", [1])
    # Synchronous epilogue reads release acc with all 128 threads.
    _barrier(func, "acc_empty", [128])
    # The MMA-side releases are tcgen05 commits.
    arrives = collect_calls(func, op="tl.tcgen05_mma_arrive")
    assert len(arrives) == 2
    for barrier in (smem_empty, acc_full):
        pointers = [
            call.args[0]
            for call in arrives
            if len(call.args) == 1
            and isinstance(call.args[0], ir.Call)
            and call.args[0].op.same_as(tvm_tirx.ir.Op.get("tl.access_ptr"))
            and len(call.args[0].args) == 3
            and isinstance(call.args[0].args[0], ir.TensorLoad)
            and call.args[0].args[0].source.same_as(barrier)
        ]
        assert len(pointers) == 1
        pointer = pointers[0]
        assert int(pointer.args[2]) == 3
        analyzer = tvm_tirx.sym.Analyzer()
        assert analyzer.can_prove(pointer.args[0].indices[0] >= 0)
        assert analyzer.can_prove(pointer.args[0].indices[0] < barrier.ty.shape[0])
        assert analyzer.can_prove(pointer.args[1] >= 1)


@tilelang.testing.requires_cuda
def test_runtime_phase_counter_for_multi_cycle_body():
    """A pipeline cycled twice per iteration of its scope needs a runtime
    phase counter instead of the linearized iteration."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in2",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag2",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out2",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(2, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 128, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 128, 0], annotations={T.WSID: "copy_out"})
                T.copy(A[i * 128 + 64, 0], A_shared, annotations={T.WSID: "copy_in2"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag2"})
                T.copy(A_frag, B[i * 128 + 64, 0], annotations={T.WSID: "copy_out2"})

    func = _materialize(kernel)
    for first_thread in (0, 128):
        _assert_phase_counter(func, _role_body(func, first_thread), updates=2)


@tilelang.testing.requires_cuda
def test_pipeline_synced_in_two_scopes_rejected():
    """A pipeline synchronizes exactly one scope; syncing buf at the root
    AND inside loop_k must be split into two (nested) pipelines."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_head",
                                    T.WSSync.producer_commit("buf"),
                                    "loop_k",
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_head_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_head_out",
                                    "loop_k",
                                ],
                            },
                        ),
                    ],
                )
            )

            T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_head"})
            T.copy(A_shared, A_frag, annotations={T.WSID: "copy_head_frag"})
            T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_head_out"})
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="synchronizes exactly one scope"):
        _materialize(kernel)


def _nested_pipeline_kernel(accum_brackets_loop=True):
    """An rmw-shaped nested binding: inside loop_k the Rescale role writes
    A_shared and the Accum role read-modify-writes it (pipeline `buf`); at
    the root, Accum producer-brackets the whole loop and Rescale reads the
    final value (pipeline `buf_outer`)."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            B_frag = T.alloc_fragment((64, 64), T.float16)

            accum_root = (
                [
                    T.WSSync.producer_acquire("buf_outer"),
                    "loop_k",
                    T.WSSync.producer_commit("buf_outer"),
                ]
                if accum_brackets_loop
                # The bracket misses the loop: Accum's writes inside loop_k
                # would not publish through the outer commit.
                else [
                    "loop_k",
                    T.WSSync.producer_acquire("buf_outer"),
                    T.WSSync.producer_commit("buf_outer"),
                ]
            )
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Rescale", warps_lo=0, warps_hi=4, max_nreg=224),
                        T.WSRole("Accum", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf", [A_shared], depth=1),
                        T.WSPipeline("buf_outer", [A_shared], depth=1),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Rescale": [
                                    T.WSSync.producer_acquire("buf"),
                                    "produce",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Accum": [
                                    T.WSSync.consumer_wait("buf"),
                                    "acc_load",
                                    "acc_store",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Rescale": [
                                    "loop_k",
                                    T.WSSync.consumer_wait("buf_outer"),
                                    "read_back",
                                    T.WSSync.consumer_release("buf_outer"),
                                    "store_out",
                                ],
                                "Accum": accum_root,
                            },
                        ),
                    ],
                )
            )

            for i in T.serial(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "produce"})
                T.copy(A_shared, B_frag, annotations={T.WSID: "acc_load"})
                T.copy(B_frag, A_shared, annotations={T.WSID: "acc_store"})
            T.copy(A_shared, A_frag, annotations={T.WSID: "read_back"})
            T.copy(A_frag, B[0, 0], annotations={T.WSID: "store_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_nested_pipelines_on_one_storage():
    """A buffer bound to two pipelines in strictly nested scopes
    materializes: each pipeline gets its own barrier pair, and every
    access resolves against the innermost binding whose scope contains
    it."""
    func = _materialize(_nested_pipeline_kernel())
    for name in ("buf_full", "buf_empty", "buf_outer_full", "buf_outer_empty"):
        _barrier(func, name)


@tilelang.testing.requires_cuda
def test_nested_pipeline_producer_must_bracket_scope():
    """The outer pipeline's producer must hold its span across the nested
    scope: its writes inside publish through the outer commit."""
    with pytest.raises(Exception, match="enclosing pipeline"):
        _materialize(_nested_pipeline_kernel(accum_brackets_loop=False))


@tilelang.testing.requires_cuda
def test_sibling_pipelines_on_one_storage_rejected():
    """Two pipelines on one buffer whose scopes are siblings are rejected:
    nothing chains their barriers, so the second loop's pre-armed acquire
    could overwrite data the first loop's consumer still reads."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf_a", [A_shared], depth=2),
                        T.WSPipeline("buf_b", [A_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_a",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf_a"),
                                    "copy_in_a",
                                    T.WSSync.producer_commit("buf_a"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf_a"),
                                    "copy_frag_a",
                                    T.WSSync.consumer_release("buf_a"),
                                    "copy_out_a",
                                ],
                            },
                        ),
                        T.WSScope(
                            "loop_b",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf_b"),
                                    "copy_in_b",
                                    T.WSSync.producer_commit("buf_b"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf_b"),
                                    "copy_frag_b",
                                    T.WSSync.consumer_release("buf_b"),
                                    "copy_out_b",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["loop_a", "loop_b"],
                                "Consumer": ["loop_a", "loop_b"],
                            },
                        ),
                    ],
                )
            )

            for i in T.serial(2, annotations={T.WSID: "loop_a"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in_a"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag_a"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out_a"})
            for i in T.serial(2, annotations={T.WSID: "loop_b"}):
                T.copy(A[128 + i * 64, 0], A_shared, annotations={T.WSID: "copy_in_b"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag_b"})
                T.copy(A_frag, B[128 + i * 64, 0], annotations={T.WSID: "copy_out_b"})

    with pytest.raises(Exception, match="strictly nested"):
        _materialize(kernel)


def _nested_ring_kernel(rescale_syncs_before_loop=False):
    """The rmw shape under an outer FOR loop with a depth-2 outer pipeline:
    O-style storage double-buffers across outer iterations (versions
    1 x 2). Rescale's in-loop produce holds only the inner span and derives
    the outer slot from its own phase (the linearized outer iteration)."""

    @T.prim_func
    def kernel(A: T.Tensor((512, 64), T.float16), B: T.Tensor((128, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            B_frag = T.alloc_fragment((64, 64), T.float16)

            rescale_wave = (
                # The syncs run before the loop reference: Rescale's phase
                # then names the NEXT cycle when produce runs — rejected.
                [
                    T.WSSync.consumer_wait("buf_outer"),
                    "read_back",
                    T.WSSync.consumer_release("buf_outer"),
                    "loop_k",
                    "store_out",
                ]
                if rescale_syncs_before_loop
                else [
                    "loop_k",
                    T.WSSync.consumer_wait("buf_outer"),
                    "read_back",
                    T.WSSync.consumer_release("buf_outer"),
                    "store_out",
                ]
            )
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Rescale", warps_lo=0, warps_hi=4, max_nreg=224),
                        T.WSRole("Accum", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf", [A_shared], depth=1),
                        T.WSPipeline("buf_outer", [A_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Rescale": [
                                    T.WSSync.producer_acquire("buf"),
                                    "produce",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Accum": [
                                    T.WSSync.consumer_wait("buf"),
                                    "acc_load",
                                    "acc_store",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(
                            "loop_w",
                            {
                                "Rescale": rescale_wave,
                                "Accum": [
                                    T.WSSync.producer_acquire("buf_outer"),
                                    "loop_k",
                                    T.WSSync.producer_commit("buf_outer"),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Rescale": ["loop_w"], "Accum": ["loop_w"]}),
                    ],
                )
            )

            for w in T.serial(2, annotations={T.WSID: "loop_w"}):
                for i in T.serial(4, annotations={T.WSID: "loop_k"}):
                    T.copy(A[(w * 4 + i) * 64, 0], A_shared, annotations={T.WSID: "produce"})
                    T.copy(A_shared, B_frag, annotations={T.WSID: "acc_load"})
                    T.copy(B_frag, A_shared, annotations={T.WSID: "acc_store"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "read_back"})
                T.copy(A_frag, B[w * 64, 0], annotations={T.WSID: "store_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_nested_pipeline_outer_ring():
    """A depth-2 outer binding over a depth-1 inner one gives the buffer
    two slots cycling with the outer loop; every access resolves the outer
    slot — from its span, or (the joint-role produce) from the role's own
    phase."""
    func = _materialize(_nested_ring_kernel())
    for name in ("buf_outer_full", "buf_outer_empty"):
        _barrier(func, name)
    assert [int(dim) for dim in _buffer(func, "A_shared").shape] == [2, 1, 64, 64]


@tilelang.testing.requires_cuda
def test_nested_pipeline_access_after_sync_rejected():
    """Once the role's sync entries of the outer pipeline have run, its
    phase names the next cycle, so a later unheld access can no longer
    derive the in-production slot."""
    with pytest.raises(Exception, match="after this role's sync"):
        _materialize(_nested_ring_kernel(rescale_syncs_before_loop=True))


def _observer_producer_kernel():
    """Nested bindings where the outer producer never writes the storage:
    the Load role TMA-fills A_shared inside loop_k, the Worker observes it
    through the inner handshake and hands the whole loop's result to the
    Out role. The collapsed scope use is what the Worker's outer commit
    publishes."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Load", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Out", warps_lo=1, warps_hi=2, max_nreg=40),
                        T.WSRole("Worker", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf", [A_shared], depth=1),
                        T.WSPipeline("buf_outer", [A_shared], depth=1),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Load": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Worker": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Load": ["loop_k"],
                                "Worker": [
                                    T.WSSync.producer_acquire("buf_outer"),
                                    "loop_k",
                                    T.WSSync.producer_commit("buf_outer"),
                                ],
                                "Out": [
                                    T.WSSync.consumer_wait("buf_outer"),
                                    "copy_out",
                                    T.WSSync.consumer_release("buf_outer"),
                                ],
                            },
                        ),
                    ],
                )
            )

            for i in T.serial(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
            T.copy(A_shared, B[0, 0], annotations={T.WSID: "copy_out"})

    return kernel


@tilelang.testing.requires_cuda
def test_scope_use_publishes_observed_writes():
    """The Worker writes nothing, yet its commit validly publishes the
    scope's TMA-filled data — its thread observed every inner handoff
    before leaving the loop, so a plain per-thread arrive suffices (the
    collapsed scope use is synchronous)."""
    func = _materialize(_observer_producer_kernel())
    _barrier(func, "buf_outer_full", [128])
    _barrier(func, "buf_outer_empty", [32])


@tilelang.testing.requires_cuda
def test_op_referenced_twice_rejected():
    # An op's statement is cloned verbatim; referencing it twice would
    # duplicate the work.
    def duplicate_ref(bodies):
        bodies["Producer"].insert(2, "copy_in")

    with pytest.raises(Exception, match="referenced more than once in the body"):
        _materialize(_smem_pipeline_kernel(schedule_edit=duplicate_ref))


@tilelang.testing.requires_cuda
def test_scope_loop_nonunit_step_rejected():
    """Phases count iterations as (loop_var - min), which a non-unit step
    breaks. The eager frontend desugars T.serial(..., step) into a
    unit-step loop plus a Bind, so stamp the For node's step field
    directly to pin the IR-level rejection."""
    func = _smem_pipeline_kernel()

    def add_step(stmt):
        if isinstance(stmt, ir.For) and T.WSID in stmt.annotations:
            return ir.For(
                stmt.var,
                stmt.begin,
                stmt.end,
                ir.IntImm("int32", 2),
                kind=stmt.kind,
                body=stmt.body,
                annotations=stmt.annotations,
            )
        return stmt

    stepped = func.with_body(tvm_ffi.structural_map(func.body, (ir.For, add_step)))
    with pytest.raises(Exception, match="non-unit loop step"):
        _materialize(stepped)


@tilelang.testing.requires_cuda
def test_guard_read_outside_span_rejected():
    """A buffer read only by an op's if-condition is an operand too: a
    guard reading a pipeline buffer after the role released it is
    rejected by the span-coverage check."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                if A_shared[0, 0] > T.float16(0.0):
                    T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="outside an open span"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_unbalanced_scope_cycles_rejected():
    """A pipeline cycling twice on the producer side but once on the
    consumer side per iteration diverges with the loop extent; every
    pipeline must cycle both sides equally often within its scope."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in2",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for _i in T.serial(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A[64, 0], A_shared, annotations={T.WSID: "copy_in2"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="parity diverges"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_source_guard_follows_shifted_iteration():
    """The source guard of an op at a later stage is substituted with the
    op's shifted iteration — it is the only condition inside the
    steady-state loop, which carries no boundary checks of its own."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((32, 64), T.float16)
            C_shared = T.alloc_shared((32, 64), T.float16)
            A_frag = T.alloc_fragment((32, 64), T.float16)
            C_frag = T.alloc_fragment((32, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("a", [A_shared], depth=2),
                        T.WSPipeline("c", [C_shared], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("a", stage=0),
                                    "copy_a",
                                    T.WSSync.producer_commit("a", stage=0),
                                    T.WSSync.producer_acquire("c", stage=1),
                                    "copy_c",
                                    T.WSSync.producer_commit("c", stage=1),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("a", stage=0),
                                    "read_a",
                                    T.WSSync.consumer_release("a", stage=0),
                                    T.WSSync.consumer_wait("c", stage=0),
                                    "read_c",
                                    T.WSSync.consumer_release("c", stage=0),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 32, 0], A_shared, annotations={T.WSID: "copy_a"})
                if i < 3:
                    T.copy(A[i * 32, 0], C_shared, annotations={T.WSID: "copy_c"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "read_a"})
                T.copy(C_shared, C_frag, annotations={T.WSID: "read_c"})

    func = _materialize(kernel)
    producer = _role_body(func, 0)
    (loop,) = collect_nodes(producer, ir.For)
    assert int(loop.begin) == 1 and int(_extent(loop)) == 3
    (guard,) = collect_nodes(loop.body, ir.If)
    tvm_tirx.ir.assert_structural_equal(guard.condition, loop.var - 1 < 3)
    (copy,) = collect_calls(guard.then_case, op="tl.tileop.tma_copy")
    _, dest = _copy_region_loads(copy)
    assert dest.source.same_as(_buffer(func, "C_shared"))


@tilelang.testing.requires_cuda
def test_warpgroup_nreg_conflict_rejected():
    """setmaxnreg allocates per warpgroup: two roles inside one warpgroup
    (warps 4g..4g+3) must request the same max_nreg."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[
                        T.WSRole("R0", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("R1", warps_lo=1, warps_hi=2, max_nreg=224),
                    ],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"R0": ["cp"], "R1": []})],
                )
            )
            T.copy(A, B, annotations={T.WSID: "cp"})

    with pytest.raises(Exception, match="allocates per warpgroup"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_idle_warps_adopt_warpgroup_nreg():
    """Idle warps sharing a warpgroup with a role execute that warpgroup's
    request — not the globally smallest donor budget (here the consumer's
    40 would break warpgroup 0's uniform 80)."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=80),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=40),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    args = _nreg_args(func)
    assert args.count((80, 1)) == 2
    assert args.count((40, 0)) == 1


@tilelang.testing.requires_cuda
def test_two_scheduled_kernels_in_one_function():
    """A function may hold several kernels, each with its own schedule;
    each is materialized under its own threadIdx.x binding, widened to
    that schedule's warp count."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[T.WSRole("R", warps_lo=0, warps_hi=4, max_nreg=224)],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"R": ["cp"]})],
                )
            )
            T.copy(A, B, annotations={T.WSID: "cp"})
        with T.Kernel(1, threads=128) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[T.WSRole("S", warps_lo=0, warps_hi=8, max_nreg=224)],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"S": ["cp2"]})],
                )
            )
            T.copy(B, A, annotations={T.WSID: "cp2"})

    func = _materialize(kernel)
    launches = [
        attr for attr in collect_nodes(func, ir.RegionStmt) if attr.op.name == "tirx.launch_thread" and attr.args[0].value == "threadIdx.x"
    ]
    assert sorted(int(launch.args[1]) for launch in launches) == [128, 256]
    for launch in launches:
        assert _nreg_args(launch.body).count((224, 1)) == 1


@tilelang.testing.requires_cuda
def test_multi_dim_thread_launch_rejected():
    """Warp roles partition threadIdx.x only: a scheduled kernel under a
    2-D thread launch would mis-assign warps."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=(64, 2)) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[T.WSRole("R", warps_lo=0, warps_hi=4, max_nreg=224)],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"R": ["cp"]})],
                )
            )
            T.copy(A, B, annotations={T.WSID: "cp"})

    with pytest.raises(Exception, match="one\\s+dimension"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_negative_warp_range_rejected():
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[T.WSRole("R", warps_lo=-1, warps_hi=1, max_nreg=224)],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"R": ["cp"]})],
                )
            )
            T.copy(A, B, annotations={T.WSID: "cp"})

    with pytest.raises(Exception, match="negative warp range"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_capacity_deadlock_rejected():
    """Two roles produce into sibling scopes in opposite orders: each
    fills its pipeline's depth and blocks on an acquire that only the
    other's never-reached consumer scope would release. Surfaces only
    when loops are modeled past two iterations."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            S1 = T.alloc_shared((64, 64), T.float16)
            S2 = T.alloc_shared((64, 64), T.float16)
            F1 = T.alloc_fragment((64, 64), T.float16)
            F2 = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("s1", [S1], depth=2),
                        T.WSPipeline("s2", [S2], depth=2),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_a",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("s1"),
                                    "copy_in1",
                                    T.WSSync.producer_commit("s1"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("s1"),
                                    "copy_frag1",
                                    T.WSSync.consumer_release("s1"),
                                    "copy_out1",
                                ],
                            },
                        ),
                        T.WSScope(
                            "loop_b",
                            {
                                "Consumer": [
                                    T.WSSync.producer_acquire("s2"),
                                    "copy_in2",
                                    T.WSSync.producer_commit("s2"),
                                ],
                                "Producer": [
                                    T.WSSync.consumer_wait("s2"),
                                    "copy_frag2",
                                    T.WSSync.consumer_release("s2"),
                                    "copy_out2",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["loop_a", "loop_b"],
                                "Consumer": ["loop_b", "loop_a"],
                            },
                        ),
                    ],
                )
            )

            for _i in T.serial(4, annotations={T.WSID: "loop_a"}):
                T.copy(A[0, 0], S1, annotations={T.WSID: "copy_in1"})
                T.copy(S1, F1, annotations={T.WSID: "copy_frag1"})
                T.copy(F1, B[0, 0], annotations={T.WSID: "copy_out1"})
            for _i in T.serial(4, annotations={T.WSID: "loop_b"}):
                T.copy(A[64, 0], S2, annotations={T.WSID: "copy_in2"})
                T.copy(S2, F2, annotations={T.WSID: "copy_frag2"})
                T.copy(F2, B[64, 0], annotations={T.WSID: "copy_out2"})

    with pytest.raises(Exception, match="deadlocks"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_guarded_scope_syncs_use_phase_counter():
    """A guard-skipped iteration produces no cycle but would still
    advance a linearized phase; sync entries inside a guarded scope must
    fall back to runtime phase counters."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(2, threads=128) as bx:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            if bx == 0:
                for i in T.serial(4, annotations={T.WSID: "loop_k"}):
                    T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                    T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    for first_thread in (0, 128):
        _assert_phase_counter(func, _role_body(func, first_thread))


@tilelang.testing.requires_cuda
def test_pipeline_on_private_storage_rejected():
    """Each role has its own physical instance of fragment storage, so a
    pipeline over it could never hand a value across roles."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_frag], depth=2)],
                    scopes=[
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["cp_in"], "Consumer": ["cp_out"]},
                        ),
                    ],
                )
            )
            T.copy(A[0, 0], A_frag, annotations={T.WSID: "cp_in"})
            T.copy(A_frag, B[0, 0], annotations={T.WSID: "cp_out"})

    with pytest.raises(Exception, match="must live in shared or tensor memory"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_duplicate_pipeline_name_rejected():
    """Sync entries reference pipelines by name; two pipelines sharing a
    name would bind both sync streams to the first."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            S1 = T.alloc_shared((64, 64), T.float16)
            S2 = T.alloc_shared((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("buf", [S1], depth=2),
                        T.WSPipeline("buf", [S2], depth=2),
                    ],
                    scopes=[
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["cp"], "Consumer": []}),
                    ],
                )
            )
            T.copy(A[0, 0], B[0, 0], annotations={T.WSID: "cp"})

    with pytest.raises(Exception, match="duplicate pipeline name"):
        _materialize(kernel)


@tilelang.testing.requires_cuda
def test_scope_body_role_never_entering_rejected():
    """A role's body in a scope its parent never enters for that role
    would plan barrier arrivals that are never emitted."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": []}),
                    ],
                )
            )

            for i in T.serial(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    with pytest.raises(Exception, match="never enters it from the parent scope"):
        _materialize(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
