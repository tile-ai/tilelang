"""Unit tests for the MaterializeWSSchedule transform.

Each test builds its input in the form the pass consumes directly: a
static-shape ``@T.prim_func`` kernel (traced at decoration — no torch, no
GPU) prepared with exactly the two passes that define the input contract,
BindTarget and MaterializeKernelLaunch. The target is pinned so the tests
are independent of the machine and of the production pipeline.
"""

import re

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm

_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _materialize(func):
    """Prepare `func` (BindTarget + MaterializeKernelLaunch — the pass's
    input contract) and apply MaterializeWSSchedule; returns the
    transformed PrimFunc."""
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    return mod


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
    script = str(func)
    # Warp-specialized branches on the thread var, in warp order.
    assert 'T.launch_thread("threadIdx.x", 256)' in script
    assert "if tx < 32:" in script
    assert "T.set_max_nreg(40, 0)" in script
    assert "T.set_max_nreg(224, 1)" in script
    # The schedule annotation is consumed.
    assert "ws_schedule" not in script
    # Every scheduling marker is stripped.
    assert "ws_op_id" not in script


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

    script = str(_materialize(kernel))
    assert "T.copy(" in script
    assert script.count("atomic_") >= 3
    assert "ws_schedule" not in script and "ws_op_id" not in script


@tilelang.testing.requires_cuda
def test_buffer_multi_versioning():
    func = _materialize(_smem_pipeline_kernel(depth=3))
    script = str(func)
    # The pipeline buffer gains a leading version dimension of `depth`.
    assert "A_shared = T.sblock_alloc_buffer((3, 64, 64)" in script
    # Accesses index the acquired version (phase % depth).
    assert re.search(r"A_shared\[\(?i(?:_\d+)? % 3\)?, 0, 0\]", script), script


@tilelang.testing.requires_cuda
def test_barrier_counts_tma_and_threads():
    func = _materialize(_smem_pipeline_kernel())
    script = str(func)
    # Producer side: the TMA data rides the tx-count; the per-thread
    # arrive of the producer warp (32 threads) closes the arrival count.
    assert "buf_full: [32, 32]" in script
    # Consumer side: synchronous fragment reads -> all 128 consumer threads.
    assert "buf_empty: [128, 128]" in script


@tilelang.testing.requires_cuda
def test_tma_copy_conversion_carries_barrier():
    func = _materialize(_smem_pipeline_kernel())
    script = str(func)
    # The producer copy is converted to tma_copy with the full barrier
    # attached; the commit is a plain per-thread arrive.
    assert "T.tma_copy(" in script
    assert "barrier=buf_full" in script
    assert "emit_arrive" not in script
    assert "T.ptx_arrive_barrier(buf_full" in script


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
    script = str(func)
    assert "buf_full: [32, 32]" in script
    assert "emit_arrive" not in script
    assert "T.ptx_arrive_barrier(buf_full" in script


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
    script = str(func)
    assert "cp_async_barrier_noinc(buf_full" in script
    assert "T.ptx_arrive_barrier(buf_full" in script
    # One deferred + one plain arrive per producer thread (32 + 32).
    assert "buf_full: [64, 64]" in script


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
    script = str(func)
    assert "no_implicit_async_commit_wait" in script
    assert "cp_async_barrier_noinc(buf_full" in script
    # The deferred arrive is the only producer arrive (32 threads).
    assert "buf_full: [32, 32]" in script
    assert "T.ptx_arrive_barrier(buf_full" not in script


@tilelang.testing.requires_cuda
def test_unsupported_preferred_copy_rejected():
    with pytest.raises(Exception, match='prefer_instruction="tma" conflicts with disable_tma=true'):
        _materialize(_smem_pipeline_kernel(kernel_edit="unsupported_tma_preference"))


@tilelang.testing.requires_cuda
def test_consumer_wait_parity():
    func = _materialize(_smem_pipeline_kernel(depth=2))
    script = str(func)
    # Producer waits for empty with inverted parity; consumer waits for full
    # with plain parity.
    assert "T.mbarrier_wait_parity(buf_empty" in script
    assert "T.mbarrier_wait_parity(buf_full" in script


@tilelang.testing.requires_cuda
def test_uniform_stages_cancel():
    # When ALL of a role's sync stages agree, the offsets cancel and the
    # loop is emitted unshifted (the documented loop-equivalence).
    func = _materialize(_smem_pipeline_kernel(consumer_wait_stage=1, consumer_release_stage=1))
    script = str(func)
    assert "for i in range(4):" in script  # original extent, no guards


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
    script = str(func)
    # The producer's steady state covers iterations [1, 4) and carries no
    # boundary checks; the stage-1 entries address the previous logical
    # iteration.
    assert "for i in range(1, 4):" in script
    assert "i >= 1" not in script and "1 <= i" not in script
    assert "if i < 4" not in script
    # Prologue (the stage-0 entries at iteration 0) precedes the loop;
    # epilogue (the stage-1 entries at iteration 3) follows it.
    producer = script[script.find("T.set_max_nreg(40, 0)") :]
    assert "T.region(C_shared[(i - 1) % 2, 0, 0]" in producer, producer
    prologue = producer.find("T.mbarrier_wait_parity(a_empty_1[0], 1)")
    loop = producer.find("for i in range(1, 4):")
    epilogue = producer.find("T.mbarrier_wait_parity(c_empty_1[1], 0)")
    assert 0 <= prologue < loop < epilogue


@tilelang.testing.requires_cuda
def test_short_static_loop_folds_to_prologue_epilogue():
    """A loop shorter than its stage shift: the unrolled prologue and
    epilogue steps cover every iteration under constant bounds, and the
    pipeline's Simplify pass folds them away together with the empty
    steady-state loop."""
    func = _materialize(_stage_offset_kernel(1))
    mod = tilelang.transform.Simplify()(tvm.IRModule.from_expr(func))
    script = str(mod["main"])
    producer = script[script.find("T.set_max_nreg(40, 0)") : script.find("T.set_max_nreg(224, 1)")]
    # One cycle of each pipeline, no loop, no residual constant guards.
    assert producer.count("T.mbarrier_wait_parity(a_empty") == 1
    assert producer.count("T.mbarrier_wait_parity(c_empty") == 1
    assert "for " not in producer
    assert "T.bool(True)" not in producer


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

            T.fill(A_frag, 0, annotations={T.WSID: "init_frag"})
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                for r, c in T.Parallel(64, 64, annotations={T.WSID: "step_frag"}):
                    A_frag[r, c] = A_frag[r, c] + T.float16(1.0)
            T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_out"})

    func = _materialize(kernel)
    script = str(func)
    # Both roles run their own copy: once in the producer branch, once in
    # the consumer branch — at the root and inside the loop scope alike.
    assert script.count("T.fill(") == 2
    assert script.count("A_frag[r, c] + T.float16(1.0)") == 2
    # Scheduling metadata is fully consumed.
    assert "ws_op_id" not in script


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
    script = str(func)
    # The op is cloned as-is (its own group sync included); the commit is
    # the plain per-thread arrive of the producer warp.
    assert "cp_async_barrier_noinc" not in script
    assert "T.ptx_commit_group()" in script
    assert "T.ptx_wait_group(0)" in script
    assert "T.ptx_arrive_barrier(buf_full" in script
    assert "buf_full: [32, 32]" in script
    assert "buf_empty: [128, 128]" in script
    # The version dimension joins the access_ptr base load.
    assert "B_shared[i % 2, 0]" in script


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
    script = str(func)
    # One guarded loop per role branch.
    assert script.count("if flag[0] > 0:") == 2
    assert "ws_op_id" not in script


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
    script = str(func)
    # One while per role, both on the scheduler state.
    assert script.count("while sched_linear_idx[0] < 4:") == 2
    # The scheduler init/advance runs in both roles.
    assert script.count("sched_current_iter[0] = 0") == 2
    assert script.count("sched_current_iter[0] = sched_current_iter[0] + 1") == 2
    # Phases are runtime counters, bumped at the commit/release entries.
    assert "buf_phase" in script
    assert script.count("buf_phase[0] = buf_phase[0] + 1") == 2
    assert "ws_op_id" not in script


@tilelang.testing.requires_cuda
def test_while_scope_stage_offset_prologue_epilogue():
    """A stage shift under a while scope unrolls into an explicit
    prologue step (the early-stage entries under the loop condition) and
    an epilogue step (the late-stage entries, guarded by the completed
    trip count) instead of per-iteration boundary guards."""
    func = _materialize(_while_scope_kernel(shifted=True))
    script = str(func)
    # The consumer's shifted body: prologue `if cond`, then the while,
    # then the trip-guarded drain step.
    consumer = script[script.find("T.set_max_nreg(224, 1)") :]
    prologue = consumer.find("if sched_linear_idx[0] < 4:")
    kernel_loop = consumer.find("while sched_linear_idx[0] < 4:")
    drain = consumer.find("if 1 <= loop_wave_trips[0]:")
    assert 0 <= prologue < kernel_loop < drain
    assert "loop_wave_trips[0] = loop_wave_trips[0] + 1" in consumer


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
    script = str(func)
    # Both source conditions survive; merging under one guard would delete
    # the `< 1` condition.
    assert "< 1" in script
    assert "< 2" in script


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
    script = str(func)
    # Both statements of the group emitted once, in the consumer branch,
    # with the pipeline read version-rebound.
    consumer = script[script.find("T.set_max_nreg(224, 1)") :]
    assert consumer.count("T.copy(T.region(A_shared[i % 2") == 1, consumer
    assert consumer.count("T.copy(T.region(A_frag[") == 1, consumer
    assert "ws_op_id" not in script


@tilelang.testing.requires_cuda
def test_unannotated_global_store_rejected():
    """A statement whose only effect is a global write must be placed by
    the schedule like any other op; unannotated it would silently vanish
    from every role body."""
    with pytest.raises(Exception, match="carries no ws op id"):
        _materialize(_smem_pipeline_kernel(kernel_edit="unannotated_global_store"))


@tilelang.testing.requires_cuda
def test_kernel_metadata_attrs_preserved():
    """Kernel-level metadata AttrStmts (T.use_swizzle,
    T.annotate_min_blocks_per_sm) wrap the statements that follow them:
    the pass schedules through them and re-wraps the rebuilt body."""
    func = _materialize(_smem_pipeline_kernel(kernel_edit="metadata_attrs"))
    script = str(func)
    swizzle = script.find('"threadblock_swizzle_pattern"')
    min_blocks = script.find('"tl.min_blocks_per_sm"')
    branch = script.find("if tx < 32:")
    assert 0 <= swizzle < min_blocks < branch
    assert "T.tma_copy(" in script


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
    script = str(func)
    # The phase is (i - 3): version indices and parities derive from it, so
    # the raw loop var must never reach a barrier index / parity position.
    assert "- 3) % 2" in script
    assert re.search(r"\bi(_\d+)? % 2", script) is None


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
def test_serial_scope_loop_and_unrolled_rejected():
    """Any serial loop can be a schedule scope (T.Pipelined is a serial
    loop whose num_stages this pass consumes); T.unroll / T.Parallel loops
    cannot."""
    # A plain T.serial scope materializes like a T.Pipelined one.
    func = _materialize(_scope_kind_kernel(T.serial))
    script = str(func)
    assert "T.mbarrier_wait_parity(" in script
    assert '"num_stages"' not in script

    # T.unroll cannot be a scope.
    with pytest.raises(Exception, match="must be a serial loop"):
        _materialize(_scope_kind_kernel(T.unroll))


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
    script = str(func)
    # The unroll survives as a loop inside the consumer branch (op-node
    # loops keep their kind), with the pipeline version index applied to
    # A_shared inside it -- statements inside op-node loops need no ids.
    assert "T.unroll(2)" in script
    assert "A_shared[i % 2, 0, s * 32]" in script


@tilelang.testing.requires_cuda
def test_idle_donor_branch():
    # Roles cover warps [0,1) and [4,8) of 8: warps 1-4 form a gap branch
    # and warps beyond the last role donate registers too.
    func = _materialize(_smem_pipeline_kernel())
    script = str(func)
    # The gap branch donates registers down to the smallest donor budget.
    assert script.count("T.set_max_nreg(40, 0)") >= 2


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

    mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    out = tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"]
    assert tvm.ir.structural_equal(out, mod["main"])


@tilelang.testing.requires_cuda
def test_tcgen05_async_arrive_count():
    """A gemm accumulating into TMEM signals with tcgen05.commit: the full
    barrier of the accumulator pipeline gets arrive count 1, and the smem
    release below the gemm becomes a tcgen05_mma_arrive."""

    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            C_tmem = T.alloc_tmem((64, 64), T.float32)
            C_frag = T.alloc_fragment((64, 64), T.float32)

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
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
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
    script = str(func)
    # The gemm is converted to the async tcgen05 form.
    assert "T.tcgen05_gemm(" in script
    # smem_empty and acc_full are signaled by the tensor core: count 1 each.
    assert "smem_empty: [1, 1]" in script
    assert "acc_full: [1]" in script
    # Synchronous epilogue reads release acc with all 128 threads.
    assert "acc_empty: [128]" in script
    # The MMA-side releases are tcgen05 commits.
    assert "T.tcgen05_mma_arrive(" in script


@tilelang.testing.requires_cuda
def test_runtime_phase_counter_for_multi_depth_spans():
    """A pipeline synchronized at two loop depths by one role needs a
    runtime phase counter instead of the linearized iteration."""

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
                                    # An extra root-level cycle: the producer
                                    # now syncs buf at two loop depths.
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

    func = _materialize(kernel)
    script = str(func)
    # Both roles track their buf phase with a local counter.
    assert "buf_phase" in script


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
        if isinstance(stmt, tvm.tirx.For) and T.WSID in stmt.annotations:
            return tvm.tirx.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                stmt.body,
                stmt.thread_binding,
                stmt.annotations,
                step=tvm.tirx.IntImm("int32", 2),
            )
        return None

    stepped = func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, add_step, ["tirx.For"]))
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
def test_one_sided_scope_cycles_rejected():
    """A pipeline cycling on one side only inside a loop scope (its
    counterpart cycles in the root here) diverges with the loop extent;
    every pipeline must cycle both sides equally often per iteration."""

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
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Producer": ["loop_k"],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                    ],
                )
            )

            for _i in T.serial(4, annotations={T.WSID: "loop_k"}):
                T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_in"})
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
    script = str(func)
    # copy_c runs at stage 1: inside the loop its source guard (i < 3)
    # follows the shifted iteration, and no boundary guard joins it.
    assert re.search(r"if i - 1 < 3", script), script
    assert "i >= 1" not in script and "1 <= i" not in script


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
    script = str(func)
    # Producer branch + idle warps 1..3 of warpgroup 0: both request 80.
    assert script.count("T.set_max_nreg(80, 0)") == 2
    assert script.count("T.set_max_nreg(40, 0)") == 1  # consumer warpgroup


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
    script = str(func)
    # Each kernel's own binding is widened to its own schedule's warps.
    assert 'T.launch_thread("threadIdx.x", 128)' in script
    assert 'T.launch_thread("threadIdx.x", 256)' in script
    assert script.count("T.set_max_nreg(224, 1)") == 2
    assert "ws_schedule" not in script and "ws_op_id" not in script


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


if __name__ == "__main__":
    tilelang.testing.main()
