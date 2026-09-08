"""Tests for AutoSchedule's "role_based" scheduler.

The pass consumes plain (schedule-free) kernels, assigns fixed roles from
lowering eligibility (Load / MMA / Store / Worker), pulls warp-private
def-use chains into their consumers' roles, derives per-storage pipelines
with alternating producer/consumer cycles, and emits a typed ``WSSchedule``
plus ``tl.ws_op_id`` markers. All lowering stays in
``MaterializeWSSchedule``. Kernels the scheduler declines must come back
byte-for-byte unchanged.
"""

import pytest

import tilelang
import tilelang.testing
import torch
from tilelang import language as T
from tilelang import tvm

_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _prepare(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    return mod


def _auto_schedule(mod, scheduler="role_based"):
    """Apply AutoSchedule with the scheduler opted in via pass config."""
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": scheduler}):
        return tilelang.cuda.transform.AutoSchedule()(mod)


def _schedule(func):
    """Run the pass; returns (scheduled module, root WSSchedule or None)."""
    scheduled = _auto_schedule(_prepare(func))
    return scheduled, _root_schedule(scheduled["main"])


def _root_schedule(func):
    result = None

    def visit(node):
        nonlocal result
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "tilelang_root":
            result = node.annotations.get("tl.ws_schedule")

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return result


def _alloc_buffers(mod):
    """name -> Buffer allocated by the scheduled kernel's root block, for
    building expected schedules that structurally match the emitted one."""
    out = {}

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer in node.alloc_buffers:
                out[str(buffer.name)] = buffer

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return out


def _pipelined_load_kernel(num_stages=2):
    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=num_stages):
                T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_tma_load_pipeline():
    """A TMA-eligible global->shared copy becomes the Load role; the shared
    buffer becomes a pipeline of depth num_stages with one cycle per
    iteration."""
    mod, schedule = _schedule(_pipelined_load_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
            ],
            pipelines=[T.WSPipeline("S", [bufs["S"]], depth=2)],
            scopes=[
                T.WSScope(
                    "loop_0",
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("S"),
                            "tl_tileop_copy_1",
                            T.WSSync.consumer_release("S"),
                            "tl_tileop_copy_2",
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("S"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("S"),
                        ],
                    },
                ),
                T.WSScope(T.WSScope.ROOT, {"Worker": ["loop_0"], "Load": ["loop_0"]}),
            ],
        ),
    )


@tilelang.testing.requires_cuda
def test_num_stages_sets_pipeline_depth():
    mod, schedule = _schedule(_pipelined_load_kernel(num_stages=3))
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(schedule.pipelines[0], T.WSPipeline("S", [bufs["S"]], depth=3))
    assert len(schedule.pipelines) == 1


def _two_cycle_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((2, 64, 64), T.float16), B: T.Tensor((2, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F0 = T.alloc_fragment((64, 64), T.float16)
            F1 = T.alloc_fragment((64, 64), T.float16)
            T.copy(A[0, 0, 0], S)
            T.copy(S, F0)
            T.copy(A[1, 0, 0], S)
            T.copy(S, F1)
            T.copy(F0, B[0, 0, 0])
            T.copy(F1, B[1, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_two_cycles_on_one_buffer_and_root_depth():
    """Reusing one shared buffer twice yields two bracket pairs — a single
    bracket around both loads would let the second overwrite unread data —
    and a root-level handoff gets depth 1."""
    mod, schedule = _schedule(_two_cycle_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
            ],
            pipelines=[T.WSPipeline("S", [bufs["S"]], depth=1)],
            scopes=[
                T.WSScope(
                    T.WSScope.ROOT,
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("S"),
                            "tl_tileop_copy_1",
                            T.WSSync.consumer_release("S"),
                            T.WSSync.consumer_wait("S"),
                            "tl_tileop_copy_3",
                            T.WSSync.consumer_release("S"),
                            "tl_tileop_copy_4",
                            "tl_tileop_copy_5",
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("S"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("S"),
                            T.WSSync.producer_acquire("S"),
                            "tl_tileop_copy_2",
                            T.WSSync.producer_commit("S"),
                        ],
                    },
                ),
            ],
        ),
    )


def _gather_kernel(worker_uses_index):
    @T.prim_func
    def kernel(
        Indices: T.Tensor((4,), T.int32),
        A: T.Tensor((8, 64, 64), T.float16),
        B: T.Tensor((8, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                idx = Indices[k]
                T.copy(A[idx, 0, 0], S)
                T.copy(S, F)
                if worker_uses_index:
                    T.copy(F, B[idx, 0, 0])
                else:
                    T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_bind_placed_with_its_single_consumer_role():
    """A pure Bind read only by the TMA copy moves into the Load role."""
    _, schedule = _schedule(_gather_kernel(worker_uses_index=False))
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "loop_0",
            {
                "Worker": [
                    T.WSSync.consumer_wait("S"),
                    "tl_tileop_copy_1",
                    T.WSSync.consumer_release("S"),
                    "tl_tileop_copy_2",
                ],
                "Load": [
                    "idx_0",
                    T.WSSync.producer_acquire("S"),
                    "tl_tileop_copy_0",
                    T.WSSync.producer_commit("S"),
                ],
            },
        ),
    )


@tilelang.testing.requires_cuda
def test_bind_duplicated_into_two_roles():
    """A pure global-reading Bind used by two roles is placed in both; the
    materializer re-emits it per role."""
    _, schedule = _schedule(_gather_kernel(worker_uses_index=True))
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "loop_0",
            {
                "Worker": [
                    "idx_0",
                    T.WSSync.consumer_wait("S"),
                    "tl_tileop_copy_1",
                    T.WSSync.consumer_release("S"),
                    "tl_tileop_copy_2",
                ],
                "Load": [
                    "idx_0",
                    T.WSSync.producer_acquire("S"),
                    "tl_tileop_copy_0",
                    T.WSSync.producer_commit("S"),
                ],
            },
        ),
    )


def _local_chain_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            tmp = T.alloc_local((1,), T.int32)
            for k in T.Pipelined(4, num_stages=2):
                tmp[0] = k * 2 % 4
                idx = tmp[0]
                T.copy(A[idx, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_bind_chain_through_local_buffer_moves_together():
    """The TMA index Bind and the local store feeding it must land in the
    Load role together: moving only the Bind would make the TMA warp read
    another warp's private register."""
    _, schedule = _schedule(_local_chain_kernel())
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "loop_0",
            {
                "Worker": [
                    T.WSSync.consumer_wait("S"),
                    "tl_tileop_copy_1",
                    T.WSSync.consumer_release("S"),
                    "tl_tileop_copy_2",
                ],
                "Load": [
                    "tmp_0",
                    "idx_0",
                    T.WSSync.producer_acquire("S"),
                    "tl_tileop_copy_0",
                    T.WSSync.producer_commit("S"),
                ],
            },
        ),
    )


def _local_accumulator_gemm(threads=128):
    @T.prim_func
    def kernel(
        A: T.Tensor((64, 128), T.float16),
        B: T.Tensor((128, 64), T.float16),
        C: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((64, 32), T.float16)
            B_shared = T.alloc_shared((32, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[0, k * 32], A_shared)
                T.copy(B[k * 32, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_worker_gemm_topology_materializes():
    mod, schedule = _schedule(_local_accumulator_gemm())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    # One pipeline per storage (merging is a TODO).
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
            ],
            pipelines=[
                T.WSPipeline("A_shared", [bufs["A_shared"]], depth=2),
                T.WSPipeline("B_shared", [bufs["B_shared"]], depth=2),
            ],
            scopes=[
                T.WSScope(
                    "loop_0",
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("A_shared"),
                            T.WSSync.consumer_wait("B_shared"),
                            "tl_tileop_gemm_0",
                            T.WSSync.consumer_release("A_shared"),
                            T.WSSync.consumer_release("B_shared"),
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("A_shared"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("A_shared"),
                            T.WSSync.producer_acquire("B_shared"),
                            "tl_tileop_copy_1",
                            T.WSSync.producer_commit("B_shared"),
                        ],
                    },
                ),
                T.WSScope(
                    T.WSScope.ROOT,
                    {
                        "Worker": ["tl_tileop_fill_0", "loop_0", "tl_tileop_copy_2"],
                        "Load": ["loop_0"],
                    },
                ),
            ],
        ),
    )

    script = str(tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"])
    assert 'T.launch_thread("threadIdx.x", 256)' in script
    assert "T.tma_copy" in script
    assert "ws_schedule" not in script


def _tmem_gemm():
    @T.prim_func
    def kernel(
        A: T.Tensor((2, 128, 64), T.float16),
        B: T.Tensor((2, 128, 64), T.float16),
        C: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_shared = T.alloc_shared((128, 64), T.float16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            C_frag = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.float16)

            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[k, 0, 0], A_shared)
                T.copy(B[k, 0, 0], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    clear_accum=k == 0,
                )
            for i in T.serial(2):
                T.copy(C_tmem[:, i * 64 : (i + 1) * 64], C_frag)
                T.copy(C_frag, C_shared)
                T.copy(C_shared, C[:, i * 64 : (i + 1) * 64])

    return kernel


@tilelang.testing.requires_cuda
def test_tmem_gemm_four_role_topology():
    """tcgen05 GEMM -> MMA role, TMA store -> Store role; the accumulator
    pipeline brackets the k loop at the root with depth 1, and the epilogue
    staging buffer cycles inside the epilogue loop."""
    mod, schedule = _schedule(_tmem_gemm())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=232),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
                T.WSRole("MMA", warps_lo=5, warps_hi=6, max_nreg=24),
                T.WSRole("Store", warps_lo=6, warps_hi=7, max_nreg=24),
            ],
            pipelines=[
                T.WSPipeline("A_shared", [bufs["A_shared"]], depth=2),
                T.WSPipeline("B_shared", [bufs["B_shared"]], depth=2),
                T.WSPipeline("C_tmem", [bufs["C_tmem"]], depth=1),
                T.WSPipeline("C_shared", [bufs["C_shared"]], depth=1),
            ],
            scopes=[
                T.WSScope(
                    "loop_1",
                    {
                        "Worker": [
                            "tl_tileop_copy_2",
                            T.WSSync.producer_acquire("C_shared"),
                            "tl_tileop_copy_3",
                            T.WSSync.producer_commit("C_shared"),
                        ],
                        "Store": [
                            T.WSSync.consumer_wait("C_shared"),
                            "tl_tileop_copy_4",
                            T.WSSync.consumer_release("C_shared"),
                        ],
                    },
                ),
                T.WSScope(
                    "loop_0",
                    {
                        "Load": [
                            T.WSSync.producer_acquire("A_shared"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("A_shared"),
                            T.WSSync.producer_acquire("B_shared"),
                            "tl_tileop_copy_1",
                            T.WSSync.producer_commit("B_shared"),
                        ],
                        "MMA": [
                            T.WSSync.consumer_wait("A_shared"),
                            T.WSSync.consumer_wait("B_shared"),
                            "tl_tileop_gemm_0",
                            T.WSSync.consumer_release("A_shared"),
                            T.WSSync.consumer_release("B_shared"),
                        ],
                    },
                ),
                T.WSScope(
                    T.WSScope.ROOT,
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("C_tmem"),
                            "loop_1",
                            T.WSSync.consumer_release("C_tmem"),
                        ],
                        "Load": ["loop_0"],
                        "MMA": [
                            T.WSSync.producer_acquire("C_tmem"),
                            "loop_0",
                            T.WSSync.producer_commit("C_tmem"),
                        ],
                        "Store": ["loop_1"],
                    },
                ),
            ],
        ),
    )

    script = str(tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"])
    assert "tcgen05_gemm" in script
    # The Store role reads C_shared through the async proxy; the handoff from
    # the Worker role's generic-proxy writes needs an explicit fence.
    assert "fence_proxy_async" in script
    assert "ws_schedule" not in script


def _tmem_gemm_unrolled_epilogue():
    """_tmem_gemm with the epilogue loop unrolled instead of serial."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 128, 64), T.float16),
        B: T.Tensor((2, 128, 64), T.float16),
        C: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_shared = T.alloc_shared((128, 64), T.float16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            C_frag = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.float16)

            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[k, 0, 0], A_shared)
                T.copy(B[k, 0, 0], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    clear_accum=k == 0,
                )
            for i in T.unroll(2):
                T.copy(C_tmem[:, i * 64 : (i + 1) * 64], C_frag)
                T.copy(C_frag, C_shared)
                T.copy(C_shared, C[:, i * 64 : (i + 1) * 64])

    return kernel


@tilelang.testing.requires_cuda
def test_unrolled_epilogue_is_scheduled():
    """Sequential loops are scopes regardless of kind: the unrolled
    epilogue schedules exactly like the serial one — the TMA store gets
    the Store role and the staging buffer cycles inside the epilogue
    scope."""
    _, schedule = _schedule(_tmem_gemm_unrolled_epilogue())
    assert schedule is not None
    # The epilogue scope is the unrolled loop; its schedule matches the
    # serial one's.
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "unroll_0",
            {
                "Worker": [
                    "tl_tileop_copy_2",
                    T.WSSync.producer_acquire("C_shared"),
                    "tl_tileop_copy_3",
                    T.WSSync.producer_commit("C_shared"),
                ],
                "Store": [
                    T.WSSync.consumer_wait("C_shared"),
                    "tl_tileop_copy_4",
                    T.WSSync.consumer_release("C_shared"),
                ],
            },
        ),
    )
    tvm.ir.assert_structural_equal(
        schedule.scopes[2],
        T.WSScope(
            T.WSScope.ROOT,
            {
                "Worker": [
                    T.WSSync.consumer_wait("C_tmem"),
                    "unroll_0",
                    T.WSSync.consumer_release("C_tmem"),
                ],
                "Load": ["loop_0"],
                "MMA": [
                    T.WSSync.producer_acquire("C_tmem"),
                    "loop_0",
                    T.WSSync.producer_commit("C_tmem"),
                ],
                "Store": ["unroll_0"],
            },
        ),
    )


def _guarded_producer_kernel():
    """The producer write is skipped on the last iteration: under
    multi-versioning the consumer would see the slot from `depth`
    iterations ago instead of the previous iteration's value."""

    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                if k < 3:
                    T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_guarded_write_into_versioned_pipeline_declines():
    """A guarded write into a multi-versioned pipeline is declined rather
    than silently re-versioned; guarded writes to single-buffered
    pipelines (FA's rescale) keep source semantics and stay schedulable."""
    prepared = _prepare(_guarded_producer_kernel())
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


def _rmw_accumulator_kernel():
    """A flash-attention-shaped accumulator: the Worker rescales C_tmem
    before the MMA role accumulates onto it, and reads it once more after
    the loop."""

    @T.prim_func
    def kernel(
        A: T.Tensor((4, 128, 64), T.float16),
        B: T.Tensor((4, 64, 64), T.float16),
        Scale: T.Tensor((4,), T.float32),
        C: T.Tensor((128, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_shared = T.alloc_shared((64, 64), T.float16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            C_frag = T.alloc_fragment((128, 64), T.float32)

            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, 0, 0], A_shared)
                T.copy(B[k, 0, 0], B_shared)
                if k > 0:
                    T.copy(C_tmem, C_frag)
                    for i, j in T.Parallel(128, 64):
                        C_frag[i, j] *= Scale[k]
                    T.copy(C_frag, C_tmem)
                T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, clear_accum=k == 0)
            T.copy(C_tmem, C_frag)
            T.copy(C_frag, C)

    return kernel


@tilelang.testing.requires_cuda
def test_rmw_accumulator_nested_pipelines():
    """A read-modify-write storage used at two loop depths is bound to two
    strictly nested pipelines: inside the loop, ownership alternates from
    the first-touching role (the Worker's rescale precedes the MMA's
    accumulate) and stays single-buffered despite the 2-stage loop; at the
    root, the loop projects one MMA-side touch — the MMA producer-brackets
    the whole loop and the post-loop read is a plain consumer run, with no
    drain pair anywhere."""
    mod, schedule = _schedule(_rmw_accumulator_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
                T.WSRole("MMA", warps_lo=5, warps_hi=6, max_nreg=24),
            ],
            pipelines=[
                T.WSPipeline("A_shared", [bufs["A_shared"]], depth=2),
                T.WSPipeline("B_shared", [bufs["B_shared"]], depth=2),
                T.WSPipeline("C_tmem", [bufs["C_tmem"]], depth=1),
                T.WSPipeline("C_tmem_root", [bufs["C_tmem"]], depth=1),
            ],
            scopes=[
                T.WSScope(
                    "loop_0",
                    {
                        "Worker": [
                            T.WSSync.producer_acquire("C_tmem"),
                            "tl_tileop_copy_2",
                            "parallel_0",
                            "tl_tileop_copy_3",
                            T.WSSync.producer_commit("C_tmem"),
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("A_shared"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("A_shared"),
                            T.WSSync.producer_acquire("B_shared"),
                            "tl_tileop_copy_1",
                            T.WSSync.producer_commit("B_shared"),
                        ],
                        "MMA": [
                            T.WSSync.consumer_wait("A_shared"),
                            T.WSSync.consumer_wait("B_shared"),
                            T.WSSync.consumer_wait("C_tmem"),
                            "tl_tileop_gemm_0",
                            T.WSSync.consumer_release("A_shared"),
                            T.WSSync.consumer_release("B_shared"),
                            T.WSSync.consumer_release("C_tmem"),
                        ],
                    },
                ),
                T.WSScope(
                    T.WSScope.ROOT,
                    {
                        "Worker": [
                            "loop_0",
                            T.WSSync.consumer_wait("C_tmem_root"),
                            "tl_tileop_copy_4",
                            T.WSSync.consumer_release("C_tmem_root"),
                            "tl_tileop_copy_5",
                        ],
                        "Load": ["loop_0"],
                        "MMA": [
                            T.WSSync.producer_acquire("C_tmem_root"),
                            "loop_0",
                            T.WSSync.producer_commit("C_tmem_root"),
                        ],
                    },
                ),
            ],
        ),
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_rmw_accumulator_numerical():
    kernel = tilelang.compile(
        _rmw_accumulator_kernel(),
        target="cuda",
        out_idx=[3],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((4, 128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((4, 64, 64), device="cuda", dtype=torch.float16)
    scale = torch.rand((4,), device="cuda", dtype=torch.float32) + 0.5
    actual = kernel(a, b, scale)
    expected = torch.zeros((128, 64), device="cuda", dtype=torch.float32)
    for k in range(4):
        if k > 0:
            expected *= scale[k]
        expected += a[k].float() @ b[k].float().T
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def _post_loop_read_kernel():
    """The Load fills S inside loop_k; the Worker reads it there AND once
    more after the loop. The collapsed loop_k also WRITES S, so ownership
    moves across loop_w iterations: the subtree flips to the Load side of
    an outer pipeline whose empty side keeps the next wave's fill off S
    until the post-loop read released it."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 4, 64, 64), T.float16),
        B: T.Tensor((2, 4, 64, 64), T.float16),
        C: T.Tensor((2, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            G = T.alloc_fragment((64, 64), T.float16)
            for w in T.Pipelined(2, num_stages=2):
                for k in T.Pipelined(4, num_stages=2):
                    T.copy(A[w, k, 0, 0], S)
                    T.copy(S, F)
                    T.copy(F, B[w, k, 0, 0])
                T.copy(S, G)
                T.copy(G, C[w, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_post_loop_read_forms_outer_pipeline():
    """Nested bindings from a same-role sibling: inside loop_k the Load
    hands S to the Worker; at loop_w the subtree hands the whole loop's S
    to the post-loop read. S carries no loop-carried dependency at either
    level, so BOTH bindings take their loops' num_stages; the post-loop
    read resolves the inner binding to its last-completed version."""
    mod, schedule = _schedule(_post_loop_read_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
            ],
            pipelines=[
                T.WSPipeline("S", [bufs["S"]], depth=2),
                T.WSPipeline("S_loop_1", [bufs["S"]], depth=2),  # wave ring
            ],
            scopes=[
                T.WSScope(
                    "loop_0",
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("S"),
                            "tl_tileop_copy_1",
                            T.WSSync.consumer_release("S"),
                            "tl_tileop_copy_2",
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("S"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("S"),
                        ],
                    },
                ),
                T.WSScope(
                    "loop_1",
                    {
                        "Worker": [
                            "loop_0",
                            T.WSSync.consumer_wait("S_loop_1"),
                            "tl_tileop_copy_3",
                            T.WSSync.consumer_release("S_loop_1"),
                            "tl_tileop_copy_4",
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("S_loop_1"),
                            "loop_0",
                            T.WSSync.producer_commit("S_loop_1"),
                        ],
                    },
                ),
                T.WSScope(T.WSScope.ROOT, {"Worker": ["loop_1"], "Load": ["loop_1"]}),
            ],
        ),
    )


@tilelang.testing.requires_cuda
def test_read_before_nested_cycle_declines():
    """Reading S BEFORE the loop that refills it: the Worker's wait would
    pair a commit that only happens later in the same wave — the ownership
    walk declines instead of emitting a deadlocking schedule."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 4, 64, 64), T.float16),
        B: T.Tensor((2, 4, 64, 64), T.float16),
        C: T.Tensor((2, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            G = T.alloc_fragment((64, 64), T.float16)
            for w in T.Pipelined(2, num_stages=1):
                T.copy(S, G)
                T.copy(G, C[w, 0, 0])
                for k in T.Pipelined(4, num_stages=1):
                    T.copy(A[w, k, 0, 0], S)
                    T.copy(S, F)
                    T.copy(F, B[w, k, 0, 0])

    prepared = _prepare(kernel)
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_post_loop_read_numerical():
    kernel = tilelang.compile(
        _post_loop_read_kernel(),
        target="cuda",
        out_idx=[1, 2],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    b, c = kernel(a)
    torch.testing.assert_close(b, a)
    torch.testing.assert_close(c, a[:, 3])


@tilelang.testing.requires_cuda
def test_pipeline_opt_in_runs_automatic_ws():
    func = _local_accumulator_gemm().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    with (
        _TARGET,
        tvm.transform.PassContext(config={"tl.enable_auto_schedule": "role_based"}),
    ):
        out = tilelang.cuda.pipeline.CUDAPassPipelineBodyPrologue(mod, _TARGET)["main"]

    script = str(out)
    assert 'T.launch_thread("threadIdx.x", 256)' in script
    assert "ws_schedule" not in script
    assert "ws_op_id" not in script


@tilelang.testing.requires_cuda
def test_no_shared_handoff_is_left_unchanged():
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            F = T.alloc_fragment((64, 64), T.float16)
            T.copy(A, F)
            T.copy(F, B)

    prepared = _prepare(kernel)
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


def _persistent_while_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((2, 64, 64), T.float16), B: T.Tensor((2, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            wave = T.alloc_local((1,), T.int32)
            wave[0] = 0
            while wave[0] < 2:
                T.copy(A[wave[0], 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[wave[0], 0, 0])
                wave[0] = wave[0] + 1

    return kernel


@tilelang.testing.requires_cuda
def test_while_scope_is_scheduled():
    """A while loop is a scope: every role re-evaluates the condition on
    its own duplicated wave counter, and the shared buffer cycles inside
    the while body (runtime phase counters in the materializer)."""
    mod, schedule = _schedule(_persistent_while_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    # The wave bump ("wave_1") is duplicated into both roles.
    tvm.ir.assert_structural_equal(
        schedule,
        T.WSSchedule(
            num_warps=8,
            roles=[
                T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=104),
                T.WSRole("Load", warps_lo=4, warps_hi=5, max_nreg=24),
            ],
            pipelines=[T.WSPipeline("S", [bufs["S"]], depth=1)],
            scopes=[
                T.WSScope(
                    "while_0",
                    {
                        "Worker": [
                            T.WSSync.consumer_wait("S"),
                            "tl_tileop_copy_1",
                            T.WSSync.consumer_release("S"),
                            "tl_tileop_copy_2",
                            "wave_1",
                        ],
                        "Load": [
                            T.WSSync.producer_acquire("S"),
                            "tl_tileop_copy_0",
                            T.WSSync.producer_commit("S"),
                            "wave_1",
                        ],
                    },
                ),
                T.WSScope(
                    T.WSScope.ROOT,
                    {"Worker": ["wave_0", "while_0"], "Load": ["wave_0", "while_0"]},
                ),
            ],
        ),
    )


@tilelang.testing.requires_cuda
def test_while_scope_numerical():
    kernel = tilelang.compile(
        _persistent_while_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((2, 64, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
def test_pipelined_load_numerical():
    kernel = tilelang.compile(
        _pipelined_load_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((4, 64, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
def test_two_cycles_numerical():
    kernel = tilelang.compile(
        _two_cycle_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((2, 64, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
def test_gather_bind_numerical():
    kernel = tilelang.compile(
        _gather_kernel(worker_uses_index=True),
        target="cuda",
        out_idx=[2],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    indices = torch.tensor([5, 2, 7, 0], device="cuda", dtype=torch.int32)
    a = torch.randn((8, 64, 64), device="cuda", dtype=torch.float16)
    actual = kernel(indices, a)
    # Rows outside `indices` are never written (the output arrives
    # uninitialized), so compare the gathered rows only.
    torch.testing.assert_close(actual[indices.long()], a[indices.long()])


@tilelang.testing.requires_cuda
def test_local_chain_numerical():
    kernel = tilelang.compile(
        _local_chain_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((4, 64, 64), device="cuda", dtype=torch.float16)
    expected = torch.stack([a[k * 2 % 4] for k in range(4)])
    torch.testing.assert_close(kernel(a), expected)


@tilelang.testing.requires_cuda
def test_worker_gemm_numerical():
    kernel = tilelang.compile(
        _local_accumulator_gemm(),
        target="cuda",
        out_idx=[2],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a, b), a @ b, rtol=1e-2, atol=1e-2)


def _two_loop_kernel():
    """The same shared buffer streams through two sequential pipelined
    loops with different num_stages."""

    @T.prim_func
    def kernel(
        A: T.Tensor((4, 64, 64), T.float16),
        B: T.Tensor((4, 64, 64), T.float16),
        C: T.Tensor((4, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])
            for k in T.Pipelined(4, num_stages=3):
                T.copy(A[3 - k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, C[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_storage_cycling_in_sibling_loops_declines():
    """A pipeline synchronizes exactly one scope, so a buffer streaming
    through two SIBLING loops would need two pipelines whose barriers
    nothing chains — the second loop's pre-armed acquire could overwrite
    data the first loop's consumer still reads. The kernel declines."""
    prepared = _prepare(_two_loop_kernel())
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


def _non_tma_layout_kernel():
    @T.prim_func
    def kernel(
        A: T.Tensor((4, 64, 64), T.float16),
        B: T.Tensor((4, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            S_perm = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            F_perm = T.alloc_fragment((64, 64), T.float16)
            # Additive row rotation is not GF(2)-linear, so the layout is
            # not TMA-expressible and the copy cannot be a TMA producer.
            T.annotate_layout({S_perm: T.Layout((64, 64), lambda i, j: [i, (i + j) % 64])})
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, 0, 0], S)
                T.copy(A[k, 0, 0], S_perm)
                T.copy(S, F)
                T.copy(S_perm, F_perm)
                for i, j in T.Parallel(64, 64):
                    F[i, j] += F_perm[i, j]
                T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_non_tma_layout_copy_stays_worker():
    """A TMA-shaped copy whose destination has a non-TMA-expressible layout
    must not become a Load: lowering would fall back to a normal copy the
    Load warp cannot signal as a transaction. It stays a Worker op and its
    buffer stays out of the pipelines."""
    mod, schedule = _schedule(_non_tma_layout_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    # S_perm stays out of the pipelines; the Worker keeps its copy.
    tvm.ir.assert_structural_equal(schedule.pipelines[0], T.WSPipeline("S", [bufs["S"]], depth=2))
    assert len(schedule.pipelines) == 1
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "loop_0",
            {
                "Worker": [
                    "tl_tileop_copy_1",
                    T.WSSync.consumer_wait("S"),
                    "tl_tileop_copy_2",
                    T.WSSync.consumer_release("S"),
                    "tl_tileop_copy_3",
                    "parallel_0",
                    "tl_tileop_copy_4",
                ],
                "Load": [
                    T.WSSync.producer_acquire("S"),
                    "tl_tileop_copy_0",
                    T.WSSync.producer_commit("S"),
                ],
            },
        ),
    )


@tilelang.testing.requires_cuda
def test_non_tma_layout_numerical():
    kernel = tilelang.compile(
        _non_tma_layout_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((4, 64, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a + a)


def _cp_async_load_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.async_copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_cp_async_copy_is_a_load_producer():
    """A cp.async tile-op copy is asynchronous — the mbarrier tracks its
    completion through the commit entry's deferred cp.async.mbarrier.arrive
    — so it leaves the workers like a TMA copy."""
    mod, schedule = _schedule(_cp_async_load_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(schedule.pipelines[0], T.WSPipeline("S", [bufs["S"]], depth=2))
    tvm.ir.assert_structural_equal(
        schedule.scopes[0],
        T.WSScope(
            "loop_0",
            {
                "Worker": [
                    T.WSSync.consumer_wait("S"),
                    "tl_tileop_copy_0",
                    T.WSSync.consumer_release("S"),
                    "tl_tileop_copy_1",
                ],
                "Load": [
                    T.WSSync.producer_acquire("S"),
                    "tl_tileop_async_copy_0",
                    T.WSSync.producer_commit("S"),
                ],
            },
        ),
    )

    script = str(tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"])
    assert "ptx_commit_group" in script
    assert "ptx_cp_async_barrier_noinc" in script


@tilelang.testing.requires_cuda
def test_cp_async_load_numerical():
    kernel = tilelang.compile(
        _cp_async_load_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((4, 64, 64), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
def test_async_wgmma_wait_kernel_is_left_unchanged():
    """T.wgmma_gemm (wg_wait = -1) returns with MMAs still reading its
    shared operands; the manual T.wait_wgmma does not touch the buffers, so
    the scheduler would release them early. Rejected until delayed wgmma
    waits are supported."""

    @T.prim_func
    def kernel(
        A: T.Tensor((64, 128), T.float16),
        B: T.Tensor((128, 64), T.float16),
        C: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 32), T.float16)
            B_shared = T.alloc_shared((32, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[0, k * 32], A_shared)
                T.copy(B[k * 32, 0], B_shared)
                T.wgmma_gemm(A_shared, B_shared, C_local)
            T.wait_wgmma(0)
            T.copy(C_local, C[0, 0])

    prepared = _prepare(kernel)
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


@tilelang.testing.requires_cuda
def test_raw_cp_async_kernel_is_left_unchanged():
    """A bare ptx_cp_async may rely on the legacy pass to synthesize its
    completion protocol; the scheduler backs off rather than emit a read of
    unlanded data."""

    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 16), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            B_shared = T.alloc_shared((16,), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.ptx_cp_async(
                    T.access_ptr(B_shared[0], "w", 16),
                    T.access_ptr(B[k, 0], "r", 16),
                    16,
                )
                T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, A[k, 0, 0])

    prepared = _prepare(kernel)
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


def _pointer_table_kernel():
    """The TMA source is a make_tensor view over a device pointer table: the
    handle Bind is scheduled into the Load role and its buffer must follow
    the SSA-freshened var."""

    @T.prim_func
    def kernel(src_ptrs: T.Tensor((1,), T.ptr), out: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            src = T.make_tensor(src_ptrs[0], (64, 64), T.float16)
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for _k in T.Pipelined(2, num_stages=2):
                T.copy(src, S)
                T.copy(S, F)
                T.copy(F, out)

    return kernel


@tilelang.testing.requires_cuda
def test_pointer_table_bind_follows_freshened_buffer():
    mod, schedule = _schedule(_pointer_table_kernel())
    assert schedule is not None
    script = str(tilelang.cuda.transform.MaterializeWSSchedule()(mod)["main"])
    assert "ws_schedule" not in script

    kernel = tilelang.compile(
        _pointer_table_kernel(),
        target="cuda",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    src = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    ptrs = torch.tensor([src.data_ptr()], device="cuda", dtype=torch.int64)
    torch.testing.assert_close(kernel(ptrs), src)


@tilelang.testing.requires_cuda
def test_unknown_scheduler_rejected():
    with pytest.raises(Exception, match="unknown auto-schedule scheduler"):
        _auto_schedule(_prepare(_pipelined_load_kernel()), scheduler="nonexistent")


@tilelang.testing.requires_cuda
def test_unschedulable_constructs_decline():
    """PreprocessIR checks the schedulability contract — atomics,
    asynchronous producers nested inside opaque ops, hand-written
    synchronization — and declines the kernel with a warning, leaving it
    byte-for-byte unchanged, like a scheduler decline."""

    @T.prim_func
    def atomic_kernel(A: T.Tensor((64, 64), T.float32), B: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.copy(A, S)
            for i, j in T.Parallel(64, 64):
                T.atomic_add(B[i, j], S[i, j])

    @T.prim_func
    def hosted_async_kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            with T.ws_op("staged"):  # opaque wrapper hosting an async g2s copy
                T.copy(A, S)
                T.copy(S, F)
            T.copy(F, B)

    @T.prim_func
    def sync_threads_kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            T.copy(A, S)
            # A block-wide barrier cannot be duplicated into role branches.
            T.sync_threads()
            T.copy(S, B)

    for kernel in (atomic_kernel, hosted_async_kernel, sync_threads_kernel):
        prepared = _prepare(kernel)
        scheduled = _auto_schedule(prepared)
        assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_tmem_gemm_numerical():
    kernel = tilelang.compile(
        _tmem_gemm(),
        target="cuda",
        out_idx=[2],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((2, 128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((2, 128, 64), device="cuda", dtype=torch.float16)
    actual = kernel(a, b)
    expected = sum(a[k].float() @ b[k].float().T for k in range(2)).half()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
def test_thread_budget_exceeded_declines():
    """Issuer warps are appended after the workers: a 1024-thread kernel
    would exceed the block-size limit, so the scheduler declines instead
    of emitting an unlaunchable kernel."""

    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=1024):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    prepared = _prepare(kernel)
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


def _waved_rmw_accumulator_kernel(depth=None, inner_depth=None):
    """The RMW accumulator inside a wave loop: each wave clears C_tmem
    (clear_accum at k == 0) and reads it once after its k loop. The
    per-wave clear is exactly the recurrence reset that
    T.annotate_ws_pipeline_depth asserts."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 4, 128, 64), T.float16),
        B: T.Tensor((2, 4, 64, 64), T.float16),
        Scale: T.Tensor((4,), T.float32),
        C: T.Tensor((2, 128, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_shared = T.alloc_shared((64, 64), T.float16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            C_frag = T.alloc_fragment((128, 64), T.float32)
            for w in T.Pipelined(2, num_stages=1):
                if depth is not None:
                    T.annotate_ws_pipeline_depth({C_tmem: depth})
                for k in T.Pipelined(4, num_stages=2):
                    if inner_depth is not None:
                        T.annotate_ws_pipeline_depth({C_tmem: inner_depth})
                    T.copy(A[w, k, 0, 0], A_shared)
                    T.copy(B[w, k, 0, 0], B_shared)
                    if k > 0:
                        T.copy(C_tmem, C_frag)
                        for i, j in T.Parallel(128, 64):
                            C_frag[i, j] *= Scale[k]
                        T.copy(C_frag, C_tmem)
                    T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, clear_accum=k == 0)
                T.copy(C_tmem, C_frag)
                T.copy(C_frag, C[w, 0, 0])

    return kernel


@tilelang.testing.requires_cuda
def test_annotated_ws_pipeline_depth_overrides_num_stages():
    """T.annotate_ws_pipeline_depth overrides the depth the enclosing
    scope's num_stages would give its pipeline."""

    @T.prim_func
    def kernel(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.annotate_ws_pipeline_depth({S: 3})
                T.copy(A[k, 0, 0], S)
                T.copy(S, F)
                T.copy(F, B[k, 0, 0])

    mod, schedule = _schedule(kernel)
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    tvm.ir.assert_structural_equal(schedule.pipelines[0], T.WSPipeline("S", [bufs["S"]], depth=3))
    assert len(schedule.pipelines) == 1


@tilelang.testing.requires_cuda
def test_annotated_ws_pipeline_depth_ring_accumulator():
    """Without the annotation the accumulator's wave-scope binding is
    pinned single-buffered (the scheduler cannot prove the per-wave
    clear); annotating the wave scope asserts it, giving a ring — and
    the guarded rescale inside the INNER binding must not veto it (a
    skipped write cycles the inner slot, not the ring's)."""
    mod, schedule = _schedule(_waved_rmw_accumulator_kernel())
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    # Inner alternation and outer binding both pinned single-buffered.
    tvm.ir.assert_structural_equal(schedule.pipelines[2], T.WSPipeline("C_tmem", [bufs["C_tmem"]], depth=1))
    tvm.ir.assert_structural_equal(schedule.pipelines[3], T.WSPipeline("C_tmem_loop_1", [bufs["C_tmem"]], depth=1))

    mod, schedule = _schedule(_waved_rmw_accumulator_kernel(depth=2))
    assert schedule is not None
    bufs = _alloc_buffers(mod)
    # The inner binding stays pinned; the annotation gives the wave ring.
    tvm.ir.assert_structural_equal(schedule.pipelines[2], T.WSPipeline("C_tmem", [bufs["C_tmem"]], depth=1))
    tvm.ir.assert_structural_equal(schedule.pipelines[3], T.WSPipeline("C_tmem_loop_1", [bufs["C_tmem"]], depth=2))


@tilelang.testing.requires_cuda
def test_annotated_ws_pipeline_depth_on_inner_binding_declines():
    """The waved accumulator's inner binding has a GUARDED writer (the
    rescale): versioning it would expose a stale slot, so the annotated
    depth declines."""
    prepared = _prepare(_waved_rmw_accumulator_kernel(inner_depth=2))
    scheduled = _auto_schedule(prepared)
    assert tvm.ir.structural_equal(scheduled["main"], prepared["main"])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_annotated_ws_pipeline_depth_ring_numerical():
    kernel = tilelang.compile(
        _waved_rmw_accumulator_kernel(depth=2),
        target="cuda",
        out_idx=[3],
        pass_configs={"tl.enable_auto_schedule": "role_based"},
    )
    a = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    scale = torch.rand((4,), device="cuda", dtype=torch.float32) + 0.5
    actual = kernel(a, b, scale)
    expected = torch.zeros((2, 128, 64), device="cuda", dtype=torch.float32)
    for w in range(2):
        for k in range(4):
            if k > 0:
                expected[w] *= scale[k]
            expected[w] += a[w, k].float() @ b[w, k].float().T
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
def test_seamless_sibling_bindings_rejected_by_materializer():
    """Two sibling loops hand the accumulator back and forth seamlessly
    (MMA->Worker in the first, Worker->MMA in the second), so no transfer
    is needed at the wave level — but the loops' pipelines would then be
    sibling bindings of one storage, whose barriers cannot chain. The
    kernel declines."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 128, 64), T.float16),
        B: T.Tensor((2, 64, 64), T.float16),
        C: T.Tensor((2, 128, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A1_shared = T.alloc_shared((128, 64), T.float16)
            B1_shared = T.alloc_shared((64, 64), T.float16)
            A2_shared = T.alloc_shared((128, 64), T.float16)
            B2_shared = T.alloc_shared((64, 64), T.float16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            C_frag = T.alloc_fragment((128, 64), T.float32)
            for w in T.serial(2):
                for _k1 in T.Pipelined(2, num_stages=2):
                    T.copy(A[0, 0, 0], A1_shared)
                    T.copy(B[0, 0, 0], B1_shared)
                    T.gemm(A1_shared, B1_shared, C_tmem, transpose_B=True, clear_accum=True)
                    T.copy(C_tmem, C_frag)
                    T.copy(C_frag, C[w, 0, 0])
                for _k2 in T.Pipelined(2, num_stages=2):
                    T.copy(A[1, 0, 0], A2_shared)
                    T.copy(B[1, 0, 0], B2_shared)
                    T.copy(C_frag, C_tmem)
                    T.gemm(A2_shared, B2_shared, C_tmem, transpose_B=True, clear_accum=False)

    mod, schedule = _schedule(kernel)
    assert schedule is not None
    with pytest.raises(Exception, match="not strictly nested"):
        tilelang.cuda.transform.MaterializeWSSchedule()(mod)


if __name__ == "__main__":
    tilelang.testing.main()
