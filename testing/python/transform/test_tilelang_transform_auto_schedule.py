"""Tests for AutoSchedule's "role_based" scheduler.

The pass consumes plain (schedule-free) kernels, assigns fixed roles from
lowering eligibility (Load / MMA / Store / Worker), pulls warp-private
def-use chains into their consumers' roles, derives per-storage pipelines
with alternating producer/consumer cycles, and emits a typed ``WSSchedule``
plus ``tl.ws_op_id`` markers. All lowering stays in
``MaterializeWSSchedule``. Kernels the scheduler declines must come back
byte-for-byte unchanged.
"""

import tilelang
import tilelang.testing
import torch
from tilelang import language as T
from tilelang import tvm
from tilelang.language.warp_specialize import WSOpRef, WSSync, WSSyncKind

_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _prepare(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(_TARGET)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    return mod


def _schedule(func):
    """Run the pass; returns (scheduled module, root WSSchedule or None)."""
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(_prepare(func))
    return scheduled, _root_schedule(scheduled["main"])


def _root_schedule(func):
    result = None

    def visit(node):
        nonlocal result
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "tilelang_root":
            result = node.annotations.get("tl.ws_schedule")

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return result


_SYNC_NAMES = [
    (WSSyncKind.PRODUCER_ACQUIRE, "acquire"),
    (WSSyncKind.PRODUCER_COMMIT, "commit"),
    (WSSyncKind.CONSUMER_WAIT, "wait"),
    (WSSyncKind.CONSUMER_RELEASE, "release"),
]


def _shape(body):
    """Body as ["op", "acquire(S)", ...] — op ids abstracted away."""
    out = []
    for instr in body:
        if isinstance(instr, WSOpRef):
            out.append("op")
        else:
            assert isinstance(instr, WSSync)
            assert instr.stage == 0, "every sync must be stage 0"
            kind = next(name for variant, name in _SYNC_NAMES if instr.kind == variant)
            out.append(f"{kind}({instr.pipeline})")
    return out


def _op_ids(body):
    return [str(instr.id) for instr in body if isinstance(instr, WSOpRef)]


def _scope_bodies(schedule, predicate):
    """The bodies of the unique scope some role body of which satisfies `predicate`."""
    matches = [scope.bodies for scope in schedule.scopes if any(predicate(role, scope.bodies[role]) for role in scope.bodies)]
    assert len(matches) == 1
    return matches[0]


def _roles(schedule):
    return [(str(role.name), role.warp_lo, role.warp_hi, role.max_nreg) for role in schedule.roles]


def _pipelines(schedule):
    return [(str(pipeline.name), [str(buffer.name) for buffer in pipeline.buffers], pipeline.depth) for pipeline in schedule.pipelines]


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
    _, schedule = _schedule(_pipelined_load_kernel())
    assert schedule is not None
    assert schedule.num_warps == 8
    assert _roles(schedule) == [("Worker", 0, 4, 0), ("Load", 4, 5, 32)]
    assert _pipelines(schedule) == [("S", ["S"], 2)]

    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["acquire(S)", "op", "commit(S)"]
    assert _shape(bodies["Worker"]) == ["wait(S)", "op", "release(S)", "op"]


@tilelang.testing.requires_cuda
def test_num_stages_sets_pipeline_depth():
    _, schedule = _schedule(_pipelined_load_kernel(num_stages=3))
    assert _pipelines(schedule) == [("S", ["S"], 3)]


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
    _, schedule = _schedule(_two_cycle_kernel())
    assert schedule is not None
    assert _pipelines(schedule) == [("S", ["S"], 1)]

    root = _scope_bodies(schedule, lambda role, body: role == "Load")
    assert _shape(root["Load"]) == [
        "acquire(S)",
        "op",
        "commit(S)",
        "acquire(S)",
        "op",
        "commit(S)",
    ]
    assert _shape(root["Worker"]) == [
        "wait(S)",
        "op",
        "release(S)",
        "wait(S)",
        "op",
        "release(S)",
        "op",
        "op",
    ]


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
    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["op", "acquire(S)", "op", "commit(S)"]
    assert _shape(bodies["Worker"]) == ["wait(S)", "op", "release(S)", "op"]
    assert not set(_op_ids(bodies["Load"])) & set(_op_ids(bodies["Worker"]))


@tilelang.testing.requires_cuda
def test_bind_duplicated_into_two_roles():
    """A pure global-reading Bind used by two roles is placed in both; the
    materializer re-emits it per role."""
    _, schedule = _schedule(_gather_kernel(worker_uses_index=True))
    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["op", "acquire(S)", "op", "commit(S)"]
    assert _shape(bodies["Worker"]) == ["op", "wait(S)", "op", "release(S)", "op"]
    bind_id = _op_ids(bodies["Load"])[0]
    assert _op_ids(bodies["Worker"])[0] == bind_id


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
    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["op", "op", "acquire(S)", "op", "commit(S)"]
    assert _shape(bodies["Worker"]) == ["wait(S)", "op", "release(S)", "op"]
    assert not set(_op_ids(bodies["Load"])) & set(_op_ids(bodies["Worker"]))


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
    assert schedule.num_warps == 8
    assert _roles(schedule) == [("Worker", 0, 4, 0), ("Load", 4, 5, 32)]
    # A_shared and B_shared share roles, depth, and cycle shape, so they
    # merge behind one barrier pair (the hand-written multi-buffer form).
    assert _pipelines(schedule) == [("A_shared_B_shared", ["A_shared", "B_shared"], 2)]

    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == [
        "acquire(A_shared_B_shared)",
        "op",
        "op",
        "commit(A_shared_B_shared)",
    ]
    assert _shape(bodies["Worker"]) == [
        "wait(A_shared_B_shared)",
        "op",
        "release(A_shared_B_shared)",
    ]

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
    assert schedule.num_warps == 8
    assert _roles(schedule) == [
        ("Worker", 0, 4, 0),
        ("Load", 4, 5, 32),
        ("MMA", 5, 6, 32),
        ("Store", 6, 7, 32),
    ]
    assert _pipelines(schedule) == [
        ("A_shared_B_shared", ["A_shared", "B_shared"], 2),
        ("C_tmem", ["C_tmem"], 1),
        ("C_shared", ["C_shared"], 1),
    ]

    root = _scope_bodies(schedule, lambda role, body: "acquire(C_tmem)" in _shape(body))
    assert _shape(root["MMA"]) == ["acquire(C_tmem)", "op", "commit(C_tmem)"]
    assert _shape(root["Worker"]) == ["wait(C_tmem)", "op", "release(C_tmem)"]
    assert _shape(root["Load"]) == ["op"]
    assert _shape(root["Store"]) == ["op"]

    epilogue = _scope_bodies(schedule, lambda role, body: "acquire(C_shared)" in _shape(body))
    assert _shape(epilogue["Worker"]) == ["op", "acquire(C_shared)", "op", "commit(C_shared)"]
    assert _shape(epilogue["Store"]) == ["wait(C_shared)", "op", "release(C_shared)"]

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
    assert _roles(schedule) == [
        ("Worker", 0, 4, 0),
        ("Load", 4, 5, 32),
        ("MMA", 5, 6, 32),
        ("Store", 6, 7, 32),
    ]
    assert _pipelines(schedule) == [
        ("A_shared_B_shared", ["A_shared", "B_shared"], 2),
        ("C_tmem", ["C_tmem"], 1),
        ("C_shared", ["C_shared"], 1),
    ]

    root = _scope_bodies(schedule, lambda role, body: "acquire(C_tmem)" in _shape(body))
    assert _shape(root["MMA"]) == ["acquire(C_tmem)", "op", "commit(C_tmem)"]
    assert _shape(root["Worker"]) == ["wait(C_tmem)", "op", "release(C_tmem)"]

    epilogue = _scope_bodies(schedule, lambda role, body: "acquire(C_shared)" in _shape(body))
    assert _shape(epilogue["Worker"]) == ["op", "acquire(C_shared)", "op", "commit(C_shared)"]
    assert _shape(epilogue["Store"]) == ["wait(C_shared)", "op", "release(C_shared)"]


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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    _, schedule = _schedule(_rmw_accumulator_kernel())
    assert schedule is not None
    assert _pipelines(schedule) == [
        ("A_shared_B_shared", ["A_shared", "B_shared"], 2),
        ("C_tmem", ["C_tmem"], 1),
        ("C_tmem_root", ["C_tmem"], 1),
    ]

    loop = _scope_bodies(schedule, lambda role, body: "acquire(A_shared_B_shared)" in _shape(body))
    assert _shape(loop["Worker"]) == [
        "acquire(C_tmem)",
        "op",
        "op",
        "op",
        "commit(C_tmem)",
    ]
    assert _shape(loop["MMA"]) == [
        "wait(A_shared_B_shared)",
        "wait(C_tmem)",
        "op",
        "release(A_shared_B_shared)",
        "release(C_tmem)",
    ]

    root = _scope_bodies(schedule, lambda role, body: "wait(C_tmem_root)" in _shape(body))
    assert _shape(root["Worker"]) == [
        "op",
        "wait(C_tmem_root)",
        "op",
        "release(C_tmem_root)",
        "op",
    ]
    assert _shape(root["MMA"]) == ["acquire(C_tmem_root)", "op", "commit(C_tmem_root)"]


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
    hands S to the Worker; at loop_w the flipped subtree hands the whole
    loop's S to the post-loop read. The inner binding is pinned
    single-buffered (an outer access cannot name an inner slot); the
    outer one takes loop_w's num_stages."""
    _, schedule = _schedule(_post_loop_read_kernel())
    assert schedule is not None
    names = [(p.name, [b.name for b in p.buffers], p.depth) for p in schedule.pipelines]
    assert len(names) == 2
    assert names[0][1] == ["S"] and names[0][2] == 1  # inner, pinned
    assert names[1][1] == ["S"] and names[1][2] == 2  # outer, wave ring

    outer_name = str(schedule.pipelines[1].name)
    wave = _scope_bodies(schedule, lambda role, body: f"acquire({outer_name})" in _shape(body))
    assert _shape(wave["Load"]) == [f"acquire({outer_name})", "op", f"commit({outer_name})"]
    assert _shape(wave["Worker"]) == [
        "op",
        f"wait({outer_name})",
        "op",
        f"release({outer_name})",
        "op",
    ]

    inner = _scope_bodies(schedule, lambda role, body: "acquire(S)" in _shape(body))
    assert _shape(inner["Load"]) == ["acquire(S)", "op", "commit(S)"]
    assert _shape(inner["Worker"]) == ["wait(S)", "op", "release(S)", "op"]


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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    _, schedule = _schedule(_persistent_while_kernel())
    assert schedule is not None
    assert _pipelines(schedule) == [("S", ["S"], 1)]

    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and "acquire(S)" in _shape(body))
    # The wave bump is duplicated into both roles inside the while body.
    assert _shape(bodies["Load"]) == ["acquire(S)", "op", "commit(S)", "op"]
    assert _shape(bodies["Worker"]) == ["wait(S)", "op", "release(S)", "op", "op"]
    assert _op_ids(bodies["Load"])[-1] == _op_ids(bodies["Worker"])[-1]


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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    _, schedule = _schedule(_non_tma_layout_kernel())
    assert schedule is not None
    assert _pipelines(schedule) == [("S", ["S"], 2)]

    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["acquire(S)", "op", "commit(S)"]
    # Worker keeps the S_perm copy: [copy S_perm, wait, read S, release,
    # read S_perm, add, store out].
    assert _shape(bodies["Worker"]) == [
        "op",
        "wait(S)",
        "op",
        "release(S)",
        "op",
        "op",
        "op",
    ]


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
    assert _roles(schedule) == [("Worker", 0, 4, 0), ("Load", 4, 5, 32)]
    assert _pipelines(schedule) == [("S", ["S"], 2)]

    bodies = _scope_bodies(schedule, lambda role, body: role == "Load" and len(body) > 1)
    assert _shape(bodies["Load"]) == ["acquire(S)", "op", "commit(S)"]

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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    scheduled = tilelang.cuda.transform.AutoSchedule("role_based")(prepared)
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
    import pytest

    with pytest.raises(Exception, match="unknown auto-schedule scheduler"):
        tilelang.cuda.transform.AutoSchedule("nonexistent")


@tilelang.testing.requires_cuda
def test_unschedulable_constructs_fail_fast():
    """Auto scheduling is opt-in, so PreprocessIR enforces the
    schedulability contract with hard failures instead of silently falling
    back: no atomics, and no asynchronous producer nested inside an opaque
    op."""
    import pytest

    @T.prim_func
    def atomic_kernel(A: T.Tensor((64, 64), T.float32), B: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.copy(A, S)
            for i, j in T.Parallel(64, 64):
                T.atomic_add(B[i, j], S[i, j])

    with pytest.raises(Exception, match="atomic"):
        tilelang.cuda.transform.AutoSchedule("role_based")(_prepare(atomic_kernel))

    @T.prim_func
    def hosted_async_kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            F = T.alloc_fragment((64, 64), T.float16)
            with T.ws_op("staged"):  # opaque wrapper hosting an async g2s copy
                T.copy(A, S)
                T.copy(S, F)
            T.copy(F, B)

    with pytest.raises(Exception, match="nested"):
        tilelang.cuda.transform.AutoSchedule("role_based")(_prepare(hosted_async_kernel))

    @T.prim_func
    def sync_threads_kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float16)
            T.copy(A, S)
            # A block-wide barrier cannot be duplicated into role branches.
            T.sync_threads()
            T.copy(S, B)

    with pytest.raises(Exception, match="synchronization"):
        tilelang.cuda.transform.AutoSchedule("role_based")(_prepare(sync_threads_kernel))


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


if __name__ == "__main__":
    tilelang.testing.main()
