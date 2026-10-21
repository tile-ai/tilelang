# ruff: noqa

import pytest

from tilelang import tvm as tvm
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm import tirx
from tvm.script import tirx as T


def run_passes(func: tvm.tirx.PrimFunc):
    mod = tvm.IRModule.from_expr(func)

    cuda_target = tvm.target.Target("cuda", host="llvm")

    mod = tvm.tirx.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)

    mod = tvm.tirx.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    return tilelang.transform.ThreadSync("shared")(mod)


def _device_func(mod):
    functions = [
        func
        for func in mod.functions.values()
        if isinstance(func, tirx.PrimFunc) and func.attrs and func.attrs.get("tirx.is_global_func", False)
    ]
    assert len(functions) == 1, f"Expected one device function, got {len(functions)}"
    return functions[0]


def _storage_syncs(root, scope="shared", *, arg_count=1):
    """Query full or named barriers by exact Op, scope and argument count."""
    return [
        call
        for call in collect_calls(root, op="tirx.tvm_storage_sync")
        if len(call.args) == arg_count and isinstance(call.args[0], tirx.StringImm) and call.args[0].value == scope
    ]


def _thread_var(func, tag="threadIdx.x"):
    variables = [
        attr.node.var for attr in collect_nodes(func, tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == tag
    ]
    assert len(variables) == 1, f"Expected one binding for {tag}, got {len(variables)}"
    return variables[0]


def _guard(root, condition):
    guards = [node for node in collect_nodes(root, tirx.IfThenElse) if tvm.ir.structural_equal(node.condition, condition)]
    assert len(guards) == 1, f"Expected one guard for {condition}, got {len(guards)}"
    return guards[0]


def _sync_stmt(root, call):
    statements = [node for node in collect_nodes(root, tirx.Evaluate) if node.value.same_as(call)]
    assert len(statements) == 1, "Expected the barrier to be one Evaluate statement"
    return statements[0]


def _assert_stmt_order(root, *statements):
    """Require direct siblings of one SeqStmt in the specified order.

    A post-order visitor or printed position cannot establish this contract:
    nodes in different branches or nested loops are not sequential siblings.
    """
    for sequence in collect_nodes(root, tirx.SeqStmt):
        positions = []
        for statement in statements:
            matches = [index for index, child in enumerate(sequence.seq) if child.same_as(statement)]
            if len(matches) != 1:
                break
            positions.append(matches[0])
        else:
            if all(first < second for first, second in zip(positions, positions[1:])):
                return
    raise AssertionError("Expected statements to be siblings in one SeqStmt in the specified order")


def _assert_syncs_inside(root, branch, *, scope="shared", arg_count=1):
    syncs = _storage_syncs(root, scope, arg_count=arg_count)
    assert syncs, f"Expected a {scope} barrier"
    guarded = _storage_syncs(branch, scope, arg_count=arg_count)
    assert all(any(call.same_as(candidate) for candidate in guarded) for call in syncs), "Barrier is outside the expected branch"
    return syncs


def _assert_partial_sync_params(syncs, thread_count):
    assert syncs, "Expected a partial barrier"
    for call in syncs:
        assert len(call.args) == 3
        assert all(isinstance(arg, tirx.IntImm) and arg.dtype == "int32" for arg in call.args[1:])
        assert 0 < int(call.args[1]) < 16, "Named barrier ID must be in the CUDA hardware range"
        assert int(call.args[2]) == thread_count, "Wrong partial-barrier participant count"


def _assert_shared_write_sync_read(branch, call):
    """Check the shared producer/barrier/consumer sequence of these kernels."""
    writes = [store for store in collect_nodes(branch, tirx.BufferStore) if store.buffer.scope() == "shared"]
    assert len(writes) == 1, "Expected one shared-memory producer"
    reads = [
        store
        for store in collect_nodes(branch, tirx.BufferStore)
        if store.buffer.scope() == "local"
        and any(load.buffer.data.same_as(writes[0].buffer.data) for load in collect_nodes(store.value, tirx.BufferLoad))
    ]
    assert len(reads) == 1, "Expected one consumer of the producer's shared buffer"
    _assert_stmt_order(branch, writes[0], _sync_stmt(branch, call), reads[0])


@tilelang.testing.requires_cuda
def test_no_sync_between_atomic_adds_to_shared():
    """Atomic WAW (and RMW) should not trigger thread-level sync insertion.

    This is a regression test for the case where ThreadSync conservatively
    treated atomic pointer accesses as conflicting and inserted syncthreads
    between atomics, degrading atomics into serialized updates.
    """

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16, 128), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.tvm_access_ptr(
                        T.type_annotation("float32"),
                        A_shared.data,
                        i * 128 + tx,
                        1,
                        3,
                    ),
                    T.float32(1),
                    T.int32(0),
                )
            )

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    assert not _storage_syncs(mod["main"])


@tilelang.testing.requires_cuda
def test_thread_sync_shared_dyn_alias_different_element_sizes():
    """Reused shared.dyn aliases with different dtypes need byte-based checks."""

    @T.prim_func(private=True)
    def func():
        buf_dyn_shmem = T.alloc_buffer((2048,), "uint8", scope="shared.dyn")
        x_ub: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        y_ub: T.handle("float8_e4m3fn", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        x_local = T.alloc_buffer((4,), "float32", scope="local")
        y_local = T.alloc_buffer((4,), "float8_e4m3fn", scope="local")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        x_ub_1 = T.decl_buffer((512,), "float32", data=x_ub, scope="shared.dyn")
        y_ub_1 = T.decl_buffer((512,), "float8_e4m3fn", data=y_ub, scope="shared.dyn")
        x_local_1 = T.decl_buffer((4,), "float32", data=x_local.data, scope="local")
        y_local_1 = T.decl_buffer((4,), "float8_e4m3fn", data=y_local.data, scope="local")
        x_local_1[T.Ramp(0, 1, 4)] = x_ub_1[T.Ramp(tx * 4, 1, 4)]
        y_local_1[T.Ramp(0, 1, 4)] = T.Cast("float8_e4m3fnx4", x_local_1[T.Ramp(0, 1, 4)])
        y_ub_1[T.Ramp(tx * 4, 1, 4)] = y_local_1[T.Ramp(0, 1, 4)]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    lowered = mod["main"]
    syncs = _storage_syncs(lowered, "shared.dyn")
    assert syncs
    fp8_writes = [
        store
        for store in collect_nodes(lowered, tirx.BufferStore)
        if store.buffer.scope() == "shared.dyn" and store.buffer.dtype == "float8_e4m3fn"
    ]
    assert len(fp8_writes) == 1
    for call in syncs:
        _assert_stmt_order(lowered, _sync_stmt(lowered, call), fp8_writes[0])


@tilelang.testing.requires_cuda
def test_thread_sync_handles_int64_tvm_access_ptr_offset():
    """Regression: shared/shared.dyn pointer offsets may be int64.

    ThreadSync used to reconstruct multidimensional indices with hardcoded
    int32 temporaries, which crashed on expressions like FloorDiv(int64, int32)
    while analyzing tvm_access_ptr from lowered atomic ops.
    """

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((128,), dtype="float32", scope="shared.dyn")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.evaluate(
            T.call_intrin(
                "float32",
                tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                T.tvm_access_ptr(
                    T.type_annotation("float32"),
                    A_shared.data,
                    T.Cast("int64", tx),
                    1,
                    3,
                ),
                T.float32(1),
                T.int32(0),
            )
        )

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    assert not _storage_syncs(mod["main"], "shared.dyn")


@tilelang.testing.requires_cuda
def test_sync_if_with_same_index():
    @T.prim_func(check_well_formed=False)
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.decl_buffer([2], dtype="float32", data=p0_arg.data)
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.alloc_buffer([1], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 8)
        T.launch_thread(threadIdx_x, 4)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if threadIdx_y < 8:
            temp_shared[threadIdx_x] = p0[0]
            temp_shared[threadIdx_x] = temp_shared[threadIdx_x]
        result_local[0] = result_local[0] + temp_shared[0]

    mod = run_passes(func)
    assert _storage_syncs(_device_func(mod))


def test_no_sync_if_with_same_index_with_modulo_if():
    """A thread-private update needs no barrier even under a divergent guard.

    Every thread accesses ``temp_shared[threadIdx_x]`` and nothing else, so the
    index is injective and no two threads ever reach the same address; the
    threads that skip the guarded write simply read an uninitialised slot, which
    is not a cross-thread hazard. Unequal participation alone (only
    ``tx % 4 == 0`` writes, everybody reads) is one of the two conditions a
    same-index hazard needs -- the other being a non-injective index -- so it
    must not by itself trigger a barrier.
    """

    @T.prim_func(check_well_formed=False)
    def func() -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.alloc_buffer([1], dtype="float32", scope="local")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.alloc_buffer([32], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 1)
        T.launch_thread(threadIdx_x, 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if threadIdx_x % 4 == 0:
            temp_shared[threadIdx_x] = p0[0]
        result_local[0] = temp_shared[threadIdx_x]

    mod = run_passes(func)
    assert not _storage_syncs(_device_func(mod))


@tilelang.testing.requires_cuda
def test_sync_read_thread_id_independent_location():
    @T.prim_func
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.decl_buffer([2], dtype="float32", data=p0_arg.data)
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.alloc_buffer([1], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 8)
        T.launch_thread(threadIdx_x, 4)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if threadIdx_x < 1:
            temp_shared[0] = p0[0]
        result_local[0] = result_local[0] + temp_shared[0] * p1[0]
        if threadIdx_x < 1:
            temp_shared[0] = p0[1]
        result_local[0] = result_local[0] + temp_shared[0] * p1[1]

    mod = run_passes(func)
    assert _storage_syncs(_device_func(mod))


@tilelang.testing.requires_cuda
def test_sync_shared():
    @T.prim_func(private=True)
    def func(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B = T.alloc_buffer((24,), "float32", scope="shared")
        C = T.alloc_buffer((1,), "float32", scope="local")
        D = T.alloc_buffer((16,), "float32", scope="shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1 = T.decl_buffer((24,), data=B.data, scope="shared")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1 = T.decl_buffer((1,), data=C.data, scope="local")
        C_1[0] = B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1 = T.decl_buffer((16,), data=D.data, scope="shared")
        D_1[threadIdx_x] = C_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1[threadIdx_x]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B_1 = T.alloc_buffer((24,), "float32", scope="shared")
        C_1 = T.alloc_buffer((1,), "float32", scope="local")
        D_1 = T.alloc_buffer((16,), "float32", scope="shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1_1 = T.decl_buffer((24,), data=B_1.data, scope="shared")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1_1 = T.decl_buffer((1,), data=C_1.data, scope="local")
        C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1_1 = T.decl_buffer((16,), data=D_1.data, scope="shared")
        D_1_1[threadIdx_x] = C_1_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1_1[threadIdx_x]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@tvm.testing.requires_cuda
def test_sync_let_stmt():
    @T.prim_func(private=True)
    def func(A: T.Buffer((16 * 512), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared = T.alloc_buffer((512,), "float32", scope="shared")
        in_thread_A_temp = T.alloc_buffer((1,), "float32", scope="local")
        cross_thread_A_temp = T.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared_1 = T.decl_buffer((512,), data=A_shared.data, scope="shared")
        for ax0 in range(512):
            A_shared_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1 = T.decl_buffer((1,), data=in_thread_A_temp.data, scope="local")
        in_thread_A_temp_1[0] = T.float32(0)
        A_temp_0 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x])
        in_thread_A_temp_1[0] = A_temp_0
        A_temp_1 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 128])
        in_thread_A_temp_1[0] = A_temp_1
        A_temp_2 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 256])
        in_thread_A_temp_1[0] = A_temp_2
        A_temp_3 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 384])
        in_thread_A_temp_1[0] = A_temp_3
        cross_thread_A_temp_1 = T.decl_buffer((1,), data=cross_thread_A_temp.data, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret(T.uint64(0), dtype="handle"),
        ):
            T.tvm_thread_allreduce(
                T.uint32(1),
                in_thread_A_temp_1[0],
                T.bool(True),
                cross_thread_A_temp_1[0],
                threadIdx_x,
            )

    @T.prim_func(private=True)
    def expected(A: T.Buffer((8192,), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared_1 = T.alloc_buffer((512,), "float32", scope="shared")
        in_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        cross_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared_1_1 = T.decl_buffer((512,), data=A_shared_1.data, scope="shared")
        for ax0 in range(512):
            A_shared_1_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1_1 = T.decl_buffer((1,), data=in_thread_A_temp_1.data, scope="local")
        in_thread_A_temp_1_1[0] = T.float32(0)
        T.tvm_storage_sync("shared")
        A_temp_0 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x])
        in_thread_A_temp_1_1[0] = A_temp_0
        A_temp_1 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 128])
        in_thread_A_temp_1_1[0] = A_temp_1
        A_temp_2 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 256])
        in_thread_A_temp_1_1[0] = A_temp_2
        A_temp_3 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 384])
        in_thread_A_temp_1_1[0] = A_temp_3
        cross_thread_A_temp_1_1 = T.decl_buffer((1,), data=cross_thread_A_temp_1.data, scope="local")
        T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret(T.uint64(0), dtype="handle"),
        )
        T.tvm_thread_allreduce(
            T.uint32(1),
            in_thread_A_temp_1_1[0],
            T.bool(True),
            cross_thread_A_temp_1_1[0],
            threadIdx_x,
        )

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@tilelang.testing.requires_cuda
def test_sync_shared_dyn_stmatrix_loop_hoist():
    @T.prim_func
    def func():
        buf_dyn_shmem = T.alloc_buffer((98304,), "uint8", scope="shared.dyn")
        tx = T.launch_thread("threadIdx.x", 384)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in T.unroll(8):
            off = (
                i // 4 * 8192
                + tx // 32 * 1024
                + tx % 16 * 64
                + (tx % 8 // 4 + i % 4 // 2) % 2 * 32
                + (tx % 4 // 2 + i % 2) % 2 * 16
                + (tx % 32 // 16 + tx % 2) % 2 * 8
            )
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.tirx.op.Op.get("tl.ptx_stmatrix"),
                    T.int32(0),
                    T.int32(4),
                    T.tvm_access_ptr(
                        T.type_annotation("uint8"),
                        buf_dyn_shmem.data,
                        off,
                        98304 - off,
                        2,
                    ),
                    T.int32(2),
                )
            )

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    lowered = mod["main"]
    syncs = _storage_syncs(lowered, "shared.dyn")
    assert syncs
    loops = collect_nodes(lowered, tirx.For)
    assert len(loops) == 1 and loops[0].kind == tirx.ForKind.UNROLLED
    assert int(loops[0].extent) == 8
    assert not _storage_syncs(loops[0], "shared.dyn")
    for call in syncs:
        _assert_stmt_order(lowered, _sync_stmt(lowered, call), loops[0])


@tilelang.testing.requires_cuda
def test_loop_carry_no_dependency_same_index():
    """Test that A[i] write followed by A[i] read in a loop does NOT need barrier.

    After iteration shift analysis:
    - Iteration i writes A[i]
    - Iteration i+1 reads A[i+1] (shifted from A[i])
    - A[i] vs A[i+1] are disjoint, so no loop-carried dependency
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Each iteration writes to A[tx], then reads from A[tx]
            # No loop-carried dependency because different iterations
            # access different locations
            temp_shared[tx] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[tx]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # Should NOT have sync inside the loop since A[tx] in iteration i
    # does not conflict with A[tx] in iteration i+1 (they're different threads' data)
    # The key insight: same thread writes and reads its own location
    assert not _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_loop_carry_with_cross_thread_dependency():
    """Test loop-carried dependency where different threads access overlapping locations.

    In this test:
    - Thread tx writes to A[tx]
    - Then reads from A[(tx + 127) % 128] (neighbor's data from previous iteration)

    After iteration shift analysis, we compare:
    - Iteration i: thread tx writes A[tx]
    - Iteration i+1: thread tx reads A[(tx + 127) % 128]

    This creates a cross-thread dependency where thread tx+1's write conflicts
    with thread tx's read in the next iteration, requiring a barrier.
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Each thread writes to its own location
            temp_shared[tx] = T.float32(i)
            # Then reads from neighbor (creates cross-thread dependency)
            result_local[0] = result_local[0] + temp_shared[(tx + 127) % 128]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # Should have sync because thread tx reads from thread (tx+127)%128's location
    # This is a WAR hazard across threads
    assert _storage_syncs(lowered)


@pytest.mark.parametrize(
    "start, stop, step, expected_syncs",
    [
        pytest.param(0, 2, None, 0, id="default_step"),
        pytest.param(0, 2, 1, 0, id="unit_step"),
        pytest.param(0, 6, 2, 1, id="non_unit_step"),
        pytest.param(3, 9, 2, 1, id="nonzero_start"),
        pytest.param(3, 8, 2, 1, id="nondivisible_extent"),
        pytest.param(3, 5, 2, 0, id="single_iteration"),
        pytest.param(3, 4, 2, 0, id="extent_smaller_than_step"),
    ],
)
@pytest.mark.parametrize("canonicalize", [False, True])
@tilelang.testing.requires_cuda
def test_loop_carry_respects_loop_step(start, stop, step, expected_syncs, canonicalize):
    """Raw and canonicalized steps must agree on adjacent-iteration hazards."""

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer((4, 32), dtype="float32", scope="shared")
        result_local = T.alloc_buffer((1,), dtype="float32", scope="local")
        tx = T.launch_thread("threadIdx.x", 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        temp_shared[0, tx] = T.float32(0)
        temp_shared[1, tx] = T.float32(0)
        temp_shared[2, tx] = T.float32(0)
        temp_shared[3, tx] = T.float32(0)
        T.tvm_storage_sync("shared")
        # Use TVM's serial builder to retain For.step; TileLang normalizes it.
        for i in T.serial(start, stop, step=step):
            temp_shared[i % 4, tx] = T.Cast("float32", i)
            result_local[0] = result_local[0] + temp_shared[(i + 2) % 4, (tx + 1) % 32]

    mod = tvm.IRModule({"main": func})
    if canonicalize:
        mod = tvm.s_tir.transform.CanonicalizeLoop()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    loops = collect_nodes(lowered, tirx.For)
    assert len(loops) == 1
    loop = loops[0]
    syncs = _storage_syncs(loop)
    assert len(syncs) == expected_syncs
    if expected_syncs:
        # The read must finish before the next iteration overwrites its row.
        writes = [store for store in collect_nodes(loop, tirx.BufferStore) if store.buffer.scope() == "shared"]
        assert len(writes) == 1
        _assert_stmt_order(loop.body, _sync_stmt(loop, syncs[0]), writes[0])


@pytest.mark.parametrize("canonicalize", [False, True])
@tilelang.testing.requires_cuda
def test_loop_carry_non_unit_step_last_pair(canonicalize):
    """The final valid pair must remain in the loop-carried dependency domain."""

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer((24, 32), dtype="float32", scope="shared")
        result_local = T.alloc_buffer((1,), dtype="float32", scope="local")
        tx = T.launch_thread("threadIdx.x", 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for j in range(24):
            temp_shared[j, tx] = T.float32(0)
        T.tvm_storage_sync("shared")
        for i in range(3, 13, 3):
            temp_shared[i, tx] = T.Cast("float32", i)
            # Only i=9 -> i=12 aliases across neighboring threads.
            result_local[0] = result_local[0] + temp_shared[21 - i, (tx + 1) % 32]

    mod = tvm.IRModule({"main": func})
    if canonicalize:
        mod = tvm.s_tir.transform.CanonicalizeLoop()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # The initialization loop must not count as a loop-carried barrier.
    loops = [
        loop
        for loop in collect_nodes(lowered, tirx.For)
        if any(load.buffer.scope() == "shared" for load in collect_nodes(loop, tirx.BufferLoad))
    ]
    assert len(loops) == 1
    loop = loops[0]
    syncs = _storage_syncs(loop)
    assert len(syncs) == 1
    writes = [store for store in collect_nodes(loop, tirx.BufferStore) if store.buffer.scope() == "shared"]
    assert len(writes) == 1
    _assert_stmt_order(loop.body, _sync_stmt(loop, syncs[0]), writes[0])


@tilelang.testing.requires_cuda
def test_loop_carry_modulo_buffering():
    """Test that A[i%2] write followed by A[i%2] read does NOT need barrier (double buffering).

    After iteration shift analysis:
    - Iteration i writes A[i%2]
    - Iteration i+1 reads A[(i+1)%2] (shifted from A[i%2])
    - A[i%2] vs A[(i+1)%2] are disjoint (0 vs 1 or 1 vs 0), so no dependency
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([2, 64], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Double buffering pattern: write to buffer[i%2], read from buffer[i%2]
            # After shift: write buffer[i%2], read buffer[(i+1)%2]
            # These are different buffers, so no conflict
            temp_shared[i % 2, tx] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[i % 2, tx]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # Each thread owns both slots, including across iterations.
    assert not _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_loop_carry_different_indices():
    """Test that A[i] write followed by A[i+1] read does NOT need barrier.

    After iteration shift analysis:
    - Iteration i writes A[i]
    - Iteration i+1 reads A[i+2] (shifted from A[i+1], becomes A[(i+1)+1] = A[i+2])
    - A[i] vs A[i+2] are disjoint, so no loop-carried dependency
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Write to A[i], read from A[i+1]
            # After shift: comparing A[i] (write) vs A[i+2] (read from i+1 shifted)
            # No overlap, no dependency
            temp_shared[i] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[i + 1]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert not _storage_syncs(lowered)


# =============================================================================
# Tests for non-uniform if condition sync hoisting
# =============================================================================


@tilelang.testing.requires_cuda
def test_sync_hoist_non_uniform_if_with_threadidx():
    """Test that sync is hoisted when if condition directly depends on threadIdx.

    When the if condition uses threadIdx, different threads may take different
    branches. If a sync is needed inside the if, it must be hoisted to before
    the if statement to avoid deadlock.
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # First, all threads write to shared memory
        temp_shared[tx] = T.float32(tx)
        # Non-uniform condition: only some threads enter the if
        if tx < 64:
            # Inside the if, we read from shared memory
            # This needs a sync, but since condition is non-uniform,
            # the sync must be hoisted to before the if
            result_local[0] = temp_shared[tx + 64]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    guard = _guard(lowered, _thread_var(lowered) < 64)
    syncs = _storage_syncs(lowered)
    assert syncs and not _storage_syncs(guard)
    writes = [store for store in collect_nodes(lowered, tirx.BufferStore) if store.buffer.scope() == "shared"]
    assert len(writes) == 1
    for call in syncs:
        _assert_stmt_order(lowered, writes[0], _sync_stmt(lowered, call), guard)


def test_no_sync_for_thread_private_read_inside_non_uniform_if():
    """A thread-private read guarded by a non-uniform condition needs no barrier.

    This is the shape that caused the original deadlock report: the condition
    reads shared memory at a threadIdx-dependent index, so a barrier inside the
    branch would hang. The guarded access is ``data_shared[tx]`` though -- the
    slot this very thread wrote just above -- so no two threads reach the same
    address and there is nothing to order.
    """

    @T.prim_func(private=True)
    def func():
        token_ids = T.alloc_buffer([128], dtype="int32", scope="shared")
        data_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # First phase: all threads write to data_shared
        data_shared[tx] = T.float32(tx)
        # Non-uniform condition: reads shared memory with threadIdx-dependent index
        # token_ids[tx] can be different for each thread (e.g., some are -1, some are valid)
        if token_ids[tx] != -1:
            # Inside the if, we read from data_shared
            result_local[0] = data_shared[tx]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert not _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_sync_inside_uniform_if_blockidx():
    """Test that sync can stay inside if when condition is uniform (blockIdx).

    When the if condition only depends on blockIdx (same for all threads in a block),
    all threads take the same branch, so sync inside the if is safe.
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 4)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # First, all threads write to shared memory
        temp_shared[tx] = T.float32(tx)
        # Uniform condition: blockIdx is same for all threads in a block
        if bx < 2:
            # Sync inside uniform if is safe - all threads in this block
            # will either all enter or all skip this branch
            result_local[0] = temp_shared[(tx + 64) % 128]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # Should have sync (either inside or outside the if is fine for uniform condition)
    assert _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_sync_inside_uniform_if_runtime_block_uniform_condition():
    """Runtime-loaded but block-uniform conditions should keep syncs in the if."""

    @T.prim_func(private=True)
    def func(flags: T.Buffer((4,), "int32")):
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 4)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if flags[bx] > 0:
            temp_shared[tx] = T.float32(tx)
            result_local[0] = temp_shared[(tx + 64) % 128]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    flags = lowered.buffer_map[lowered.params[0]]
    guard = _guard(lowered, tirx.BufferLoad(flags, [_thread_var(lowered, "blockIdx.x")]) > 0)
    syncs = _assert_syncs_inside(lowered, guard.then_case)
    assert len(syncs) == 1
    _assert_shared_write_sync_read(guard.then_case, syncs[0])


@tilelang.testing.requires_cuda
def test_sync_hoist_nested_non_uniform_if():
    """Test sync hoisting with nested if statements where outer is non-uniform."""

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # Write to shared memory
        temp_shared[tx] = T.float32(tx)
        # Outer non-uniform condition
        if tx < 64:
            # Inner condition (also non-uniform)
            if tx < 32:
                # Sync needed here must be hoisted all the way out
                result_local[0] = temp_shared[tx + 64]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    guard = _guard(lowered, _thread_var(lowered) < 64)
    syncs = _storage_syncs(lowered)
    assert syncs and not _storage_syncs(guard)
    writes = [store for store in collect_nodes(lowered, tirx.BufferStore) if store.buffer.scope() == "shared"]
    assert len(writes) == 1
    for call in syncs:
        _assert_stmt_order(lowered, writes[0], _sync_stmt(lowered, call), guard)


def test_no_sync_for_thread_private_read_inside_non_uniform_if_in_loop():
    """Same shape as above inside a loop, and still thread private.

    Iteration ``k`` writes ``data_shared[tx]`` and the guarded read of iteration
    ``k`` or ``k + 1`` targets the same slot, always from the same thread, so
    program order already orders them.
    """

    @T.prim_func(private=True)
    def func():
        token_ids = T.alloc_buffer([128], dtype="int32", scope="shared")
        data_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for k in range(2):
            # Write to shared memory
            data_shared[tx] = T.float32(tx + k)
            # Non-uniform if inside loop
            if token_ids[tx] != -1:
                result_local[0] = result_local[0] + data_shared[tx]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert not _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_no_sync_needed_uniform_accesses():
    """Test that no extra sync is added when accesses are already safe.

    When each thread only accesses its own data (no cross-thread dependency),
    no sync is needed even inside an if statement.
    """

    @T.prim_func(private=True)
    def func():
        temp_local = T.alloc_buffer([1], dtype="float32", scope="local")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        temp_local[0] = T.float32(tx)
        # Non-uniform condition but no shared memory access
        if tx < 64:
            result_local[0] = temp_local[0]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    # No sync needed - only local memory is accessed
    assert not _storage_syncs(lowered)


def test_no_sync_for_thread_private_write_read_by_if_condition_in_loop():
    """A non-uniform condition reading the slot the same thread just wrote.

    The write and the condition's read are both ``token_ids[tx]``, so the pair is
    thread private even though the condition is divergent and sits in a loop.
    """

    @T.prim_func(private=True)
    def func():
        token_ids = T.alloc_buffer([128], dtype="int32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for k in range(2):
            # Write to shared memory
            token_ids[tx] = T.int32(k - 2)
            # Non-uniform if inside loop
            if token_ids[tx] >= 0:
                result_local[0] = T.float32(1)

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert not _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_cross_thread_waw_with_shifted_partition_requires_sync():
    """A later writer can target a slot owned by another thread."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        s[tx] = T.cast(tx, "float32")
        if tx < 6:
            s[tx + 10] = T.float32(99)

    lowered = _device_func(run_passes(func))
    syncs = _storage_syncs(lowered)
    assert len(syncs) == 1
    first_write = collect_nodes(lowered, tirx.BufferStore)[0]
    guard = _guard(lowered, _thread_var(lowered) < 6)
    _assert_stmt_order(lowered, first_write, _sync_stmt(lowered, syncs[0]), guard)


@tilelang.testing.requires_cuda
def test_provably_disjoint_waw_partitions_need_no_sync():
    """Disjoint write partitions should not acquire a conservative barrier."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        s[tx] = T.cast(tx, "float32")
        s[tx + 64] = T.float32(99)

    lowered = _device_func(run_passes(func))
    assert not _storage_syncs(lowered), "Disjoint partitions need no barrier"


@pytest.mark.parametrize("active_threads", [6, 64])
@tilelang.testing.requires_cuda
def test_thread_private_waw_needs_no_sync(active_threads):
    """Thread-private writes remain ordered even when participation differs."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((64,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        s[tx] = T.cast(tx, "float32")
        if tx < active_threads:
            s[tx] = T.float32(99)

    lowered = _device_func(run_passes(func))
    assert not _storage_syncs(lowered), "Thread-private writes need no barrier"


@tilelang.testing.requires_cuda
def test_waw_bind_definition_uses_per_thread_instances():
    """Bind definitions must not collapse the two WAW thread instances."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        idx: T.int32 = tx + 10
        s[idx - 10] = T.cast(tx, "float32")
        if tx < 6:
            s[idx] = T.float32(99)

    lowered = _device_func(run_passes(func))
    assert _storage_syncs(lowered), "Bind sharing hid a cross-thread WAW hazard"


@tilelang.testing.requires_cuda
def test_partial_sync_non_warp_multiple_rejected():
    """Regression test for issue #2556: a required barrier inside a divergent
    region that splits a warp must be a compile-time error, not silently
    dropped or lowered to an invalid partial barrier."""
    import pytest

    @T.prim_func(private=True)
    def func():
        S = T.alloc_buffer((64,), dtype="float32", scope="shared")
        acc = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        # 48 participating threads: not a warp multiple.
        if tx < 48:
            acc[0] = T.float32(0)
            for i in range(2):
                S[tx] = T.float32(1)
                # Cross-thread read: planner must insert a sync here.
                acc[0] += S[47 - tx]

    mod = tvm.IRModule({"main": func})
    with pytest.raises(Exception, match="not warp-uniform"):
        tilelang.transform.ThreadSync("shared")(mod)


@tilelang.testing.requires_cuda
def test_partial_sync_warp_multiple_still_lowered():
    """Control for issue #2556: the same pattern with a warp-multiple
    participating thread count must still lower to a partial barrier."""

    @T.prim_func(private=True)
    def func():
        S = T.alloc_buffer((64,), dtype="float32", scope="shared")
        acc = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        # 32 participating threads: exactly one warp.
        if tx < 32:
            acc[0] = T.float32(0)
            for i in range(2):
                S[tx] = T.float32(1)
                acc[0] += S[31 - tx]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    guard = _guard(lowered, _thread_var(lowered) < 32)
    syncs = _assert_syncs_inside(lowered, guard.then_case, arg_count=3)
    _assert_partial_sync_params(syncs, 32)
    assert not _storage_syncs(guard)


# =============================================================================
# Tests for flat Bind definitions inside the recorded constraint sets
#
# ``AccessEntry.cset`` snapshots the constraint stack that is live when an access
# is recorded, so it also carries the definitions of the flat ``Bind`` statements
# that precede the access. A definition is not a participation constraint -- it
# never restricts which threads execute an access -- and feeding it to the prover
# as a proposition ``var == value`` distorts both queries in ``FindConflict``:
#
# * the same-index path compares the two constraint sets for logical equivalence.
#   A write sitting behind extra definitions can never be proven equivalent to
#   the matching read, so a thread-private read-modify-write is reported as a
#   hazard -- a spurious barrier, which is undefined behaviour once the access
#   sits under a thread-divergent guard.
# * the cross-thread path instantiates the same code twice, once per thread. A
#   definition shared between the two instances forces the two thread variables
#   to agree, which shrinks -- or empties -- the domain the disjointness proof
#   runs on, so a real hazard can be "proven" away.
# =============================================================================


def test_no_sync_for_thread_private_read_modify_write():
    """``v = s[tx]; s[tx] = f(v)`` is thread private and needs no barrier.

    Every thread reads and writes the one element it owns, so no two threads
    touch the same address. The read is recorded while evaluating the bind's
    value, making the write's constraint set the read's plus ``v == s[tx]``.
    """

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        v: T.float32 = s[tx]
        s[tx] = v * T.float32(2)

    lowered = _device_func(run_passes(func))
    assert not _storage_syncs(lowered)


def test_no_sync_for_thread_private_pair_read_modify_write():
    """A butterfly-looking update where each thread owns both slots it touches.

    Thread ``t`` reads and writes exactly ``{2t, 2t+1}``; those sets are pairwise
    disjoint, so there is no cross-thread dependency to order.
    """

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        x1: T.float32 = s[tx * 2]
        x2: T.float32 = s[tx * 2 + 1]
        s[tx * 2] = x1 + x2
        s[tx * 2 + 1] = x1 - x2

    lowered = _device_func(run_passes(func))
    assert not _storage_syncs(lowered)


def test_no_plain_sync_inside_divergent_symbolic_guard():
    """A barrier must never be emitted inside a thread-divergent guard.

    ``bx * 4 + tx // 32 < n`` mixes a thread variable with a runtime parameter,
    so for a tail block only some warps enter. A block-wide barrier inside the
    branch is then waited on by a subset of the block, which hangs. Whether or
    not the pass considers the guarded update a hazard, any barrier it emits has
    to sit outside the guard.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 32)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if bx * 4 + tx // 32 < n:
            v: T.float32 = s[tx]
            s[tx] = v * T.float32(2)

    lowered = _device_func(run_passes(func))
    guards = collect_nodes(lowered, tirx.IfThenElse)
    assert len(guards) == 1
    assert not _storage_syncs(guards[0])
    for call in _storage_syncs(lowered):
        _assert_stmt_order(lowered, _sync_stmt(lowered, call), guards[0])


def test_unorderable_hazard_in_divergent_guard_is_reported(capfd):
    """A hazard confined to a divergent branch has no correct barrier placement.

    ``bx * 4 + tx // 32 < n`` mixes a thread variable with a runtime parameter, so
    for a tail block only some warps enter, and both ends of the hazard are inside
    that branch. A barrier inside it hangs on the threads that skip the branch,
    while one in front of it is reached by everyone but no longer separates the
    accesses. Ordering this needs the branch split around the barrier, which the
    pass cannot express, so reporting the program is the only thing it can get
    right. Where the barrier ends up is deliberately not asserted.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 32)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if bx * 4 + tx // 32 < n:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[(tx + 64) % 128]

    run_passes(func)
    warnings = capfd.readouterr().err
    assert "no longer separates" in warnings, f"Expected the pass to report the hazard:\n{warnings}"


def test_unorderable_hazard_in_divergent_else_branch_is_reported(capfd):
    """The else branch needs the same treatment as the then branch.

    Syncs found in either arm are tracked separately, so a hazard placed only in
    the else arm exercises the second of the two paths. It is as unorderable as
    the one above, so again only the report is asserted.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if tx < n:
            l[0] = 0.0
        else:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[(tx + 64) % 128]

    run_passes(func)
    warnings = capfd.readouterr().err
    assert "no longer separates" in warnings, f"Expected the pass to report the hazard:\n{warnings}"


def test_aligned_else_partial_sync_is_preserved():
    """An unsafe 16-thread then arm must not remove a valid else warp barrier."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((64,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 48)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if tx >= 32:
            l[0] = T.float32(0)
        else:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[31 - tx]

    lowered = _device_func(run_passes(func))
    guard = _guard(lowered, _thread_var(lowered) >= 32)
    assert guard.else_case is not None
    syncs = _assert_syncs_inside(lowered, guard.else_case, arg_count=3)
    _assert_partial_sync_params(syncs, 32)
    assert not _storage_syncs(guard.then_case, arg_count=3)
    assert not _storage_syncs(lowered)


def test_misaligned_else_partial_sync_is_rejected():
    """Consecutive threads spanning two partial warps cannot use bar.sync."""
    import pytest

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((64,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if tx == 0 or tx >= 33:
            l[0] = T.float32(0)
        else:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[33 - tx]

    with pytest.raises(Exception, match="not warp-uniform"):
        run_passes(func)


def test_sync_stays_inside_guard_proven_uniform_by_premise(capfd):
    """A premise can make a threadIdx-dependent guard provably block uniform.

    ``tx < n`` mentions a thread variable, so no inspection of the condition can
    call it uniform. But given ``n % 128 == 0`` and a block of 128 threads, either
    all of them enter or none do. This is the one shape here that does have a
    correct answer, so the placement is asserted: the barrier stays inside the
    branch, where it still separates the two accesses, and nothing is reported.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.Assert(n % 128 == 0, "n must be a multiple of the block size"):
            if tx < n:
                s[tx] = T.cast(tx, "float32")
                l[0] = s[(tx + 64) % 128]

    lowered = _device_func(run_passes(func))
    warnings = capfd.readouterr().err
    parameters = [param for param in lowered.params if param.dtype == "int32"]
    assert len(parameters) == 1
    guard = _guard(lowered, _thread_var(lowered) < parameters[0])
    syncs = _assert_syncs_inside(lowered, guard.then_case)
    for call in syncs:
        _assert_shared_write_sync_read(guard.then_case, call)
    assert "no longer separates" not in warnings, f"A provably uniform guard should not be reported:\n{warnings}"


def test_premise_weaker_than_the_block_is_not_taken_as_uniform(capfd):
    """The premise has to rule divergence out for the block size actually used.

    ``n % 64 == 0`` with a block of 128 threads still admits ``n == 64``, where
    half the block enters. This pins down the boundary of the check above: it must
    accept a premise only when the premise really excludes divergence. The guard
    is therefore treated as divergent, which for this shape means reported.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.Assert(n % 64 == 0, "n must be a multiple of half the block"):
            if tx < n:
                s[tx] = T.cast(tx, "float32")
                l[0] = s[(tx + 64) % 128]

    run_passes(func)
    warnings = capfd.readouterr().err
    assert "no longer separates" in warnings, f"A premise that still allows divergence must not be accepted:\n{warnings}"


def test_premise_holds_across_an_enclosing_guard():
    """A premise stated outside an enclosing branch still reaches the inner guard.

    The constraint set accumulates down the nesting, so ``n % 128 == 0`` is
    available where ``tx < n`` is judged even though an ``if n > 0`` sits in
    between.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.Assert(n % 128 == 0, "n must be a multiple of the block size"):
            if n > 0:
                if tx < n:
                    s[tx] = T.cast(tx, "float32")
                    l[0] = s[(tx + 64) % 128]

    lowered = _device_func(run_passes(func))
    parameters = [param for param in lowered.params if param.dtype == "int32"]
    assert len(parameters) == 1
    guard = _guard(lowered, _thread_var(lowered) < parameters[0])
    syncs = _assert_syncs_inside(lowered, guard.then_case)
    for call in syncs:
        _assert_shared_write_sync_read(guard.then_case, call)


def test_premise_covers_every_thread_dimension():
    """Uniformity has to hold over all of threadIdx.x/y/z, not just x.

    With ``threadIdx.y`` extended, two threads compared for divergence may differ
    in ``y`` as well as ``x``; the guard is still uniform, and the barrier stays.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((256,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 2)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.Assert(n % 128 == 0, "n must be a multiple of the x extent"):
            if tx < n:
                s[ty * 128 + tx] = T.cast(tx, "float32")
                l[0] = s[ty * 128 + (tx + 64) % 128]

    lowered = _device_func(run_passes(func))
    parameters = [param for param in lowered.params if param.dtype == "int32"]
    assert len(parameters) == 1
    guard = _guard(lowered, _thread_var(lowered) < parameters[0])
    syncs = _assert_syncs_inside(lowered, guard.then_case)
    for call in syncs:
        _assert_shared_write_sync_read(guard.then_case, call)


def test_assume_serves_as_a_premise_like_an_assert():
    """``T.assume`` reaches the check the same way an assert does.

    An assume is what a caller-facing constraint looks like in practice, and it
    arrives as a scoped attribute rather than a flat statement, so it exercises
    the other of the two shapes a premise can take.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.assume(n % 128 == 0)
        if tx < n:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[(tx + 64) % 128]

    mod = tvm.IRModule.from_expr(func)
    cuda_target = tvm.target.Target("cuda", host="llvm")
    mod = tvm.tirx.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)
    mod = tvm.tirx.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    lowered = _device_func(tilelang.transform.ThreadSync("shared")(mod))

    parameters = [param for param in lowered.params if param.dtype == "int32"]
    assert len(parameters) == 1
    guard = _guard(lowered, _thread_var(lowered) < parameters[0])
    syncs = _assert_syncs_inside(lowered, guard.then_case)
    for call in syncs:
        _assert_shared_write_sync_read(guard.then_case, call)


def test_hazard_across_split_divergent_symbolic_guard_is_ordered():
    """The accepted shape for the program above: split the guard by hand.

    With the branch split around the synchronization point, the two accesses no
    longer sit in one branch: the conflict spans the guard, so the barrier is
    anchored in the enclosing sequence and is reached by every thread.
    """

    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 32)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if bx * 4 + tx // 32 < n:
            s[tx] = T.cast(tx, "float32")
        if bx * 4 + tx // 32 < n:
            l[0] = s[(tx + 64) % 128]

    lowered = _device_func(run_passes(func))
    guards = collect_nodes(lowered, tirx.IfThenElse)
    assert len(guards) == 2
    writers = [
        guard for guard in guards if any(store.buffer.scope() == "shared" for store in collect_nodes(guard.then_case, tirx.BufferStore))
    ]
    assert len(writers) == 1
    readers = [guard for guard in guards if not guard.same_as(writers[0])]
    assert tvm.ir.structural_equal(writers[0].condition, readers[0].condition)
    syncs = _storage_syncs(lowered)
    assert syncs
    assert all(not _storage_syncs(guard) for guard in guards)
    for call in syncs:
        _assert_stmt_order(lowered, writers[0], _sync_stmt(lowered, call), readers[0])


def test_bind_definition_does_not_hide_cross_thread_hazard():
    """A bind that is live across both accesses must not weaken the analysis.

    ``idx`` is defined before both the write and the read, so both constraint
    sets carry its definition. Sharing it would state ``idx == tx1 + 7`` and
    ``idx == tx2 + 7``, forcing ``tx1 == tx2`` against the proof's ``tx1 != tx2``:
    every query would hold vacuously and the real hazard -- thread ``t`` reads the
    slot owned by ``t + 7`` -- would be "proven" absent.
    """

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        idx: T.int32 = tx + 7
        s[tx] = T.cast(tx, "float32")
        l[0] = s[idx % 128]

    lowered = _device_func(run_passes(func))
    assert _storage_syncs(lowered)


def test_bool_bind_does_not_hide_cross_thread_hazard():
    """Same defect, reached through a boolean bind.

    Boolean binds matter in the full pipeline because let-inlining only folds
    integer values away, so a predicate such as ``tx < 64`` reaches ThreadSync
    intact. Sharing it between the two instances asserts
    ``(tx1 < 64) == (tx2 < 64)``, confining both threads to the same half of the
    block and letting the disjointness proof succeed where it must not.
    """

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        c: T.bool = tx < 64
        s[tx] = T.cast(tx, "float32")
        l[0] = s[(tx + 64) % 128] + T.cast(c, "float32")

    lowered = _device_func(run_passes(func))
    assert _storage_syncs(lowered)


def test_opaque_bind_index_does_not_hide_cross_thread_hazard():
    """A bind whose definition is dropped still needs a per-side instance.

    ``a`` is loaded from a buffer, so its definition is dropped from the
    constraint set rather than substituted away. The variable survives in the two
    indices, so each side still needs its own copy: sharing one ``a`` would let
    the prover assume both threads loaded the same value and discharge
    ``4*a != 4*a + 4``, hiding the hazard between the thread writing ``s[a]`` and
    the thread whose ``a`` is one lower. Spelling the load out at both indices
    instead of binding it must give the same answer.
    """

    @T.prim_func(private=True)
    def func(idx_buf: T.Buffer((128,), "int32")):
        s = T.alloc_buffer((256,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        a: T.int32 = idx_buf[tx]
        s[a] = T.cast(tx, "float32")
        l[0] = s[a + 1]

    @T.prim_func(private=True)
    def without_bind(idx_buf: T.Buffer((128,), "int32")):
        s = T.alloc_buffer((256,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        s[idx_buf[tx]] = T.cast(tx, "float32")
        l[0] = s[idx_buf[tx] + 1]

    lowered = _device_func(run_passes(func))
    assert _storage_syncs(lowered)
    assert _storage_syncs(_device_func(run_passes(without_bind))), "Control case lost its barrier"


def test_cross_thread_hazard_still_requires_sync():
    """Reading another thread's slot is a genuine hazard and must be ordered."""

    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        s[tx] = T.cast(tx, "float32")
        l[0] = s[(tx + 64) % 128]

    lowered = _device_func(run_passes(func))
    assert _storage_syncs(lowered)


def test_sync_may_stay_inside_block_uniform_guard():
    """A guard that does not mention a thread variable is block uniform.

    All threads agree on it, so a barrier inside the branch is reached by the
    whole block and does not need hoisting.
    """

    @T.prim_func(private=True)
    def func(flag: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if flag < 0:
            s[tx] = T.cast(tx, "float32")
            x1: T.float32 = s[(tx + 64) % 128]
            s[tx] = x1 * T.float32(2)

    lowered = _device_func(run_passes(func))
    guard = _guard(lowered, lowered.params[-1] < 0)
    _assert_syncs_inside(lowered, guard.then_case)


@tilelang.testing.requires_cuda
def test_unbounded_atomic_touched_range_stays_conservative():
    """Regression: an atomic whose address is data-dependent relaxes its
    touched interval to (-inf, +inf), and those symbolic infinity sentinels
    are handle-typed. Comparing them against a bounded pointer access inside
    PointerAccessIsDisjoint threw `Cannot match type int32 vs handle`; the
    planner must instead answer "not disjoint" and keep the sync."""

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((256,), dtype="float32", scope="shared")
        idx = T.alloc_buffer((16,), dtype="int32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        # Bounded pointer access covering the whole buffer (fill-style).
        T.evaluate(
            T.call_extern(
                "handle",
                "fake_fill",
                T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 256, 2),
            )
        )
        # Data-dependent atomic destination: the touched interval relaxes to
        # everything once the enclosing loop is summarized.
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.address_of(A_shared[idx[i]]),
                    T.float32(1),
                    T.int32(0),
                )
            )

    mod = tvm.IRModule({"main": func})
    # Must not throw; the unprovable pair keeps its conservative barrier.
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert _storage_syncs(lowered)


@tilelang.testing.requires_cuda
def test_sync_grid_acts_as_shared_barrier():
    """grid.sync() synchronizes every thread of every block, so the planner
    must treat it as a block-level barrier too: accesses on the two sides of
    a sync_grid need no extra __syncthreads(), and (regression) the
    unprovable fill-vs-data-dependent-atomic pair is never even compared."""

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((256,), dtype="float32", scope="shared")
        idx = T.alloc_buffer((16,), dtype="int32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.evaluate(
            T.call_extern(
                "handle",
                "fake_fill",
                T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 256, 2),
            )
        )
        T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.sync_grid")))
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.address_of(A_shared[idx[i]]),
                    T.float32(1),
                    T.int32(0),
                )
            )

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    lowered = mod["main"]
    assert not _storage_syncs(lowered)


@pytest.mark.parametrize(
    "args",
    [
        [tirx.StringImm("shared.dyn")],
        [tirx.StringImm("shared"), tirx.IntImm("int32", 1), tirx.IntImm("int32", 32)],
    ],
    ids=["wrong_scope", "partial_barrier"],
)
def test_full_sync_assertion_rejects_wrong_scope_or_partial_barrier(args):
    call = tirx.Call("int32", tvm.ir.Op.get("tirx.tvm_storage_sync"), args)
    with pytest.raises(AssertionError):
        assert _storage_syncs(tirx.Evaluate(call)), "Expected a full shared-memory barrier"


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (0, 2, 1)])
def test_shared_write_sync_read_assertion_checks_statement_order(order):
    shared = tirx.decl_buffer((1,), "float32", scope="shared")
    local = tirx.decl_buffer((1,), "float32", scope="local")
    write = tirx.BufferStore(shared, tirx.FloatImm("float32", 1), [0])
    call = tirx.call_intrin("int32", "tirx.tvm_storage_sync", "shared")
    sync = tirx.Evaluate(call)
    read = tirx.BufferStore(local, tirx.BufferLoad(shared, [0]), [0])
    statements = (write, sync, read)
    body = tirx.SeqStmt([statements[index] for index in order])
    if order == (0, 1, 2):
        _assert_shared_write_sync_read(body, call)
    else:
        with pytest.raises(AssertionError, match="siblings in one SeqStmt"):
            _assert_shared_write_sync_read(body, call)


@pytest.mark.parametrize("location", ["expected", "outside", "other_guard", "else"])
def test_sync_branch_assertion_checks_the_actual_guard_subtree(location):
    condition = tirx.Var("condition", "bool")
    call = tirx.call_intrin("int32", "tirx.tvm_storage_sync", "shared")
    sync = tirx.Evaluate(call)
    expected_branch = sync if location == "expected" else tirx.Evaluate(0)
    guard = tirx.IfThenElse(condition, expected_branch, sync if location == "else" else None)
    if location == "outside":
        root = tirx.SeqStmt([guard, sync])
    elif location == "other_guard":
        root = tirx.SeqStmt([guard, tirx.IfThenElse(tirx.Var("other", "bool"), sync, None)])
    else:
        root = guard
    selected = _guard(root, condition)
    if location == "expected":
        _assert_syncs_inside(root, selected.then_case)
    else:
        with pytest.raises(AssertionError, match="outside the expected branch"):
            _assert_syncs_inside(root, selected.then_case)


@pytest.mark.parametrize("region", ["branch", "loop"])
def test_statement_order_assertion_rejects_nested_regions(region):
    first, second = tirx.Evaluate(1), tirx.Evaluate(2)
    if region == "branch":
        nested = tirx.IfThenElse(tirx.Var("guard", "bool"), second, None)
    else:
        nested = tirx.For(tirx.Var("i", "int32"), 0, 2, tirx.ForKind.SERIAL, second)
    root = tirx.SeqStmt([first, nested])
    with pytest.raises(AssertionError, match="siblings in one SeqStmt"):
        _assert_stmt_order(root, first, second)


@pytest.mark.parametrize("barrier_id, thread_count", [(1, 32), (0, 32), (16, 32), (1, 64)])
def test_partial_sync_assertion_checks_id_and_participant_count(barrier_id, thread_count):
    call = tirx.call_intrin("int32", "tirx.tvm_storage_sync", "shared", barrier_id, thread_count)
    if (barrier_id, thread_count) == (1, 32):
        _assert_partial_sync_params([call], 32)
    else:
        with pytest.raises(AssertionError):
            _assert_partial_sync_params([call], 32)


def _thread_sync_calls(func):
    mod = tilelang.transform.ThreadSync("shared")(tvm.IRModule({"main": func}))
    barriers = []

    def collect(node):
        if isinstance(node, tvm.tirx.Call) and node.op == tvm.ir.Op.get("tirx.tvm_storage_sync"):
            barriers.append(tuple(int(arg) for arg in node.args[1:]))

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    return barriers


def test_disjoint_partial_sync_scopes_use_distinct_ids():
    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 128)
        if tx // 64 == 0:
            for i in range(2):
                T.evaluate(T.tvm_storage_sync("shared"))
                T.evaluate(T.tvm_storage_sync("shared"))
        else:
            for j in range(2):
                T.evaluate(T.tvm_storage_sync("shared"))
                T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert [call[1] for call in barriers] == [64, 64, 64, 64]
    assert barriers[0] == barriers[1]
    assert barriers[2] == barriers[3]
    assert barriers[0][0] != barriers[2][0]


def test_partial_sync_assume_scopes_use_distinct_ids():
    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 128)
        with T.attr(tx // 64 == 0, "tl.assume", 1):
            T.evaluate(T.tvm_storage_sync("shared"))
        with T.attr(tx // 64 == 1, "tl.assume", 1):
            T.evaluate(T.tvm_storage_sync("shared"))
        with T.attr(tx // 64 == 0, "tl.assume", 1):
            T.evaluate(T.tvm_storage_sync("shared"))
        T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 4
    assert [call[1] for call in barriers[:3]] == [64, 64, 64]
    assert barriers[0] == barriers[2]
    assert barriers[0][0] != barriers[1][0]
    assert barriers[3] == ()


@pytest.mark.parametrize("guard", [lambda tx: tx < 64, lambda tx: tx // 64 == 0], ids=["direct", "quotient"])
def test_equivalent_partial_thread_guards_lower_to_the_same_barriers(guard):
    """Fragment replication guards need partial barriers within the loop."""

    @T.prim_func(private=True)
    def func(A: T.Buffer((2, 64), "float16"), Out: T.Buffer((2, 64), "float16")):
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        shared = T.alloc_buffer((64,), "float16", scope="shared")
        if guard(tx):
            for k in range(2):
                shared[tx] = A[k, tx]
                Out[k, tx] = shared[(tx + 1) % 64]
        T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 3
    assert barriers[0] == barriers[1]
    assert barriers[0][1] == 64
    assert barriers[2] == ()


def test_partial_sync_preserves_outer_constraint_across_block_scope():
    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 128)
        T.evaluate(T.tvm_storage_sync("shared"))
        if tx < 64:
            with T.sblock("guarded_sync"):
                T.evaluate(T.tvm_storage_sync("shared"))
        else:
            with T.sblock("other_sync"):
                T.evaluate(T.tvm_storage_sync("shared"))
        T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 4
    assert barriers[0] == barriers[3] == ()
    assert barriers[1][1] == barriers[2][1] == 64
    assert barriers[1][0] != barriers[2][0]


def test_equivalent_partial_thread_guards_reuse_barrier_id():
    """The producer offset and a bound replica index preserve participant IDs."""

    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 256)
        replica: T.int32 = (tx - 128) // 64
        if tx >= 128 and replica == 0:
            T.evaluate(T.tvm_storage_sync("shared"))
        if tx >= 128 and tx < 192:
            T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 2
    assert barriers[0] == barriers[1]
    assert barriers[0][1] == 64


def test_partial_sync_assert_constraint_stays_in_scope():
    @T.prim_func(private=True)
    def func():
        bx = T.launch_thread("blockIdx.x", 32)
        tx = T.launch_thread("threadIdx.x", 128)
        if bx < 16:
            assert tx // 64 == 0, "replica zero"
            T.evaluate(T.tvm_storage_sync("shared"))
        T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 2
    assert barriers[0][1] == 64
    assert barriers[1] == ()


@pytest.mark.parametrize(
    "guard",
    [lambda tx: tx % 4 == 0, lambda tx: (tx // 32) % 2 == 0, lambda tx: tvm.tirx.Or(tx < 64, tx >= 224)],
    ids=["sparse_lanes", "alternating_warps", "separated_intervals"],
)
def test_partial_sync_rejects_nonrectangular_threads(guard):
    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 256)
        if guard(tx):
            T.evaluate(T.tvm_storage_sync("shared"))

    with pytest.raises(tvm.error.InternalError, match="rectangle"):
        _thread_sync_calls(func)


def test_partial_sync_multidimensional_rectangle():
    @T.prim_func(private=True)
    def func():
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 2)
        if tx // 32 == 0:
            T.evaluate(T.tvm_storage_sync("shared"))

    barriers = _thread_sync_calls(func)
    assert len(barriers) == 1
    assert barriers[0][1] == 64


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("axis", ["y", "z"])
@pytest.mark.parametrize("shift", [0, 1])
def test_waw_uses_all_thread_axes(axis, shift):
    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((64,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 4)
        ty = T.launch_thread("threadIdx.y", 4)
        tz = T.launch_thread("threadIdx.z", 4)
        s[tx + 4 * ty + 16 * tz] = T.float32(1)
        if axis == "y":
            s[tx + 4 * ((ty + shift) % 4) + 16 * tz] = T.float32(2)
        else:
            s[tx + 4 * ty + 16 * ((tz + shift) % 4)] = T.float32(2)

    lowered = _device_func(run_passes(func))
    assert len(_storage_syncs(lowered)) == shift


if __name__ == "__main__":
    tilelang.testing.main()
