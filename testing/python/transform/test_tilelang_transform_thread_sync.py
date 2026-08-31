# ruff: noqa

from tilelang import tvm as tvm
import tilelang.testing
from tvm.script import tirx as T


def run_passes(func: tvm.tirx.PrimFunc):
    mod = tvm.IRModule.from_expr(func)

    cuda_target = tvm.target.Target("cuda", host="llvm")

    mod = tvm.tirx.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)

    mod = tvm.tirx.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    return tilelang.transform.ThreadSync("shared")(mod)


def run_passes_script(func: tvm.tirx.PrimFunc) -> str:
    return str(run_passes(func).script())


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync inserted for atomic ops:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared.dyn")' in s, f"Expected sync:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared.dyn")')
    write_pos = s.index(" = y_local_1[0:4]")
    assert sync_pos < write_pos, f"Sync should appear before aliased fp8 write:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared.dyn")' not in s, f"Unexpected sync inserted for single atomic op:\n{s}"


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
    assert "T.tvm_storage_sync" in str(mod.script())


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
    assert "T.tvm_storage_sync" not in str(mod.script())


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
    assert "T.tvm_storage_sync" in str(mod.script())


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared.dyn")' in s
    # Ensure the sync appears before the unrolled loop
    assert s.index('T.tvm_storage_sync("shared.dyn")') < s.index("for i in T.unroll(8)")


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
    s = str(mod.script())
    # Should NOT have sync inside the loop since A[tx] in iteration i
    # does not conflict with A[tx] in iteration i+1 (they're different threads' data)
    # The key insight: same thread writes and reads its own location
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync in loop:\n{s}"


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
    s = str(mod.script())
    # Should have sync because thread tx reads from thread (tx+127)%128's location
    # This is a WAR hazard across threads
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected sync for cross-thread dependency:\n{s}"


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
    s = str(mod.script())
    # Should NOT have sync inside loop due to modulo buffering analysis
    # Note: This test verifies the modulo analysis capability
    print(f"Modulo buffering result:\n{s}")


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
    s = str(mod.script())
    print(f"Different indices result:\n{s}")


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
    s = str(mod.script())
    # Sync should appear before the if statement
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected sync:\n{s}"
    # The sync should be before the if, not inside it
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < 64")
    assert sync_pos < if_pos, f"Sync should be before if statement:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync:\n{s}"


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
    s = str(mod.script())
    # Should have sync (either inside or outside the if is fine for uniform condition)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected sync:\n{s}"


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
    s = str(mod.script())
    assert s.count('T.tvm_storage_sync("shared")') == 1, f"Expected exactly one sync:\n{s}"
    if_pos = s.index("if flags[bx] > 0")
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    assert sync_pos > if_pos, f"Block-uniform runtime condition should keep sync inside if:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected sync:\n{s}"
    # Sync should be before the outermost non-uniform if
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < 64")
    assert sync_pos < if_pos, f"Sync should be hoisted before outer if:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync:\n{s}"


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
    s = str(mod.script())
    # No sync needed - only local memory is accessed
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync:\n{s}"


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
    import re

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
    s = str(mod.script())
    assert re.search(r'tvm_storage_sync\("shared",\s*\d+,\s*32\)', s), f"Expected a partial barrier with thread_count=32:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync for a thread-private update:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' not in s, f"Unexpected sync for per-thread private pairs:\n{s}"


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

    s = run_passes_script(func)
    if 'T.tvm_storage_sync("shared")' in s:
        sync_pos = s.index('T.tvm_storage_sync("shared")')
        if_pos = s.index("if ")
        assert sync_pos < if_pos, f"Barrier must be hoisted out of the divergent guard:\n{s}"


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

    run_passes_script(func)
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

    run_passes_script(func)
    warnings = capfd.readouterr().err
    assert "no longer separates" in warnings, f"Expected the pass to report the hazard:\n{warnings}"


def test_aligned_else_partial_sync_is_preserved():
    """An unsafe 16-thread then arm must not remove a valid else warp barrier."""
    import re

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

    s = run_passes_script(func)
    assert re.search(r'tvm_storage_sync\("shared",\s*\d+,\s*32\)', s), f"Expected a 32-thread barrier:\n{s}"
    assert s.index('T.tvm_storage_sync("shared"') > s.index("else:"), f"Barrier must stay in else branch:\n{s}"


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
        run_passes_script(func)


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

    s = run_passes_script(func)
    warnings = capfd.readouterr().err
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < n")
    assert sync_pos > if_pos, f"Barrier should stay inside the uniform guard:\n{s}"
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

    run_passes_script(func)
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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < n")
    assert sync_pos > if_pos, f"Barrier should stay inside the uniform guard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < n")
    assert sync_pos > if_pos, f"Barrier should stay inside the uniform guard:\n{s}"


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
    s = str(tilelang.transform.ThreadSync("shared")(mod).script())

    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < n")
    assert sync_pos > if_pos, f"Barrier should stay inside the uniform guard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Missing barrier: a bind definition hid a cross-thread hazard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Missing barrier: a boolean bind hid a cross-thread hazard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Missing barrier: an opaque bind index hid a cross-thread hazard:\n{s}"
    assert 'T.tvm_storage_sync("shared")' in run_passes_script(without_bind), "Control case lost its barrier"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"


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

    s = run_passes_script(func)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a barrier for a cross-thread hazard:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected a conservative barrier:\n{s}"


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
    s = str(mod.script())
    assert 'T.tvm_storage_sync("shared")' not in s, f"sync_grid already covers the barrier:\n{s}"


if __name__ == "__main__":
    tilelang.testing.main()
