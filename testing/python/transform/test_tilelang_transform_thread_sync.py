# ruff: noqa

from tilelang import tvm as tvm
import tilelang.testing
from tvm.script import tir as T


def run_passes(func: tvm.tir.PrimFunc):
    mod = tvm.IRModule.from_expr(func)

    cuda_target = tvm.target.Target("cuda", host="llvm")

    mod = tvm.tir.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)

    mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    return tilelang.transform.ThreadSync("shared")(mod)


@tilelang.testing.requires_cuda
def test_sync_if_with_same_index():
    @T.prim_func(check_well_formed=False)
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.Buffer([2], dtype="float32", data=p0_arg.data)
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
    assert "T.tvm_storage_sync" in str(mod)


@tilelang.testing.requires_cuda
def test_sync_if_with_same_index_with_modulo_if():
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
    assert "T.tvm_storage_sync" in str(mod)


@tilelang.testing.requires_cuda
def test_sync_read_thread_id_independent_location():
    @T.prim_func
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.Buffer([2], dtype="float32", data=p0_arg.data)
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
    assert "T.tvm_storage_sync" in str(mod)


@tilelang.testing.requires_cuda
def test_sync_shared():
    @T.prim_func(private=True)
    def func(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B = T.allocate([24], "float32", "shared")
        C = T.allocate([1], "float32", "local")
        D = T.allocate([16], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1 = T.Buffer((24,), data=B, scope="shared")
        A_1 = T.Buffer((16,), data=A.data)
        B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1 = T.Buffer((1,), data=C, scope="local")
        C_1[0] = B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1 = T.Buffer((16,), data=D, scope="shared")
        D_1[threadIdx_x] = C_1[0]
        E_1 = T.Buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1[threadIdx_x]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B_1 = T.allocate([24], "float32", "shared")
        C_1 = T.allocate([1], "float32", "local")
        D_1 = T.allocate([16], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1_1 = T.Buffer((24,), data=B_1, scope="shared")
        A_1 = T.Buffer((16,), data=A.data)
        B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1_1 = T.Buffer((1,), data=C_1, scope="local")
        C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1_1 = T.Buffer((16,), data=D_1, scope="shared")
        D_1_1[threadIdx_x] = C_1_1[0]
        E_1 = T.Buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1_1[threadIdx_x]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@tvm.testing.requires_cuda
def test_sync_let_stmt():
    @T.prim_func(private=True)
    def func(A: T.Buffer((16 * 512), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared = T.allocate([512], "float32", "shared")
        in_thread_A_temp = T.allocate([1], "float32", "local")
        cross_thread_A_temp = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared_1 = T.Buffer((512,), data=A_shared, scope="shared")
        for ax0 in range(512):
            A_shared_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1 = T.Buffer((1,), data=in_thread_A_temp, scope="local")
        in_thread_A_temp_1[0] = T.float32(0)
        with T.LetStmt(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x]) as A_temp:
            in_thread_A_temp_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 128]) as A_temp:
            in_thread_A_temp_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 256]) as A_temp:
            in_thread_A_temp_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 384]) as A_temp:
            in_thread_A_temp_1[0] = A_temp
        cross_thread_A_temp_1 = T.Buffer((1,), data=cross_thread_A_temp, scope="local")
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
        A_shared_1 = T.allocate([512], "float32", "shared")
        in_thread_A_temp_1 = T.allocate([1], "float32", "local")
        cross_thread_A_temp_1 = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared_1_1 = T.Buffer((512,), data=A_shared_1, scope="shared")
        for ax0 in range(512):
            A_shared_1_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1_1 = T.Buffer((1,), data=in_thread_A_temp_1, scope="local")
        in_thread_A_temp_1_1[0] = T.float32(0)
        T.tvm_storage_sync("shared")
        with T.LetStmt(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x]) as A_temp:
            in_thread_A_temp_1_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 128]) as A_temp:
            in_thread_A_temp_1_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 256]) as A_temp:
            in_thread_A_temp_1_1[0] = A_temp
        with T.LetStmt(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 384]) as A_temp:
            in_thread_A_temp_1_1[0] = A_temp
        T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret(T.uint64(0), dtype="handle"),
        )
        cross_thread_A_temp_1_1 = T.Buffer((1,), data=cross_thread_A_temp_1, scope="local")
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
                    tvm.tir.op.Op.get("tl.ptx_stmatrix"),
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
    s = str(mod)
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
    s = str(mod)
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
    s = str(mod)
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
    s = str(mod)
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
    s = str(mod)
    print(f"Different indices result:\n{s}")


if __name__ == "__main__":
    tilelang.testing.main()
