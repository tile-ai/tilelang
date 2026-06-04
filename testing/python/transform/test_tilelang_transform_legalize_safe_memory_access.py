from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm.tirx.stmt_functor import ir_transform, post_order_visit


def _strip_block_reads_writes(stmt, strip_annotations: bool = False):
    """Strip non-behavioral block metadata before structural comparison."""

    def _postorder(op):
        if isinstance(op, tvm.tirx.SBlock):
            annotations = {} if strip_annotations else op.annotations
            return tvm.tirx.SBlock(
                op.iter_vars,
                [],
                [],
                op.name_hint,
                op.body,
                op.init,
                op.alloc_buffers,
                op.match_buffers,
                annotations,
            )

    return ir_transform(stmt, None, _postorder)


def _collect_call_nodes(stmt, op_names):
    if isinstance(op_names, str):
        op_names = {op_names}
    else:
        op_names = set(op_names)
    calls = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) in op_names:
            calls.append(node)

    post_order_visit(stmt, _visit)
    return calls


def _is_call_to(expr, op_name):
    return isinstance(expr, tvm.tirx.Call) and isinstance(expr.op, tvm.ir.Op) and str(expr.op.name) == op_name


def _is_int_zero(expr):
    return isinstance(expr, tvm.tirx.IntImm) and int(expr.value) == 0


def _assert_tl_access_ptr_bases_are_buffer_loads(stmt):
    for call in _collect_call_nodes(stmt, "tl.access_ptr"):
        assert isinstance(call.args[0], tvm.tirx.BufferLoad)


def _count_if_then_else(stmt):
    count = 0

    def _visit(node):
        nonlocal count
        if isinstance(node, tvm.tirx.IfThenElse):
            count += 1

    post_order_visit(stmt, _visit)
    return count


def _assert_legalize_matches_expected(before, expected, strip_annotations: bool = False):
    mod = tvm.IRModule({before.attrs["global_symbol"]: before})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    _assert_tl_access_ptr_bases_are_buffer_loads(before.body)
    _assert_tl_access_ptr_bases_are_buffer_loads(transformed["main"].body)
    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body, strip_annotations),
        _strip_block_reads_writes(expected.body, strip_annotations),
    )


def vectorize_access_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A_shared[tid, j] = A[tid + M_offset, j + N_offset]

    @T.prim_func
    def expected(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()

            for j in T.serial(N):
                A_shared[tid, j] = T.if_then_else(
                    j + N_offset < N, T.if_then_else(tid + M_offset < M, A[tid + M_offset, j + N_offset], T.float32(0)), T.float32(0)
                )

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body),
        _strip_block_reads_writes(expected.body),
    )


def vectorize_access_with_atmoic_add_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A_shared[tid, j] = A[tid + M_offset, j + N_offset]
                T.atomic_add(A[tid + M_offset, j + N_offset], 1)

    @T.prim_func
    def expected(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()

            for j in T.serial(N):
                A_shared[tid, j] = T.if_then_else(
                    j + N_offset < N, T.if_then_else(tid + M_offset < M, A[tid + M_offset, j + N_offset], T.float32(0)), T.float32(0)
                )
                # Nest if-then-else is expected, do not flatten it to pass structural equal check
                if j + N_offset < N:  # noqa: SIM102
                    if tid + M_offset < M:
                        T.atomic_add(A[tid + M_offset, j + N_offset], 1)

    return main, expected


def assert_vectorize_access_with_atmoic_add(M: int = 64, N: int = 64):
    func, expected = vectorize_access_with_atmoic_add_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    print(transformed)
    print(expected)
    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body),
        _strip_block_reads_writes(expected.body),
    )


def oob_store_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A[tid + M_offset, j + N_offset] = 1

    @T.prim_func
    def expected(
        A: T.Tensor((M, N), dtype=dtype),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            for j in T.serial(N):
                if j + N_offset < N:  # noqa: SIM102
                    if tid + M_offset < M:
                        A[tid + M_offset, j + N_offset] = T.float32(1.0)

    return main, expected


def assert_oob_store_legalize(M: int = 64, N: int = 64):
    func, expected = oob_store_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body),
        _strip_block_reads_writes(expected.body),
    )


def cp_async_access_ptr_legalize():
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
    ):
        A_shared = T.alloc_buffer((16,), dtype=dtype, scope="shared")
        for i in T.serial(4):
            T.ptx_cp_async(
                T.access_ptr(A_shared[i * 4], "w", 4),
                T.access_ptr(A[i * 4 + 8], "r", 4),
                4,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
    ):
        A_shared = T.alloc_buffer((16,), dtype=dtype, scope="shared")
        for i in T.serial(4):
            T.ptx_cp_async(
                T.access_ptr(A_shared[i * 4], "w", 4),
                T.access_ptr(A[i * 4 + 8], "r", 4),
                4,
                i < 2,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)

    return main, expected


def assert_cp_async_access_ptr_legalize():
    func, expected = cp_async_access_ptr_legalize()
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = _collect_call_nodes(body, {"tirx.ptx_cp_async", "tl.ptx_cp_async"})
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 4 for call in cp_async_calls)
    _assert_legalize_matches_expected(func, expected)


def cp_async_access_ptr_nonzero_safe_value_legalize():
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
    ):
        with T.sblock("root"):
            T.reads()
            T.writes()
            T.sblock_attr({"safe_value_map": {A.data: T.float16(3)}})
            A_shared = T.sblock_alloc_buffer((16,), dtype=dtype, scope="shared")
            for i in T.serial(4):
                T.ptx_cp_async(
                    T.access_ptr(A_shared[i * 4], "w", 4),
                    T.access_ptr(A[i * 4 + 8], "r", 4),
                    4,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
    ):
        with T.sblock("root"):
            T.reads()
            T.writes()
            T.sblock_attr({"safe_value_map": {A.data: T.float16(3)}})
            A_shared = T.sblock_alloc_buffer((16,), dtype=dtype, scope="shared")
            for i in T.serial(4):
                if i < 2:
                    T.ptx_cp_async(
                        T.access_ptr(A_shared[i * 4], "w", 4),
                        T.access_ptr(A[i * 4 + 8], "r", 4),
                        4,
                    )
                else:
                    A_shared[i * 4] = T.float16(3)
            T.ptx_commit_group()
            T.ptx_wait_group(0)

    return main, expected


def atomic_load_access_ptr_legalize():
    dtype = T.int32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((4,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i] = T.atomic_load(A[i * 4 + 10], memory_order="acquire")

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((4,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i] = T.if_then_else(
                i < 2,
                T.atomic_load(A[i * 4 + 10], memory_order="acquire"),
                T.int32(0),
            )

    return main, expected


def atomic_add_return_access_ptr_legalize():
    dtype = T.int32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((4,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i] = T.atomic_add(A[i * 4 + 10], T.int32(1), return_prev=True)

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((4,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i] = T.if_then_else(
                i < 2,
                T.atomic_add(A[i * 4 + 10], T.int32(1), return_prev=True),
                T.int32(0),
            )

    return main, expected


def atomic_store_access_ptr_legalize():
    dtype = T.int32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            T.atomic_store(A[i * 4 + 10], T.int32(1), memory_order="release")

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            if i * 4 + 10 < 16:
                T.atomic_store(A[i * 4 + 10], T.int32(1), memory_order="release")

    return main, expected


def call_extern_access_ptr_mask_legalize(access_type: str):
    dtype = T.int32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            T.call_extern(
                "handle",
                "use_ptr",
                T.access_ptr(A[i * 4 + 10], access_type, 1),
            )

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            if i * 4 + 10 < 16:
                T.call_extern(
                    "handle",
                    "use_ptr",
                    T.access_ptr(A[i * 4 + 10], access_type, 1),
                )

    return main, expected


def call_extern_multiple_access_ptrs_legalize():
    dtype = T.int32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((12,), dtype=dtype),
    ):
        for i in T.serial(4):
            T.call_extern(
                "handle",
                "use_two_ptrs",
                T.access_ptr(A[i * 4 + 10], "r", 1),
                T.access_ptr(B[i * 4 + 6], "w", 1),
            )

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((12,), dtype=dtype),
    ):
        for i in T.serial(4):
            if i * 4 + 6 < 12:  # noqa: SIM102
                if i * 4 + 10 < 16:
                    T.call_extern(
                        "handle",
                        "use_two_ptrs",
                        T.access_ptr(A[i * 4 + 10], "r", 1),
                        T.access_ptr(B[i * 4 + 6], "w", 1),
                    )

    return main, expected


def assert_cp_async_access_ptr_nonzero_safe_value_legalize():
    func, expected = cp_async_access_ptr_nonzero_safe_value_legalize()
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = _collect_call_nodes(body, {"tirx.ptx_cp_async", "tl.ptx_cp_async"})
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 3 for call in cp_async_calls)
    assert _count_if_then_else(body) > 0
    _assert_legalize_matches_expected(func, expected, strip_annotations=True)


def assert_atomic_load_access_ptr_legalize():
    func, expected = atomic_load_access_ptr_legalize()
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_buffer_loads(body)
    if_then_else_calls = _collect_call_nodes(body, "tirx.if_then_else")
    assert any(
        len(call.args) == 3 and _is_call_to(call.args[1], "tl.atomic_load_elem_op") and _is_int_zero(call.args[2])
        for call in if_then_else_calls
    )
    _assert_legalize_matches_expected(func, expected)


def assert_atomic_add_return_access_ptr_legalize():
    func, expected = atomic_add_return_access_ptr_legalize()
    _assert_legalize_matches_expected(func, expected)


def assert_atomic_store_access_ptr_legalize():
    func, expected = atomic_store_access_ptr_legalize()
    _assert_legalize_matches_expected(func, expected)


def assert_call_extern_access_ptr_mask_legalize(access_type: str):
    func, expected = call_extern_access_ptr_mask_legalize(access_type)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_buffer_loads(body)
    assert _count_if_then_else(body) > 0
    _assert_legalize_matches_expected(func, expected)


def assert_call_extern_multiple_access_ptrs_legalize():
    func, expected = call_extern_multiple_access_ptrs_legalize()
    _assert_legalize_matches_expected(func, expected)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


def test_vectorize_access_with_atmoic_add():
    assert_vectorize_access_with_atmoic_add(64, 64)


def test_oob_store():
    assert_oob_store_legalize(64, 64)


def test_cp_async_access_ptr_oob():
    assert_cp_async_access_ptr_legalize()


def test_cp_async_access_ptr_nonzero_safe_value_oob():
    assert_cp_async_access_ptr_nonzero_safe_value_legalize()


def test_atomic_load_access_ptr_oob():
    assert_atomic_load_access_ptr_legalize()


def test_atomic_add_return_access_ptr_oob():
    assert_atomic_add_return_access_ptr_legalize()


def test_atomic_store_access_ptr_oob():
    assert_atomic_store_access_ptr_legalize()


def test_call_extern_access_ptr_read_mask_oob():
    assert_call_extern_access_ptr_mask_legalize("r")


def test_call_extern_access_ptr_write_mask_oob():
    assert_call_extern_access_ptr_mask_legalize("w")


def test_call_extern_access_ptr_readwrite_mask_oob():
    assert_call_extern_access_ptr_mask_legalize("rw")


def test_call_extern_multiple_access_ptrs_oob():
    assert_call_extern_multiple_access_ptrs_legalize()


if __name__ == "__main__":
    tilelang.testing.main()
