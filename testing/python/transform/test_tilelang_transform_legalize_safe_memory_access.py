import math

import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
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


def _materialize_launch(func):
    """Run the launch materialization the pipeline performs before
    LegalizeSafeMemoryAccess, so thread indices carry their extents."""
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target("cuda"))(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    return mod[func.attrs["global_symbol"]]


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


def _apply_negative_index_then_safe_memory(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    return tl.transform.LegalizeSafeMemoryAccess()(mod)


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
                A_shared[tid, j] = T.if_then_else(tid < M - M_offset and j < N - N_offset, A[tid + M_offset, j + N_offset], T.float32(0))

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    func, expected = _materialize_launch(func), _materialize_launch(expected)
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
                A_shared[tid, j] = T.if_then_else(tid < M - M_offset and j < N - N_offset, A[tid + M_offset, j + N_offset], T.float32(0))
                # Nest if-then-else is expected, do not flatten it to pass structural equal check
                if j + N_offset < N:  # noqa: SIM102
                    if tid + M_offset < M:
                        T.atomic_add(A[tid + M_offset, j + N_offset], 1)

    return main, expected


def assert_vectorize_access_with_atmoic_add(M: int = 64, N: int = 64):
    func, expected = vectorize_access_with_atmoic_add_legalize(M, N)
    func, expected = _materialize_launch(func), _materialize_launch(expected)
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
                if tid < M - M_offset and j < N - N_offset:
                    A[tid + M_offset, j + N_offset] = T.float32(1.0)

    return main, expected


def assert_oob_store_legalize(M: int = 64, N: int = 64):
    func, expected = oob_store_legalize(M, N)
    func, expected = _materialize_launch(func), _materialize_launch(expected)
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


def atomic_addx2_return_access_ptr_legalize():
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((8,), dtype=dtype),
        out: T.Tensor((8,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i * 2 : i * 2 + 2] = T.atomic_addx2(A[i * 4 + 10], B[i * 2], return_prev=True)

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((8,), dtype=dtype),
        out: T.Tensor((8,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i * 2 : i * 2 + 2] = T.if_then_else(
                i < 2,
                T.atomic_addx2(A[i * 4 + 10], B[i * 2], return_prev=True),
                T.Broadcast(T.float32(0), 2),
            )

    return main, expected


def atomic_addx4_return_access_ptr_legalize():
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i * 4 : i * 4 + 4] = T.atomic_addx4(A[i * 4 + 8], B[i * 4], return_prev=True)

    @T.prim_func
    def expected(
        A: T.Tensor((16,), dtype=dtype),
        B: T.Tensor((16,), dtype=dtype),
        out: T.Tensor((16,), dtype=dtype),
    ):
        for i in T.serial(4):
            out[i * 4 : i * 4 + 4] = T.if_then_else(
                i < 2,
                T.atomic_addx4(A[i * 4 + 8], B[i * 4], return_prev=True),
                T.Broadcast(T.float32(0), 4),
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


def assert_atomic_vector_add_return_access_ptr_legalize():
    for make_case in (atomic_addx2_return_access_ptr_legalize, atomic_addx4_return_access_ptr_legalize):
        func, expected = make_case()
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


def assert_runtime_unknown_sign_vector_negative_index_legalize():
    @T.prim_func
    def main(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[T.Ramp(t - 2, 1, 4)]

    @T.prim_func
    def expected(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[
                    T.Shuffle(
                        [
                            T.Select(t < 2, t + 1022, t - 2),
                            T.Select(t < 1, t + 1023, t - 1),
                            t,
                            t + 1,
                        ],
                        [0, 1, 2, 3],
                    )
                ]

    transformed = _apply_negative_index_then_safe_memory(main)
    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body),
        _strip_block_reads_writes(expected.body),
    )


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


@pytest.mark.parametrize("safe_value", [-0.0, 0.0], ids=["negative_zero", "positive_zero"])
def test_cp_async_signed_zero_safe_value_oob(safe_value):
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32)):
        with T.sblock("root"):
            T.sblock_attr({"safe_value_map": {A.data: T.float32(safe_value)}})
            shared = T.sblock_alloc_buffer((2,), T.float32, scope="shared")
            for i in T.serial(2):
                T.ptx_cp_async(
                    T.access_ptr(shared[i], "w", 1),
                    T.access_ptr(A[i + 15], "r", 1),
                    1,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)

    mod = tl.transform.LegalizeSafeMemoryAccess()(tvm.IRModule.from_expr(main))
    body = mod["main"].body
    calls = _collect_call_nodes(body, "tl.ptx_cp_async")
    assert len(calls) == 1
    if math.copysign(1.0, safe_value) > 0:
        assert len(calls[0].args) == 4
        assert _count_if_then_else(body) == 0
    else:
        assert len(calls[0].args) == 3
        assert _count_if_then_else(body) == 1
        stores = []
        post_order_visit(body, lambda node: stores.append(node) if isinstance(node, tvm.tirx.BufferStore) else None)
        assert len(stores) == 1
        fill = stores[0].value
        assert isinstance(fill, tvm.tirx.FloatImm)
        assert fill.value == 0.0 and math.copysign(1.0, fill.value) < 0


def test_atomic_load_access_ptr_oob():
    assert_atomic_load_access_ptr_legalize()


def test_atomic_add_return_access_ptr_oob():
    assert_atomic_add_return_access_ptr_legalize()


def test_atomic_vector_add_return_access_ptr_oob():
    assert_atomic_vector_add_return_access_ptr_legalize()


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


def test_runtime_unknown_sign_vector_negative_index_oob():
    assert_runtime_unknown_sign_vector_negative_index_legalize()


def test_dynamic_index_load_store_conditions_are_combined():
    @T.prim_func
    def main(
        A: T.Tensor((16,), T.float32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[index]
        A[index] = T.float32(1)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = _collect_call_nodes(body, "tirx.if_then_else")
    assert len(load_guards) == 1
    assert isinstance(load_guards[0].args[0], tvm.tirx.And)
    assert isinstance(load_guards[0].args[1], tvm.tirx.BufferLoad)
    assert isinstance(load_guards[0].args[2], tvm.tirx.FloatImm)
    assert float(load_guards[0].args[2].value) == 0.0

    store_guards = []

    def _collect_store_guard(node):
        if isinstance(node, tvm.tirx.IfThenElse):
            store_guards.append(node)

    post_order_visit(body, _collect_store_guard)
    assert len(store_guards) == 1
    assert isinstance(store_guards[0].condition, tvm.tirx.And)
    assert isinstance(store_guards[0].then_case, tvm.tirx.BufferStore)
    tvm.ir.assert_structural_equal(load_guards[0].args[0], store_guards[0].condition)


def test_nested_buffer_load_index_is_safely_rewritten():
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((4,), T.int32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[B[index]]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data
    b_data = main.buffer_map[main.params[1]].data

    def _is_load_from(expr, buffer_data):
        return isinstance(expr, tvm.tirx.BufferLoad) and expr.buffer.data.same_as(buffer_data)

    def _assert_safe_b_index(expr):
        assert _is_call_to(expr, "tirx.if_then_else")
        assert isinstance(expr.args[0], tvm.tirx.And)
        assert _is_load_from(expr.args[1], b_data)
        assert _is_int_zero(expr.args[2])

    all_guards = _collect_call_nodes(body, "tirx.if_then_else")
    inner_guards = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], a_data)]
    assert len(inner_guards) == 1

    inner_guard = inner_guards[0]
    outer_guards = [call for call in all_guards if len(call.args) == 3 and call.args[1].same_as(inner_guard)]
    assert len(outer_guards) == 1

    outer_guard = outer_guards[0]
    for fallback in (inner_guard.args[2], outer_guard.args[2]):
        assert isinstance(fallback, tvm.tirx.FloatImm)
        assert float(fallback.value) == 0.0
    _assert_safe_b_index(inner_guard.args[1].indices[0])

    for comparison in (inner_guard.args[0], outer_guard.args[0]):
        assert isinstance(comparison, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))
        safe_indices = [operand for operand in (comparison.a, comparison.b) if _is_call_to(operand, "tirx.if_then_else")]
        assert len(safe_indices) == 1
        _assert_safe_b_index(safe_indices[0])

    guarded_b_indices = [
        call for call in _collect_call_nodes(body, "tirx.if_then_else") if len(call.args) == 3 and _is_load_from(call.args[1], b_data)
    ]
    assert len(guarded_b_indices) == 1


def test_ramp_load_conditions_are_combined():
    lanes = 4

    @T.prim_func
    def main(
        A: T.Tensor((16,), T.float32),
        out: T.Tensor((lanes,), T.float32),
        base: T.int32,
    ):
        out[T.Ramp(0, 1, lanes)] = A[T.Ramp(base, 1, lanes)]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = [
        call
        for call in _collect_call_nodes(body, "tirx.if_then_else")
        if len(call.args) == 3 and isinstance(call.args[1], tvm.tirx.BufferLoad)
    ]
    assert len(load_guards) == 1

    guard = load_guards[0]
    assert guard.args[0].dtype.lanes == 1
    assert isinstance(guard.args[0], tvm.tirx.And)
    assert guard.args[1].dtype.lanes == lanes
    assert isinstance(guard.args[2], tvm.tirx.Broadcast)
    assert guard.args[2].dtype.lanes == lanes

    predicate_nodes = []
    post_order_visit(guard.args[0], predicate_nodes.append)
    comparisons = [node for node in predicate_nodes if isinstance(node, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))]
    # Prefix constraints eliminate the three lower-bound checks implied by
    # `0 <= base`, while retaining the four progressively tighter upper bounds.
    assert len(comparisons) == lanes + 1
    assert all(not isinstance(node, tvm.tirx.BufferLoad) for node in predicate_nodes)
    assert all(node.dtype.lanes == 1 for node in predicate_nodes if hasattr(node, "dtype"))
    assert len(_collect_call_nodes(guard.args[0], "tirx.if_then_else")) == 0


def _collect_nodes(stmt, node_type):
    nodes = []

    def _visit(node):
        if isinstance(node, node_type):
            nodes.append(node)

    post_order_visit(stmt, _visit)
    return nodes


def _comparison_uses_var(comparison, var):
    assert isinstance(comparison, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))
    return comparison.a.same_as(var) or comparison.b.same_as(var)


def _assert_single_opaque_call(body):
    calls = _collect_call_nodes(body, "tirx.call_extern")
    assert len(calls) == 1
    assert all(str(call.args[0].value) == "tl_test_opaque_get_index" for call in calls)


def _assert_expr_guard_scope(body, index_var):
    guards = [call for call in _collect_call_nodes(body, "tirx.if_then_else") if len(call.args) == 3]
    assert len(guards) == 4
    assert all(not isinstance(guard.args[0], tvm.tirx.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = outer_lower.args[1]
    assert _comparison_uses_var(outer_lower.args[0], index_var)
    assert _comparison_uses_var(outer_upper.args[0], index_var)

    inner_lower = outer_upper.args[1]
    inner_upper = inner_lower.args[1]
    assert _is_call_to(inner_lower, "tirx.if_then_else")
    assert _is_call_to(inner_upper, "tirx.if_then_else")
    assert not _comparison_uses_var(inner_lower.args[0], index_var)
    assert not _comparison_uses_var(inner_upper.args[0], index_var)
    assert isinstance(inner_upper.args[1], tvm.tirx.BufferLoad)
    return inner_upper.args[1]


def _assert_stmt_guard_scope(body, index_var):
    guards = [guard for guard in _collect_nodes(body, tvm.tirx.IfThenElse)]
    assert len(guards) == 4
    assert all(not isinstance(guard.condition, tvm.tirx.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = outer_lower.then_case
    assert _comparison_uses_var(outer_lower.condition, index_var)
    assert _comparison_uses_var(outer_upper.condition, index_var)

    inner_lower = outer_upper.then_case
    inner_upper = inner_lower.then_case
    assert isinstance(inner_lower, tvm.tirx.IfThenElse)
    assert isinstance(inner_upper, tvm.tirx.IfThenElse)
    assert not _comparison_uses_var(inner_lower.condition, index_var)
    assert not _comparison_uses_var(inner_upper.condition, index_var)
    assert isinstance(inner_upper.then_case, tvm.tirx.BufferStore)
    return inner_upper.then_case


def test_opaque_index_guard_preserves_lazy_evaluation():
    @T.prim_func
    def main(
        A: T.Tensor((4, 4), T.float32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[
            T.call_extern("int32", "tl_test_opaque_get_index") + index,
            index,
        ]

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data

    load = _assert_expr_guard_scope(body, main.params[-1])
    assert load.buffer.data.same_as(a_data)
    assert len(_collect_call_nodes(load.indices[0], "tirx.call_extern")) == 1
    _assert_single_opaque_call(body)


def test_opaque_store_guard_preserves_lazy_evaluation():
    @T.prim_func
    def main(
        A: T.Tensor((4, 4), T.float32),
        index: T.int32,
    ):
        A[
            T.call_extern("int32", "tl_test_opaque_get_index") + index,
            index,
        ] = T.float32(1)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.buffer_map[main.params[0]].data

    store = _assert_stmt_guard_scope(body, main.params[-1])
    assert store.buffer.data.same_as(a_data)
    assert len(_collect_call_nodes(store.indices[0], "tirx.call_extern")) == 1
    _assert_single_opaque_call(body)


def test_producer_load_index_guard_preserves_lazy_evaluation():
    producer = tvm.te.placeholder((4,), dtype="int32", name="P")
    buffer = tvm.tirx.decl_buffer((4, 4), "float32", name="A")
    output = tvm.tirx.decl_buffer((1,), "float32", name="out")
    producer_index = tvm.tirx.Var("producer_index", "int32")
    later_index = tvm.tirx.Var("later_index", "int32")
    mutable_index = tvm.tirx.ProducerLoad(producer, [producer_index])
    load = tvm.tirx.BufferLoad(buffer, [mutable_index, later_index])
    body = tvm.tirx.BufferStore(output, load, [0])
    func = tvm.tirx.PrimFunc(
        [buffer.data, output.data, producer_index, later_index],
        body,
        buffer_map={buffer.data: buffer, output.data: output},
    ).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    guarded = transformed["main"].body.value
    conditions = []
    for _ in range(4):
        assert _is_call_to(guarded, "tirx.if_then_else")
        conditions.append(guarded.args[0])
        guarded = guarded.args[1]

    assert isinstance(guarded, tvm.tirx.BufferLoad)
    assert all(not isinstance(condition, tvm.tirx.And) for condition in conditions)
    assert all(_comparison_uses_var(condition, later_index) for condition in conditions[:2])
    assert all(not _collect_nodes(condition, tvm.tirx.ProducerLoad) for condition in conditions[:2])
    assert all(_collect_nodes(condition, tvm.tirx.ProducerLoad) for condition in conditions[2:])
    assert _collect_nodes(guarded.indices[0], tvm.tirx.ProducerLoad)


_DIV_OPS = {
    "truncdiv": T.truncdiv,
    "truncmod": T.truncmod,
    "floordiv": T.floordiv,
    "floormod": T.floormod,
}


def _make_division_index_load(op_name, divisor, symbolic=False):
    div_op = _DIV_OPS[op_name]

    if symbolic:

        @T.prim_func
        def main(
            A: T.Tensor((8, 8), T.float32),
            out: T.Tensor((1,), T.float32),
            index: T.int32,
            divisor: T.int32,
        ):
            out[0] = A[div_op(index, divisor), index]
    else:

        @T.prim_func
        def main(
            A: T.Tensor((8, 8), T.float32),
            out: T.Tensor((1,), T.float32),
            index: T.int32,
        ):
            out[0] = A[div_op(index, divisor), index]

    return main


def _division_load_guards(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    return _collect_call_nodes(transformed["main"].body, "tirx.if_then_else")


@pytest.mark.parametrize("op_name", list(_DIV_OPS))
def test_positive_constant_divisors_flatten_safe_load_guards(op_name):
    guards = _division_load_guards(_make_division_index_load(op_name, 16))

    assert len(guards) == 1
    assert isinstance(guards[0].args[0], tvm.tirx.And)
    assert isinstance(guards[0].args[1], tvm.tirx.BufferLoad)
    assert not _collect_call_nodes(guards[0].args[0], "tirx.if_then_else")


@pytest.mark.parametrize("op_name", list(_DIV_OPS))
def test_negative_constant_divisors_keep_safe_load_guards_nested(op_name):
    guards = _division_load_guards(_make_division_index_load(op_name, -16))

    assert len(guards) > 1
    assert all(not isinstance(guard.args[0], tvm.tirx.And) for guard in guards)


def test_symbolic_divisor_keeps_safe_load_guards_nested():
    guards = _division_load_guards(_make_division_index_load("truncdiv", None, symbolic=True))

    assert len(guards) > 1
    assert all(not isinstance(guard.args[0], tvm.tirx.And) for guard in guards)


if __name__ == "__main__":
    tilelang.testing.main()
