import math

import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm_tirx
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
import tvm_ffi
from tvm_tirx import ir, tirx
from testing.python.transform._safe_memory_oracle import (
    assert_nested_lookup,
    assert_nonzero_async_padding,
    assert_predicate_table,
    assert_same_memory_effects,
)


def _single_stmt(stmt):
    while isinstance(stmt, ir.SeqStmt) and len(stmt.seq) == 1:
        stmt = stmt.seq[0]
    return stmt


def _operand_count(call):
    return int(call.attrs.get("operand_count", len(call.args))) if call.attrs else len(call.args)


def _materialize_launch(func):
    """Run the launch materialization the pipeline performs before
    LegalizeSafeMemoryAccess, so thread indices carry their extents."""
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    mod = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda"))(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    return mod[func.attrs["global_symbol"]]


def _is_call_to(expr, op_name):
    return isinstance(expr, ir.Call) and isinstance(expr.op, tvm_tirx.ir.Op) and str(expr.op.name) == op_name


def _is_int_zero(expr):
    return isinstance(expr, ir.IntImm) and int(expr.value) == 0


def _assert_tl_access_ptr_bases_are_buffer_loads(stmt):
    for call in collect_calls(stmt, op="tl.access_ptr"):
        assert isinstance(call.args[0], ir.TensorLoad)


def _assert_legalize_matches_expected(before, expected, strip_annotations: bool = False):
    mod = tvm_tirx.IRModule({before.attrs["global_symbol"]: before})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    _assert_tl_access_ptr_bases_are_buffer_loads(before.body)
    _assert_tl_access_ptr_bases_are_buffer_loads(transformed["main"].body)
    assert_same_memory_effects(transformed["main"], expected)


def _apply_negative_index_then_safe_memory(func):
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    assert_same_memory_effects(transformed["main"], expected)


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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    print(transformed)
    print(expected)
    assert_same_memory_effects(transformed["main"], expected)


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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    assert_same_memory_effects(transformed["main"], expected)


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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(cp_async_calls) > 0
    # Multi-element transfers retain their valid prefix through a guarded
    # scalar fallback. One cp.async predicate alone cannot express that case.
    assert all(_operand_count(call) == 3 for call in cp_async_calls)
    assert collect_nodes(body, ir.If)
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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(cp_async_calls) > 0
    assert all(_operand_count(call) == 3 for call in cp_async_calls)
    assert collect_nodes(body, ir.If)
    _assert_tl_access_ptr_bases_are_buffer_loads(body)
    assert_nonzero_async_padding(transformed["main"], expected)


def assert_atomic_load_access_ptr_legalize():
    func, expected = atomic_load_access_ptr_legalize()
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_buffer_loads(body)
    if_then_else_calls = collect_calls(body, op="prim.if_then_else")
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
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_buffer_loads(body)
    assert collect_nodes(body, ir.If)
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
    assert_same_memory_effects(transformed["main"], expected)


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

    mod = tl.transform.LegalizeSafeMemoryAccess()(tvm_tirx.IRModule.from_expr(main))
    body = mod["main"].body
    calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(calls) == 1
    if math.copysign(1.0, safe_value) > 0:
        assert len(calls[0].args) == 4
        assert not collect_nodes(body, ir.If)
    else:
        assert len(calls[0].args) == 3
        assert len(collect_nodes(body, ir.If)) == 1
        stores = collect_nodes(body, ir.TensorStore)
        assert len(stores) == 1
        fill = stores[0].value
        # The fallback retains a lazy source guard, so an invalid source is
        # never dereferenced. Its padding value must preserve the sign bit.
        assert _is_call_to(fill, "prim.if_then_else")
        assert isinstance(fill.args[1], ir.TensorLoad)
        fill = fill.args[2]
        assert isinstance(fill, ir.FloatImm)
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

    mod = tvm_tirx.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = collect_calls(body, op="prim.if_then_else")
    assert len(load_guards) == 1
    assert isinstance(load_guards[0].args[0], ir.prim.And)
    assert isinstance(load_guards[0].args[1], ir.TensorLoad)
    assert isinstance(load_guards[0].args[2], ir.FloatImm)
    assert float(load_guards[0].args[2].value) == 0.0

    store_guards = collect_nodes(body, ir.If)
    assert len(store_guards) == 1
    assert isinstance(store_guards[0].condition, ir.prim.And)
    assert isinstance(_single_stmt(store_guards[0].then_case), ir.TensorStore)
    tvm_tirx.ir.assert_structural_equal(load_guards[0].args[0], store_guards[0].condition)


def test_nested_buffer_load_index_is_safely_rewritten():
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((4,), T.int32),
        out: T.Tensor((1,), T.float32),
        index: T.int32,
    ):
        out[0] = A[B[index]]

    mod = tvm_tirx.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.params[0]
    b_data = main.params[1]

    def _is_load_from(expr, buffer_data):
        return isinstance(expr, ir.TensorLoad) and expr.source.same_as(buffer_data)

    def _assert_safe_b_index(expr):
        assert _is_call_to(expr, "prim.if_then_else")
        assert isinstance(expr.args[0], ir.prim.And)
        assert _is_load_from(expr.args[1], b_data)
        assert _is_int_zero(expr.args[2])

    all_guards = collect_calls(body, op="prim.if_then_else")
    inner_guards = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], a_data)]
    assert len(inner_guards) == 1

    inner_guard = inner_guards[0]
    outer_guards = [call for call in all_guards if len(call.args) == 3 and call.args[1].same_as(inner_guard)]
    assert len(outer_guards) == 1

    outer_guard = outer_guards[0]
    for fallback in (inner_guard.args[2], outer_guard.args[2]):
        assert isinstance(fallback, ir.FloatImm)
        assert float(fallback.value) == 0.0
    _assert_safe_b_index(inner_guard.args[1].indices[0])

    for comparison in (inner_guard.args[0], outer_guard.args[0]):
        assert isinstance(comparison, (ir.prim.LT, ir.prim.LE, ir.prim.GT, ir.prim.GE))
        safe_indices = [operand for operand in (comparison.a, comparison.b) if _is_call_to(operand, "prim.if_then_else")]
        assert len(safe_indices) == 1
        _assert_safe_b_index(safe_indices[0])

    guarded_b_indices = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], b_data)]
    # The typed mutator may rebuild a pure producer expression at each use.
    # Every such expression must keep the same lazy bound check and fallback.
    assert guarded_b_indices
    for call in guarded_b_indices:
        _assert_safe_b_index(call)
        tvm_tirx.ir.assert_structural_equal(call, guarded_b_indices[0])
    producer_loads = [load for load in collect_nodes(body, ir.TensorLoad) if load.source.same_as(b_data)]
    assert {load for load in producer_loads} == {call.args[1] for call in guarded_b_indices}
    assert_nested_lookup(transformed["main"])


def test_ramp_load_conditions_are_combined():
    lanes = 4

    @T.prim_func
    def main(
        A: T.Tensor((16,), T.float32),
        out: T.Tensor((lanes,), T.float32),
        base: T.int32,
    ):
        out[T.Ramp(0, 1, lanes)] = A[T.Ramp(base, 1, lanes)]

    mod = tvm_tirx.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    load_guards = [
        call for call in collect_calls(body, op="prim.if_then_else") if len(call.args) == 3 and isinstance(call.args[1], ir.TensorLoad)
    ]
    assert len(load_guards) == 1

    guard = load_guards[0]
    assert guard.args[0].ty.dtype.lanes == 1
    assert isinstance(guard.args[0], ir.prim.And)
    assert guard.args[1].ty.dtype.lanes == lanes
    assert isinstance(guard.args[2], ir.prim.Broadcast)
    assert guard.args[2].ty.dtype.lanes == lanes

    predicate_nodes = collect_nodes(guard.args[0], tvm_ffi.Object)
    comparisons = [node for node in predicate_nodes if isinstance(node, (ir.prim.LT, ir.prim.LE, ir.prim.GT, ir.prim.GE))]
    # Target analysis retains a lower and upper bound for each lane. Preserve
    # the scalar, memory-free combined guard and independently check its truth
    # table instead of requiring the old Analyzer's redundant-check removal.
    assert len(comparisons) == 2 * lanes
    assert_predicate_table(
        transformed["main"],
        guard.args[0],
        main.params[-1],
        [(base, 0 <= base and base + lanes <= 16) for base in [-(2**31), *range(-4, 21), 2**31 - 4, 2**31 - 1]],  # noqa: SIM300 -- Interval oracle notation.
    )
    assert all(not isinstance(node, ir.TensorLoad) for node in predicate_nodes)
    assert all(node.ty.dtype.lanes == 1 for node in predicate_nodes if isinstance(node, ir.Expr) and isinstance(node.ty, ir.PrimType))
    assert_call_count(guard.args[0], op="prim.if_then_else", count=0)


def _comparison_uses_var(comparison, var):
    assert isinstance(comparison, (ir.prim.LT, ir.prim.LE, ir.prim.GT, ir.prim.GE))
    return comparison.a.same_as(var) or comparison.b.same_as(var)


def _assert_single_opaque_call(body):
    calls = collect_calls(body, op="tirx.call_extern")
    assert len(calls) == 1
    assert all(str(call.args[0].value) == "tl_test_opaque_get_index" for call in calls)


def _assert_expr_guard_scope(body, index_var):
    guards = [call for call in collect_calls(body, op="prim.if_then_else") if len(call.args) == 3]
    assert len(guards) == 4
    assert all(not isinstance(guard.args[0], ir.prim.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = outer_lower.args[1]
    assert _comparison_uses_var(outer_lower.args[0], index_var)
    assert _comparison_uses_var(outer_upper.args[0], index_var)

    inner_lower = outer_upper.args[1]
    inner_upper = inner_lower.args[1]
    assert _is_call_to(inner_lower, "prim.if_then_else")
    assert _is_call_to(inner_upper, "prim.if_then_else")
    assert not _comparison_uses_var(inner_lower.args[0], index_var)
    assert not _comparison_uses_var(inner_upper.args[0], index_var)
    assert isinstance(inner_upper.args[1], ir.TensorLoad)
    return inner_upper.args[1]


def _assert_stmt_guard_scope(body, index_var):
    guards = collect_nodes(body, ir.If)
    assert len(guards) == 4
    assert all(not isinstance(guard.condition, ir.prim.And) for guard in guards)

    outer_lower = guards[-1]
    outer_upper = _single_stmt(outer_lower.then_case)
    assert _comparison_uses_var(outer_lower.condition, index_var)
    assert _comparison_uses_var(outer_upper.condition, index_var)

    inner_lower = _single_stmt(outer_upper.then_case)
    inner_upper = _single_stmt(inner_lower.then_case)
    assert isinstance(inner_lower, ir.If)
    assert isinstance(inner_upper, ir.If)
    assert not _comparison_uses_var(inner_lower.condition, index_var)
    assert not _comparison_uses_var(inner_upper.condition, index_var)
    assert isinstance(_single_stmt(inner_upper.then_case), ir.TensorStore)
    return _single_stmt(inner_upper.then_case)


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

    mod = tvm_tirx.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.params[0]

    load = _assert_expr_guard_scope(body, main.params[-1])
    assert load.source.same_as(a_data)
    assert_call_count(load.indices[0], op="tirx.call_extern", count=1)
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

    mod = tvm_tirx.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    a_data = main.params[0]

    store = _assert_stmt_guard_scope(body, main.params[-1])
    assert store.dest.same_as(a_data)
    assert_call_count(store.indices[0], op="tirx.call_extern", count=1)
    _assert_single_opaque_call(body)


def test_producer_load_index_guard_preserves_lazy_evaluation():
    producer = ir.Var("P", tirx.TensorType((4,), "int32"))
    buffer = ir.Var("A", tirx.TensorType((4, 4), "float32"))
    output = ir.Var("out", tirx.TensorType((1,), "float32"))
    producer_index = ir.Var("producer_index", "int32")
    later_index = ir.Var("later_index", "int32")
    mutable_index = ir.TensorLoad(producer, [producer_index])
    load = ir.TensorLoad(buffer, [mutable_index, later_index])
    body = ir.TensorStore(output, [0], load)
    func = tirx.Function(
        [buffer, output, producer, producer_index, later_index],
        [body],
    ).with_attr("global_symbol", "main")
    mod = tvm_tirx.IRModule.from_expr(func)
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    guarded = _single_stmt(transformed["main"].body).value
    conditions = []
    for _ in range(4):
        assert _is_call_to(guarded, "prim.if_then_else")
        conditions.append(guarded.args[0])
        guarded = guarded.args[1]

    assert isinstance(guarded, ir.TensorLoad)
    assert all(not isinstance(condition, ir.prim.And) for condition in conditions)
    assert all(_comparison_uses_var(condition, later_index) for condition in conditions[:2])
    assert all(
        not [load for load in collect_nodes(condition, ir.TensorLoad) if load.source.same_as(producer)] for condition in conditions[:2]
    )
    assert all([load for load in collect_nodes(condition, ir.TensorLoad) if load.source.same_as(producer)] for condition in conditions[2:])
    assert [load for load in collect_nodes(guarded.indices[0], ir.TensorLoad) if load.source.same_as(producer)]


if __name__ == "__main__":
    tilelang.testing.main()
