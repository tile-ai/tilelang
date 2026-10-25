from testing.python.transform._native_ir import make_function
from tilelang._compat.tensor import tensor_parameter_map

import math

import pytest
import tvm_ffi

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
from tilelang._compat.stmt_functor import ir_transform


def _strip_block_reads_writes(stmt, strip_annotations: bool = False):
    """Strip non-behavioral block metadata before structural comparison."""

    def _postorder(op):
        if isinstance(op, tvm.ir.RegionStmt):
            annotations = (
                {}
                if strip_annotations
                else {key: op.attrs[key] for key in op.attrs.keys() if key not in ("reads", "writes", "safe_value_map")}
            )
            return tvm.ir.RegionStmt(op.op, op.args, op.body_params, annotations, op.body, op.result_vars, op.loc)
        if isinstance(op, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE)):
            promoted = tvm.get_global_func("tl.PromoteIndexToInt64")(op)
            return tvm.arith.Analyzer().canonical_simplify(promoted)

    function = tvm.tirx.PrimFunc(tvm.tirx.analysis.undefined_vars(stmt), ir_transform(stmt, None, _postorder))
    mod = tl.transform.LowerAccessPtr()(tvm.IRModule({"main": function}))

    def _canonicalize(op):
        if isinstance(op, tvm.tirx.And) and tvm_ffi.structural_equal(op.a, op.b):
            return op.a
        if isinstance(op, tvm.tirx.LE) and isinstance(op.b, tvm.ir.IntImm) and int(op.b) < (1 << (op.b.ty.dtype.bits - 1)) - 1:
            return op.a < tvm.ir.const(int(op.b) + 1, op.b.ty)
        if isinstance(op, tvm.ir.Call) and op.op.name == "tirx.address_of" and op.attrs is not None:
            attributes = {key: value for key, value in op.attrs.items() if key != "tl.access_region"}
            return tvm.ir.Call(op.op, op.args, attrs=attributes, ty_args=op.ty_args, ty=op.ty, loc=op.loc)

    return ir_transform(tl.transform.Simplify()(mod)["main"].body, None, _canonicalize)


def _materialize_launch(func):
    """Run the launch materialization the pipeline performs before
    LegalizeSafeMemoryAccess, so thread indices carry their extents."""
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target("cuda"))(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    return mod[func.attrs["global_symbol"]]


def _is_call_to(expr, op_name):
    return isinstance(expr, tvm.tirx.Call) and isinstance(expr.op, tvm.ir.Op) and str(expr.op.name) == op_name


def _is_int_zero(expr):
    return isinstance(expr, tvm.tirx.IntImm) and int(expr.value) == 0


def _assert_tl_access_ptr_bases_are_tensor_accesses(stmt):
    for call in collect_calls(stmt, op="tl.access_ptr"):
        assert isinstance(call.args[0], (tvm.ir.TensorLoad, tvm.ir.TensorRegion))


def _assert_legalize_matches_expected(before, expected, strip_annotations: bool = False):
    mod = tvm.IRModule({before.attrs["global_symbol"]: before})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    _assert_tl_access_ptr_bases_are_tensor_accesses(before.body)
    _assert_tl_access_ptr_bases_are_tensor_accesses(transformed["main"].body)
    tvm.ir.assert_structural_equal(
        _strip_block_reads_writes(transformed["main"].body, strip_annotations),
        _strip_block_reads_writes(expected.body, strip_annotations),
        map_free_vars=True,
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
        map_free_vars=True,
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
                if tid < M - M_offset and j < N - N_offset:
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
        map_free_vars=True,
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
        map_free_vars=True,
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
    cp_async_calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 4 for call in cp_async_calls)
    _assert_legalize_matches_expected(func, expected)


def cp_async_access_ptr_nonzero_safe_value_legalize():
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((16,), dtype=dtype),
    ):
        with T.scope():
            T.annotate_safe_value({A: T.float16(3)})
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
        with T.scope():
            T.annotate_safe_value({A: T.float16(3)})
            A_shared = T.alloc_buffer((16,), dtype=dtype, scope="shared")
            for i in T.serial(4):
                if i < 2:
                    T.ptx_cp_async(
                        T.access_ptr(A_shared[i * 4], "w", 4),
                        T.access_ptr(A[i * 4 + 8], "r", 4),
                        4,
                    )
                else:
                    for lane in T.serial(4):
                        A_shared[i * 4 + lane] = T.float16(3)
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
            if i < 2:
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
    cp_async_calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 3 for call in cp_async_calls)
    assert collect_nodes(body, tvm.tirx.IfThenElse)
    _assert_legalize_matches_expected(func, expected, strip_annotations=True)


def assert_atomic_load_access_ptr_legalize():
    func, expected = atomic_load_access_ptr_legalize()
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_tensor_accesses(body)
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
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body

    _assert_tl_access_ptr_bases_are_tensor_accesses(body)
    assert collect_nodes(body, tvm.tirx.IfThenElse)
    _assert_legalize_matches_expected(func, expected)


def assert_call_extern_multiple_access_ptrs_legalize():
    func, expected = call_extern_multiple_access_ptrs_legalize()
    _assert_legalize_matches_expected(func, expected)


def assert_runtime_unknown_sign_vector_negative_index_legalize():
    import numpy as np

    @T.prim_func
    def main(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        for t in T.serial(4):
            B[t, T.Ramp(0, 1, 4)] = A[T.Ramp(t - 2, 1, 4)]

    @T.prim_func
    def expected(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
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
    values = tvm.runtime.tensor(np.arange(1024, dtype="float32"))
    actual = tvm.runtime.empty((4, 4), "float32")
    reference = tvm.runtime.empty((4, 4), "float32")
    tvm.tirx.compile(transformed, target="llvm")(values, actual)
    tvm.tirx.compile(expected, target="llvm")(values, reference)
    np.testing.assert_array_equal(actual.numpy(), reference.numpy())
    np.testing.assert_array_equal(actual.numpy(), (np.arange(4)[:, None] - 2 + np.arange(4)[None, :]) % 1024)


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
        with T.scope():
            T.annotate_safe_value({A: T.float32(safe_value)})
            shared = T.alloc_buffer((2,), T.float32, scope="shared")
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
    calls = collect_calls(body, op="tl.ptx_cp_async")
    assert len(calls) == 1
    if math.copysign(1.0, safe_value) > 0:
        assert len(calls[0].args) == 4
        assert not collect_nodes(body, tvm.tirx.IfThenElse)
    else:
        assert len(calls[0].args) == 3
        assert len(collect_nodes(body, tvm.tirx.IfThenElse)) == 1
        stores = collect_nodes(body, tvm.tirx.BufferStore)
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

    load_guards = collect_calls(body, op="prim.if_then_else")
    assert len(load_guards) == 1
    assert isinstance(load_guards[0].args[0], tvm.tirx.And)
    assert isinstance(load_guards[0].args[1], tvm.tirx.BufferLoad)
    assert isinstance(load_guards[0].args[2], tvm.tirx.FloatImm)
    assert float(load_guards[0].args[2].value) == 0.0

    store_guards = collect_nodes(body, tvm.tirx.IfThenElse)
    assert len(store_guards) == 1
    assert isinstance(store_guards[0].condition, tvm.tirx.And)
    assert isinstance(store_guards[0].then_case.seq[0], tvm.tirx.BufferStore)
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
    source, indices = main.params[:2]

    def _is_load_from(expr, tensor):
        return isinstance(expr, tvm.ir.TensorLoad) and expr.source.same_as(tensor)

    def _assert_safe_b_index(expr):
        assert _is_call_to(expr, "prim.if_then_else")
        assert isinstance(expr.args[0], tvm.tirx.And)
        assert _is_load_from(expr.args[1], indices)
        assert _is_int_zero(expr.args[2])

    all_guards = collect_calls(body, op="prim.if_then_else")
    inner_guards = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], source)]
    assert len(inner_guards) == 1

    inner_guard = inner_guards[0]
    assert isinstance(inner_guard.args[2], tvm.ir.FloatImm) and float(inner_guard.args[2].value) == 0.0
    snapshots = collect_nodes(body, tvm.tirx.Let)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.body.same_as(inner_guard)
    assert inner_guard.args[1].indices[0].same_as(snapshot.var)
    _assert_index_bounds(inner_guard.args[0], snapshot.var)
    _assert_safe_b_index(snapshot.value)

    guarded_b_indices = [call for call in all_guards if len(call.args) == 3 and _is_load_from(call.args[1], indices)]
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
        for call in collect_calls(body, op="prim.if_then_else")
        if len(call.args) == 3 and isinstance(call.args[1], tvm.tirx.BufferLoad)
    ]
    assert len(load_guards) == 1

    guard = load_guards[0]
    assert guard.args[0].ty.dtype.lanes == 1
    assert isinstance(guard.args[0], tvm.tirx.And)
    assert guard.args[1].ty.dtype.lanes == lanes
    assert isinstance(guard.args[2], tvm.tirx.Broadcast)
    assert guard.args[2].ty.dtype.lanes == lanes

    predicate_nodes = collect_nodes(guard.args[0], tvm.ffi.Object)
    comparisons = [node for node in predicate_nodes if isinstance(node, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))]
    assert comparisons
    base = main.params[-1]
    analyzer = tvm.arith.Analyzer()
    for offset in (-2147483648, -4, -1, 0, 12, 13, 15, 16, 2147483647):
        condition = tvm.tirx.stmt_functor.substitute(guard.args[0], {base: tvm.ir.const(offset, "int32")})
        assert bool(int(analyzer.simplify(condition))) == (0 <= offset and offset + lanes <= 16)
    assert all(not isinstance(node, tvm.tirx.BufferLoad) for node in predicate_nodes)
    assert all(node.dtype.lanes == 1 for node in predicate_nodes if hasattr(node, "dtype"))
    assert_call_count(guard.args[0], op="prim.if_then_else", count=0)


def _comparison_uses_var(comparison, var):
    assert isinstance(comparison, (tvm.tirx.LT, tvm.tirx.LE, tvm.tirx.GT, tvm.tirx.GE))
    return any(variable.same_as(var) for variable in collect_nodes(comparison, tvm.ir.Var))


def _assert_index_bounds(condition, variable):
    assert isinstance(condition, tvm.tirx.And)
    comparisons = [node for node in collect_nodes(condition, tvm.ffi.Object) if isinstance(node, (tvm.tirx.LT, tvm.tirx.LE))]
    assert len(comparisons) == 2
    assert all(_comparison_uses_var(comparison, variable) for comparison in comparisons)


def _assert_single_opaque_call(body):
    calls = collect_calls(body, op="tirx.call_extern")
    assert len(calls) == 1
    assert all(str(call.args[0].value) == "tl_test_opaque_get_index" for call in calls)


def _assert_expr_guard_scope(body, index_var):
    guards = [call for call in collect_calls(body, op="prim.if_then_else") if len(call.args) == 3]
    assert len(guards) == 2
    outer = guards[-1]
    _assert_index_bounds(outer.args[0], index_var)
    snapshot = outer.args[1]
    assert isinstance(snapshot, tvm.tirx.Let)
    inner = snapshot.body
    assert _is_call_to(inner, "prim.if_then_else")
    _assert_index_bounds(inner.args[0], snapshot.var)
    assert isinstance(inner.args[1], tvm.ir.TensorLoad)
    assert inner.args[1].indices[0].same_as(snapshot.var)
    return inner.args[1], snapshot


def _assert_stmt_guard_scope(body, index_var):
    guards = collect_nodes(body, tvm.tirx.IfThenElse)
    assert len(guards) == 2
    outer = guards[-1]
    _assert_index_bounds(outer.condition, index_var)
    snapshot, inner = outer.then_case.seq
    assert isinstance(snapshot, tvm.ir.Bind)
    assert isinstance(inner, tvm.ir.If)
    _assert_index_bounds(inner.condition, snapshot.var)
    store = inner.then_case.seq[0]
    assert isinstance(store, tvm.ir.TensorStore)
    assert store.indices[0].same_as(snapshot.var)
    return store, snapshot


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
    load, snapshot = _assert_expr_guard_scope(body, main.params[-1])
    assert load.source.same_as(main.params[0])
    assert_call_count(snapshot.value, op="tirx.call_extern", count=1)
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
    store, snapshot = _assert_stmt_guard_scope(body, main.params[-1])
    assert store.dest.same_as(main.params[0])
    assert_call_count(snapshot.value, op="tirx.call_extern", count=1)
    _assert_single_opaque_call(body)


def test_tensor_producer_index_guard_preserves_lazy_evaluation():
    producer = tvm.tirx.decl_buffer((4,), dtype="int32", name="P")
    buffer = tvm.tirx.decl_buffer((4, 4), "float32", name="A")
    output = tvm.tirx.decl_buffer((1,), "float32", name="out")
    producer_index = tvm.tirx.Var("producer_index", "int32")
    later_index = tvm.tirx.Var("later_index", "int32")
    mutable_index = tvm.ir.TensorLoad(producer, [producer_index])
    load = tvm.tirx.BufferLoad(buffer, [mutable_index, later_index])
    body = tvm.tirx.BufferStore(output, load, [0])
    func = tvm.tirx.PrimFunc([buffer, output, producer, producer_index, later_index], body).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)

    guarded = transformed["main"].body.seq[0].value
    assert _is_call_to(guarded, "prim.if_then_else")
    _assert_index_bounds(guarded.args[0], later_index)
    assert not collect_nodes(guarded.args[0], tvm.ir.TensorLoad)
    snapshot = guarded.args[1]
    assert isinstance(snapshot, tvm.tirx.Let)
    producer_loads = collect_nodes(snapshot.value, tvm.ir.TensorLoad)
    assert len(producer_loads) == 1 and producer_loads[0].source.same_as(producer)
    _assert_index_bounds(snapshot.body.args[0], snapshot.var)
    assert snapshot.body.args[1].source.same_as(buffer)
    assert snapshot.body.args[1].indices[0].same_as(snapshot.var)


if __name__ == "__main__":
    tilelang.testing.main()
