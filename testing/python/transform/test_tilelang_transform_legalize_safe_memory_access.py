from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm.tir.stmt_functor import ir_transform, post_order_visit


def _strip_block_reads_writes(stmt):
    """Strip reads and writes from all blocks, replacing them with empty lists."""

    def _postorder(op):
        if isinstance(op, tvm.tir.Block):
            return tvm.tir.Block(
                op.iter_vars,
                [],
                [],
                op.name_hint,
                op.body,
                op.init,
                op.alloc_buffers,
                op.match_buffers,
                op.annotations,
            )

    return ir_transform(stmt, None, _postorder, ["tir.Block"])


def _collect_call_nodes(stmt, op_names):
    if isinstance(op_names, str):
        op_names = {op_names}
    else:
        op_names = set(op_names)
    calls = []

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) in op_names:
            calls.append(node)

    post_order_visit(stmt, _visit)
    return calls


def _count_if_then_else(stmt):
    count = 0

    def _visit(node):
        nonlocal count
        if isinstance(node, tvm.tir.IfThenElse):
            count += 1

    post_order_visit(stmt, _visit)
    return count


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

            T.reads(A[tid + M_offset, N_offset : N + N_offset])
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

            T.reads(A[tid + M_offset, N_offset : N + N_offset])
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
            T.writes(A[tid + M_offset, N_offset : N + N_offset])
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


def cp_async_access_ptr_legalize(N: int = 16, offset: int = 10):
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype=dtype),
    ):
        A_shared = T.alloc_buffer((N,), dtype=dtype, scope="shared")
        for i in T.serial(4):
            T.ptx_cp_async(
                T.access_ptr(A_shared[i * 4], "w", 4),
                T.access_ptr(A[i * 4 + offset], "r", 4),
                4,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)

    @T.prim_func
    def expected(
        A: T.Tensor((N,), dtype=dtype),
    ):
        A_shared = T.alloc_buffer((N,), dtype=dtype, scope="shared")
        for i in T.serial(4):
            T.ptx_cp_async(
                T.access_ptr(A_shared[i * 4], "w", 4),
                T.access_ptr(A[i * 4 + offset], "r", 4),
                4,
                i * 4 + offset < N,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)

    return main, expected


def assert_cp_async_access_ptr_legalize(N: int = 16):
    func, _ = cp_async_access_ptr_legalize(N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = _collect_call_nodes(body, {"tir.ptx_cp_async", "tl.ptx_cp_async"})
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 4 for call in cp_async_calls)


def cp_async_access_ptr_nonzero_safe_value_legalize(N: int = 16, offset: int = 10, pad_value: int = 3):
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype=dtype),
    ):
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"safe_value_map": {A.data: T.float16(pad_value)}})
            A_shared = T.alloc_buffer((N,), dtype=dtype, scope="shared")
            for i in T.serial(4):
                T.ptx_cp_async(
                    T.access_ptr(A_shared[i * 4], "w", 4),
                    T.access_ptr(A[i * 4 + offset], "r", 4),
                    4,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)

    @T.prim_func
    def expected(
        A: T.Tensor((N,), dtype=dtype),
    ):
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"safe_value_map": {A.data: T.float16(pad_value)}})
            A_shared = T.alloc_buffer((N,), dtype=dtype, scope="shared")
            for i in T.serial(4):
                if i * 4 + offset < N:
                    T.ptx_cp_async(
                        T.access_ptr(A_shared[i * 4], "w", 4),
                        T.access_ptr(A[i * 4 + offset], "r", 4),
                        4,
                    )
            T.ptx_commit_group()
            T.ptx_wait_group(0)

    return main, expected


def assert_cp_async_access_ptr_nonzero_safe_value_legalize(N: int = 16):
    func, _ = cp_async_access_ptr_nonzero_safe_value_legalize(N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    body = transformed["main"].body
    cp_async_calls = _collect_call_nodes(body, {"tir.ptx_cp_async", "tl.ptx_cp_async"})
    assert len(cp_async_calls) > 0
    assert all(len(call.args) == 3 for call in cp_async_calls)
    assert _count_if_then_else(body) > 0


def test_vectorize_access():
    assert_vectorize_access(64, 64)


def test_vectorize_access_with_atmoic_add():
    assert_vectorize_access_with_atmoic_add(64, 64)


def test_oob_store():
    assert_oob_store_legalize(64, 64)


def test_cp_async_access_ptr_oob():
    assert_cp_async_access_ptr_legalize(16)


def test_cp_async_access_ptr_nonzero_safe_value_oob():
    assert_cp_async_access_ptr_nonzero_safe_value_legalize(16)


if __name__ == "__main__":
    tilelang.testing.main()
