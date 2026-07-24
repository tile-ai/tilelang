import tilelang
from tilelang import tvm as tvm
from tvm.ir import IRModule
from tvm.tirx.stmt_functor import post_order_visit
import tilelang.testing
import tilelang.language as T


def collect_nodes(stmt, node_type):
    nodes = []

    def visit(node):
        if isinstance(node, node_type):
            nodes.append(node)

    post_order_visit(stmt, visit)
    return nodes


def count_if_then_else(stmt):
    return len(collect_nodes(stmt, tvm.tirx.IfThenElse))


def count_calls_to(stmt, op_name):
    return sum(
        isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op_name
        for node in collect_nodes(stmt, tvm.tirx.Call)
    )


def apply_merge_if(func, lower_access_ptr=False):
    module = IRModule.from_expr(func)
    if lower_access_ptr:
        module = tilelang.transform.LowerAccessPtr()(module)
    before_merge = module
    after_merge = tilelang.transform.MergeIfStmt()(module)
    return before_merge["main"], after_merge["main"]


def assert_merge_if_unchanged(func, lower_access_ptr=False):
    before_merge, after_merge = apply_merge_if(func, lower_access_ptr)
    tvm.ir.assert_structural_equal(before_merge, after_merge, True)
    assert count_if_then_else(after_merge.body) == 2


def merge_if_test():
    @T.prim_func
    def main():
        A = T.alloc_fragment((1,), T.float16)
        B = T.alloc_fragment((1,), T.float16)
        C = T.alloc_fragment((1,), T.float16)
        D = T.alloc_fragment((1,), T.float16)
        if A[0] == 0:
            A[0] = 0
        if B[0] == 0:
            B[0] = 0
        if C[0] == 0:
            C[0] = 0
        if D[0] == 0:
            D[0] = 0

    return main


def merge_if_written_buffer_test():
    """Test: second if's condition reads buffer written in first if's body.

    Should NOT merge because the second if's condition (A[0] == 0) reads A,
    which the first if's body writes to (A[0] = 7). Merging would incorrectly
    reuse the first condition and skip re-evaluating the second one.
    """

    @T.prim_func
    def main():
        A = T.alloc_fragment((1,), T.int32)
        Out = T.alloc_fragment((1,), T.int32)
        if A[0] == 0:
            A[0] = 7
        if A[0] == 0:
            Out[0] = 20

    return main


def merge_if_safe_test():
    """Test: same condition, body writes a different buffer (safe to merge).

    Both ifs share condition A[0] == 0, which reads buffer A.
    Neither body writes to A (they write to Out), so the condition cannot be
    invalidated. The pass should merge these into a single if.
    """

    @T.prim_func
    def main():
        A = T.alloc_fragment((2,), T.int32)
        Out = T.alloc_fragment((2,), T.int32)
        if A[0] == 0:
            Out[0] = 1
        if A[0] == 0:
            Out[1] = 2

    return main


def merge_if_atomic_write_test():
    """An atomic write to the condition buffer must prevent merging."""

    @T.prim_func
    def main():
        condition_buffer = T.alloc_fragment((1,), T.int32)
        output = T.alloc_fragment((1,), T.int32)
        if condition_buffer[0] == 0:
            T.atomic_store(condition_buffer[0], 7)
        if condition_buffer[0] == 0:
            output[0] = 20

    return main


def merge_if_atomic_read_condition_test():
    """A call-based condition read must observe writes from prior bodies."""

    @T.prim_func
    def main():
        condition_buffer = T.alloc_fragment((1,), T.int32)
        output = T.alloc_fragment((1,), T.int32)
        if T.atomic_load(condition_buffer[0]) == 0:
            condition_buffer[0] = 7
        if T.atomic_load(condition_buffer[0]) == 0:
            output[0] = 20

    return main


def merge_if_safe_atomic_write_test():
    """An atomic write to an unrelated buffer must remain mergeable."""

    @T.prim_func
    def main():
        condition_buffer = T.alloc_fragment((1,), T.int32)
        output = T.alloc_fragment((2,), T.int32)
        if condition_buffer[0] == 0:
            T.atomic_store(output[0], 1)
        if condition_buffer[0] == 0:
            output[1] = 2

    return main


def merge_if_address_of_write_test():
    """An opaque pointer call may invalidate a buffer-backed condition."""

    @T.prim_func
    def main():
        condition_buffer = T.alloc_fragment((1,), T.int32)
        output = T.alloc_fragment((1,), T.int32)
        if condition_buffer[0] == 0:
            T.evaluate(
                T.call_extern(
                    "handle",
                    "custom_ptr_store",
                    T.address_of(condition_buffer[0]),
                )
            )
        if condition_buffer[0] == 0:
            output[0] = 20

    return main


def test_merge_if():
    func = merge_if_test()
    original_module = IRModule.from_expr(func)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    tvm.ir.assert_structural_equal(original_module["main"], transformed["main"], True)


def test_merge_if_written_buffer():
    """Regression test for #2538: if conditions must not be invalidated by prior body writes.

    When two ifs share the same condition and the second's condition reads a buffer
    that the first's body writes to, the pass must NOT merge them.
    """
    func = merge_if_written_buffer_test()
    original_module = IRModule.from_expr(func)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    tvm.ir.assert_structural_equal(original_module["main"], transformed["main"], True)
    assert count_if_then_else(transformed["main"].body) == 2


def test_merge_if_safe():
    """Test that safe cases (same condition, body writes different buffer) are merged."""
    func = merge_if_safe_test()
    original_module = IRModule.from_expr(func)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    assert count_if_then_else(original_module["main"].body) == 2
    assert count_if_then_else(transformed["main"].body) == 1
    merged_if = collect_nodes(transformed["main"].body, tvm.tirx.IfThenElse)[0]
    assert len(collect_nodes(merged_if.then_case, tvm.tirx.BufferStore)) == 2


def test_merge_if_atomic_write():
    """Cover tl.access_ptr and its lowered tvm_access_ptr representation."""
    func = merge_if_atomic_write_test()
    assert_merge_if_unchanged(func)
    assert_merge_if_unchanged(func, lower_access_ptr=True)


def test_merge_if_atomic_read_condition():
    """The condition read remains visible after LowerAccessPtr."""
    func = merge_if_atomic_read_condition_test()
    assert_merge_if_unchanged(func)
    assert_merge_if_unchanged(func, lower_access_ptr=True)


def test_merge_if_safe_atomic_write():
    """Pointer-based writes to another buffer should not block merging."""
    func = merge_if_safe_atomic_write_test()
    for lower_access_ptr in (False, True):
        before_merge, after_merge = apply_merge_if(func, lower_access_ptr)
        assert count_if_then_else(before_merge.body) == 2
        assert count_if_then_else(after_merge.body) == 1
        merged_if = collect_nodes(after_merge.body, tvm.tirx.IfThenElse)[0]
        assert count_calls_to(merged_if.then_case, "tl.atomic_store_elem_op") == 1
        assert len(collect_nodes(merged_if.then_case, tvm.tirx.BufferStore)) == 1


def test_merge_if_address_of_write():
    """Opaque address_of calls conservatively block invalidating merges."""
    assert_merge_if_unchanged(merge_if_address_of_write_test())


if __name__ == "__main__":
    tilelang.testing.main()
