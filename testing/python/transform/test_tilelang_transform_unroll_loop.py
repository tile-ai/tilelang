import tilelang
import pytest
import tvm_tirx
import tilelang.language as T
import tvm_ffi
from tvm_tirx import ir


def post_order_visit(value, visit):
    tvm_ffi.structural_walk(value, (ir.Stmt, visit), order="post")


def test_unroll_decl_buffer_defs_are_fresh():
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A_flat = T.decl_buffer((16,), "float32", data=A.data)
                A_flat[0] = T.float32(i)

    after = tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))
    body = after["main"].body

    decls = [stmt for stmt in body.seq if isinstance(stmt, ir.Bind)]
    stores = [stmt for stmt in body.seq if isinstance(stmt, ir.TensorStore)]

    assert len(decls) == 2
    assert len(stores) == 2
    assert not decls[0].var.same_as(decls[1].var)
    assert tvm_ffi.structural_equal(decls[0].value.args[0], decls[1].value.args[0])
    assert stores[0].dest.same_as(decls[0].var)
    assert stores[1].dest.same_as(decls[1].var)


def test_unroll_explicit_loops_only():
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                A[j + 2] = j

    after = tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))
    body = after["main"].body

    assert isinstance(body, ir.SeqStmt)
    assert isinstance(body.seq[0], ir.TensorStore)
    assert isinstance(body.seq[1], ir.TensorStore)
    assert isinstance(body.seq[2], ir.For)
    assert body.seq[2].kind == ir.ForKind.UNROLLED


def test_unroll_explicit_loops_preserves_dynamic_and_factor_loops():
    class Before:
        @T.prim_func
        def main(n: T.int32, A: T.Buffer((16,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(n):
                A[j] = j
            for k in T.unroll(8, annotations={"pragma_unroll_factor": 4}):
                A[k + 8] = k

    after = tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, ir.For) else None,
    )

    assert [loop.var.name for loop in loops] == ["j", "k"]
    assert loops[0].kind == ir.ForKind.UNROLLED
    assert int(loops[1].annotations["pragma_unroll_factor"]) == 4


def test_explicit_unroll_rejects_break_targeting_loop():
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
                if A[i] != 0:
                    T.loop_break()

    with pytest.raises(ValueError, match="cannot be fully expanded"):
        tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))


def test_non_explicit_unroll_allows_break():
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2):
                A[i] = i
                if A[i] != 0:
                    T.loop_break()

    after = tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))
    assert isinstance(after["main"].body.seq[0], ir.For)
    assert after["main"].body.seq[0].kind == ir.ForKind.UNROLLED


def test_explicit_unroll_allows_break_targeting_nested_loop():
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                for j in T.serial(2):
                    A[i * 2 + j] = j
                    if A[i * 2 + j] != 0:
                        T.loop_break()

    after = tilelang.transform.UnrollLoop()(tvm_tirx.IRModule({"main": Before.main}))
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, ir.For) else None,
    )

    assert len(loops) == 2
    assert all(loop.kind == ir.ForKind.DEFAULT for loop in loops)


if __name__ == "__main__":
    test_unroll_decl_buffer_defs_are_fresh()
    test_unroll_explicit_loops_only()
    test_unroll_explicit_loops_preserves_dynamic_and_factor_loops()
