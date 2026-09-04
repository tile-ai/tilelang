import tilelang
import pytest
from tilelang import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.tirx.stmt_functor import post_order_visit


def test_unroll_decl_buffer_defs_are_fresh():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A_flat = T.decl_buffer((16,), "float32", data=A.data)
                A_flat[0] = T.float32(i)

    after = tilelang.transform.UnrollLoop()(Before)
    body = after["main"].body

    decls = [stmt for stmt in body.seq if isinstance(stmt, tvm.tirx.DeclBuffer)]
    stores = [stmt for stmt in body.seq if isinstance(stmt, tvm.tirx.BufferStore)]

    assert len(decls) == 2
    assert len(stores) == 2
    assert not decls[0].buffer.same_as(decls[1].buffer)
    assert decls[0].buffer.data.same_as(decls[1].buffer.data)
    assert stores[0].buffer.same_as(decls[0].buffer)
    assert stores[1].buffer.same_as(decls[1].buffer)


def test_unroll_explicit_loops_only():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                A[j + 2] = j

    after = tilelang.transform.UnrollLoop()(Before)
    body = after["main"].body

    assert isinstance(body, tvm.tirx.SeqStmt)
    assert isinstance(body.seq[0], tvm.tirx.BufferStore)
    assert isinstance(body.seq[1], tvm.tirx.BufferStore)
    assert isinstance(body.seq[2], tvm.tirx.For)
    assert body.seq[2].kind == tvm.tirx.ForKind.UNROLLED


def test_unroll_explicit_loops_preserves_dynamic_and_factor_loops():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32, A: T.Buffer((16,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(n):
                A[j] = j
            for k in T.unroll(8, annotations={"pragma_unroll_factor": 4}):
                A[k + 8] = k

    after = tilelang.transform.UnrollLoop()(Before)
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None,
    )

    assert [loop.loop_var.name for loop in loops] == ["j", "k"]
    assert loops[0].kind == tvm.tirx.ForKind.UNROLLED
    assert int(loops[1].annotations["pragma_unroll_factor"]) == 4


def test_explicit_unroll_rejects_break_targeting_loop():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
                if A[i] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    with pytest.raises(ValueError, match="cannot be fully expanded"):
        tilelang.transform.UnrollLoop()(Before)


def test_non_explicit_unroll_allows_break():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2):
                A[i] = i
                if A[i] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    after = tilelang.transform.UnrollLoop()(Before)
    assert isinstance(after["main"].body, tvm.tirx.For)
    assert after["main"].body.kind == tvm.tirx.ForKind.UNROLLED


def test_explicit_unroll_allows_break_targeting_nested_loop():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                for j in T.serial(2):
                    A[i * 2 + j] = j
                    if A[i * 2 + j] != 0:
                        T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))

    after = tilelang.transform.UnrollLoop()(Before)
    loops = []
    post_order_visit(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None,
    )

    assert len(loops) == 2
    assert all(loop.kind == tvm.tirx.ForKind.SERIAL for loop in loops)


if __name__ == "__main__":
    test_unroll_decl_buffer_defs_are_fresh()
    test_unroll_explicit_loops_only()
    test_unroll_explicit_loops_preserves_dynamic_and_factor_loops()
