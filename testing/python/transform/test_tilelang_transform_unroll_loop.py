import tilelang
from tilelang import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


def test_unroll_decl_buffer_defs_are_fresh():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for i in T.unroll(2):
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


if __name__ == "__main__":
    test_unroll_decl_buffer_defs_are_fresh()
