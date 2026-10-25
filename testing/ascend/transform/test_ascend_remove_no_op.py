import tilelang
from tvm_tirx import ir
from tilelang import tvm
from tilelang.ascend import transform
from tilelang._compat import tirx
import tilelang.language as T


def test_ascend_remove_no_op_preserves_handle_store():
    value = tirx.Var("value", "handle")
    buffer = tirx.decl_buffer((1,), "uint64", scope="local.var")
    store = tirx.BufferStore(buffer, tirx.reinterpret("uint64", value), [0])
    mod = tvm.IRModule.from_expr(tirx.PrimFunc([buffer, value], store, ret_type=ir.TupleType([])))

    after = transform.AscendRemoveNoOp()(mod)["main"]

    assert len(after.body.seq) == 1
    tvm.ir.assert_structural_equal(after.body.seq[0], store)


def test_ascend_remove_no_op_removes_numeric_self_store():
    @T.prim_func
    def before(a: T.Buffer((1,), "float32")):
        a[0] = a[0]

    mod = tvm.IRModule({"main": before})
    after = transform.AscendRemoveNoOp()(mod)["main"]

    assert isinstance(after.body, ir.SeqStmt)
    assert not after.body.seq


if __name__ == "__main__":
    tilelang.testing.main()
