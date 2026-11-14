import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from tvm.script import tirx as T


def test_ascend_remove_no_op_preserves_handle_store():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("handle"), "local.var")
    data = tirx.Var("pointer", pointer_type)
    value = tirx.Var("value", "handle")
    buffer = tirx.decl_buffer((1,), "handle", data=data, scope="local.var")
    store = tirx.BufferStore(buffer, value, [0])
    mod = tvm.IRModule.from_expr(tirx.PrimFunc([data, value], store))

    after = transform.AscendRemoveNoOp()(mod)["main"]

    assert isinstance(after.body, tirx.BufferStore)
    tvm.ir.assert_structural_equal(after.body, store)


def test_ascend_remove_no_op_removes_numeric_self_store():
    @T.prim_func(private=True)
    def before(a: T.Buffer((1,), "float32")):
        a[0] = a[0]

    mod = tvm.IRModule.from_expr(before)
    after = transform.AscendRemoveNoOp()(mod)["main"]

    assert isinstance(after.body, tirx.Evaluate)


if __name__ == "__main__":
    tilelang.testing.main()
