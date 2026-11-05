from tvm_tirx import ir
from testing.ascend._ir import nodes, tensor
import tilelang
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import tirx
import tilelang.language as T


def test_ascend_remove_no_op_preserves_handle_store():
    # Native tensors carry primitive elements. Preserve the externally visible
    # handle store through its explicit 64-bit pointer representation.
    pointer_type = ir.PtrType(ir.PrimType("uint64"), "local.var")
    data = ir.Var("pointer", pointer_type)
    value = ir.Var("value", ir.PtrType(ir.PrimType("void")))
    buffer = tensor((1,), "uint64", scope="local.var")
    declare = ir.Bind(buffer, ir.Call("tirx.decl_tensor", [data], ty_args=[buffer.ty]))
    store = ir.TensorStore(buffer, [0], tirx.reinterpret(value, ty="uint64"))
    mod = tvm_tirx.IRModule.from_expr(tirx.Function([data, value], [declare, store]))

    after = transform.AscendRemoveNoOp()(mod)["main"]

    (result,) = nodes(after.body, ir.TensorStore)
    tvm_tirx.ir.assert_structural_equal(result, store)


def test_ascend_remove_no_op_removes_numeric_self_store():
    @T.prim_func(private=True)
    def before(a: T.Buffer((1,), "float32")):
        a[0] = a[0]

    mod = tvm_tirx.IRModule.from_expr(before)
    after = transform.AscendRemoveNoOp()(mod)["main"]

    assert isinstance(after.body, ir.SeqStmt) and len(after.body) == 0


if __name__ == "__main__":
    tilelang.testing.main()
