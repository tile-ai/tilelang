import tilelang
import tilelang.testing
from tilelang import tvm
import tilelang.language as T


def _collect_tvm_access_ptr_offsets(func: tvm.tir.PrimFunc):
    offsets = []

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tir.tvm_access_ptr":
            offsets.append(node.args[2])

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return offsets


def test_flatten_buffer_promotes_tvm_access_ptr_offset_to_int64():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 30):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 4,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int64"


def test_flatten_buffer_keeps_safe_tvm_access_ptr_offset_int32():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 20):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 2,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int32"


if __name__ == "__main__":
    tilelang.testing.main()
