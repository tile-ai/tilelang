import tilelang
import tilelang.testing
import numpy as np
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def _collect_tvm_access_ptr_offsets(func: tirx.Function):
    offsets = []

    def _visit(node):
        if isinstance(node.op, ir.Op) and node.op.name == "tl.ptr_access":
            offsets.append(node.args[1])

    tvm_ffi.structural_walk(func.body, (ir.Call, _visit), order="post")
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
    assert before_offsets[0].ty == ir.PrimType("int32")

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert after_offsets[0].ty == ir.PrimType("int64")


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
    assert before_offsets[0].ty == ir.PrimType("int32")

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert after_offsets[0].ty == ir.PrimType("int32")


def test_flatten_buffer_executes_large_element_offsets_without_narrow_overflow():
    # Inspect addresses, without dereferencing the intentionally distant
    # element. Two loop iterations suffice to cover values beyond int32.
    source = ir.Var("source", tirx.TensorType([1], "float16", data_alignment=1))
    output = ir.Var("output", tirx.TensorType([2], "uint64", data_alignment=1))
    index = ir.Var("index", "int32")
    start = (1 << 30) - 2
    pointer = ir.Call("tl.ptr_access", [source.data, index * 4, ir.IntImm("int32", 1), ir.IntImm("int32", 1)])
    byte_distance = tirx.reinterpret(pointer, ty="uint64") - tirx.reinterpret(source.data, ty="uint64")
    function = tirx.Function(
        [source, output],
        [ir.For(index, start, start + 2, 1, ir.ForKind.DEFAULT, [ir.TensorStore(output, [index - start], byte_distance)])],
    ).with_attr("global_symbol", "main")
    module = tilelang.transform.FlattenBuffer()(tvm.IRModule({"main": function}))
    module = tilelang.transform.LowerAccessPtr()(module)
    assert tirx.analysis.check_well_formed(module)
    executable = compile_packed_function(module, c_codegen="compute")
    input_array = np.zeros(1, dtype="float16")
    actual = np.zeros(2, dtype="uint64")
    executable(tvm_ffi.from_dlpack(input_array), tvm_ffi.from_dlpack(actual))
    expected = np.asarray([start * 4 * 2, (start + 1) * 4 * 2], dtype="uint64")
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    tilelang.testing.main()
