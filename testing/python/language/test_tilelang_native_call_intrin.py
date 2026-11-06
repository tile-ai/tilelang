"""Public dtype-first factories and source shape comparisons on native IR."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.language.tir._constructors import BufferRegion
from tilelang.utils.language import _strip_source_call_annotations, is_full_region, prim_expr_equal


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64", "handle"])
@pytest.mark.parametrize("op_name", ["tirx.call_extern", "tirx.call_pure_extern"])
def test_dtype_first_external_intrinsics_have_concrete_return_type(dtype, op_name):
    argument = T.int32(3)
    call = T.call_intrin(dtype, op_name, "identity", argument)
    expected = ir.PtrType(ir.PrimType("void")) if dtype == "handle" else ir.PrimType(dtype)
    assert call.ty == expected
    assert not isinstance(call.ty, ir.MissingType)
    assert call.args[1].same_as(argument)
    call.validate()


def test_registered_intrinsic_return_contract_stays_authoritative():
    call = T.call_intrin("int32", ir.Op.get("tirx.gpu_storage_sync"), "shared")
    assert call.ty == ir.PrimType("void")
    call.validate()


def annotated(argument, value, dtype="int32"):
    return T.call_intrin(dtype, "tirx.call_pure_extern", "extent_identity", argument, annotations={"source_tag": value})


@pytest.mark.parametrize("nested", [False, True])
def test_source_annotations_do_not_change_weak_shape_comparison(nested):
    n = T.int32()
    left, right = annotated(n, 1), annotated(n, 2)
    if nested:
        left, right = annotated(left + 1, 3), annotated(right + 1, 4)
    before = [ir.save_json(left), ir.save_json(right)]
    assert not tvm_ffi.structural_equal(left, right)
    assert not ir.prim.expr_deep_equal(left, right)
    assert prim_expr_equal(left, right)
    assert [ir.save_json(left), ir.save_json(right)] == before
    assert not tvm_ffi.structural_equal(left, right)


@pytest.mark.parametrize("difference", ["argument", "dtype", "variable_identity", "native_attributes", "operator"])
def test_source_comparison_keeps_semantic_differences(difference):
    n = T.int32()
    left = annotated(n, 1)
    right = annotated(n, 2)
    if difference == "argument":
        right = annotated(n + 1, 2)
    elif difference == "dtype":
        right = annotated(n, 2, "int64")
    elif difference == "variable_identity":
        right = annotated(T.int32(), 2)
    elif difference == "native_attributes":
        left = ir.Call(left.op, left.args, attrs={**dict(left.attrs), "native_semantics": 1}, ty_args=left.ty_args, ty=left.ty)
        right = ir.Call(right.op, right.args, attrs={**dict(right.attrs), "native_semantics": 2}, ty_args=right.ty_args, ty=right.ty)
    else:
        right = ir.Call("tirx.call_extern", right.args, attrs=right.attrs, ty_args=right.ty_args, ty=right.ty)
    assert not prim_expr_equal(left, right)


def test_unencoded_native_attributes_are_not_source_annotations():
    n = T.int32()
    left = ir.Call("tirx.call_pure_extern", [ir.StringImm("extent_identity"), n], attrs={"source_tag": 1}, ty=ir.PrimType("int32"))
    right = ir.Call("tirx.call_pure_extern", [ir.StringImm("extent_identity"), n], attrs={"source_tag": 2}, ty=ir.PrimType("int32"))
    assert not prim_expr_equal(left, right)


@pytest.mark.parametrize("malformation", ["schema", "count", "payload", "version"])
def test_malformed_annotation_codec_is_not_discarded(malformation):
    # Static annotations now live directly in attrs. Exercise the remaining
    # pre-lowering codec with a real variable use in the source metadata.
    call = annotated(T.int32(), T.int32())
    attrs, operands = dict(call.attrs), list(call.args)
    if malformation == "schema":
        attrs["annotation_schema"] = ["unknown"]
    elif malformation == "count":
        attrs["operand_count"] = len(operands)
    elif malformation == "payload":
        operands[-1] = T.int32(0)
    else:
        attrs["schema_version"] = 2
    malformed = ir.Call(call.op, operands, attrs=attrs, ty_args=call.ty_args, ty=call.ty)
    assert _strip_source_call_annotations(malformed).same_as(malformed)


def test_annotated_shape_calls_reach_real_cumsum_frontend():
    @T.prim_func
    def kernel(n: T.int32):
        A = T.alloc_buffer(
            (T.call_intrin("int32", "tirx.call_pure_extern", "extent_identity", n, annotations={"source_tag": 1}),), "int32", "local"
        )
        B = T.alloc_buffer(
            (T.call_intrin("int32", "tirx.call_pure_extern", "extent_identity", n, annotations={"source_tag": 2}),), "int32", "local"
        )
        T.cumsum(A, B)

    assert tirx.analysis.check_well_formed(kernel)
    calls = []
    tvm_ffi.structural_walk(kernel, (ir.Call, lambda call: calls.append(call)))
    assert sum(isinstance(call.op, ir.Op) and call.op.name == "tl.tileop.cumsum" for call in calls) == 1


def test_full_region_uses_the_same_source_annotation_comparison():
    n = T.int32()
    extent = annotated(n, 1)
    other = annotated(n, 2)
    tensor = ir.Var("A", tirx.TensorType((extent,), "int32"))
    region = BufferRegion(tensor, [ir.Range.from_min_extent(0, other)])
    assert is_full_region(region)
    assert not is_full_region(BufferRegion(tensor, [ir.Range.from_min_extent(1, other)]))
    assert not is_full_region(BufferRegion(tensor, [ir.Range.from_min_extent(0, other + 1)]))
