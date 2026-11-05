"""Public argument metadata survives native IR construction and cache reload."""

import json

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params


def test_rank_zero_tensor_is_distinct_from_scalar_and_pointer():
    parameters = [
        ir.Var("value", "float32"),
        ir.Var("scalar_tensor", tirx.TensorType([], "float32", layout="default")),
        ir.Var("pointer", ir.PtrType("float32", "global")),
    ]
    metadata = [KernelParam.from_var(value) for value in parameters]
    assert [value.kind for value in metadata] == ["scalar", "tensor", "pointer"]
    assert [value.is_scalar() for value in metadata] == [True, False, True]
    assert [value.shape for value in metadata] == [[], [], []]
    restored = load_kernel_params(dump_kernel_params(metadata))
    assert [value.kind for value in restored] == ["scalar", "tensor", "pointer"]
    assert not restored[1].is_scalar()
    assert str(restored[2].dtype) == "handle"


def test_shape_graph_keeps_shared_and_distinct_symbol_identities():
    n = ir.Var("n", "int64")
    another_n = ir.Var("n", "int64")
    tensors = [
        ir.Var("A", tirx.TensorType([n, 64], "float16", layout="default")),
        ir.Var("B", tirx.TensorType([n * 2, 8], "int8", layout="default")),
        ir.Var("C", tirx.TensorType([another_n], "float32", layout="default")),
    ]
    metadata = [KernelParam.from_var(value) for value in tensors]
    assert metadata[0].shape[0].same_as(n)
    text = dump_kernel_params(metadata)
    json.loads(text)
    restored = load_kernel_params(text)
    restored_n = restored[0].shape[0]
    assert restored[1].shape[0].a.same_as(restored_n)
    assert not restored[2].shape[0].same_as(restored_n)
    assert isinstance(restored[0].shape[1], int) and restored[0].shape[1] == 64
    assert tvm_ffi.structural_equal(restored[1].shape[0], restored_n * 2)


@pytest.mark.parametrize(
    "dtype,kind", [("uint32", "unsigned"), ("bool", "boolean"), ("float8_e4m3", "float8"), ("float4_e2m1fn", "float4")]
)
def test_dtype_classification_uses_native_codes(dtype, kind):
    param = KernelParam.from_var(ir.Var("A", tirx.TensorType([4], dtype, layout="default")))
    assert getattr(param, f"is_{kind}")()
    restored = load_kernel_params(dump_kernel_params([param]))[0]
    assert getattr(restored, f"is_{kind}")()
    assert (restored.dtype.type_code, restored.dtype.bits, restored.dtype.lanes) == (
        param.dtype.type_code,
        param.dtype.bits,
        param.dtype.lanes,
    )


@pytest.mark.parametrize("dtype", ["custom[tfloat32]32", "custom[float4_e2m1_unpacked]8x2"])
def test_custom_storage_dtype_is_preserved_in_parameter_cache(dtype):
    expected = tvm_ffi.dtype(dtype)
    param = KernelParam.from_var(ir.Var("A", tirx.TensorType([4], expected, layout="default")))
    restored = load_kernel_params(dump_kernel_params([param]))[0]
    assert (restored.dtype.type_code, restored.dtype.bits, restored.dtype.lanes) == (
        expected.type_code,
        expected.bits,
        expected.lanes,
    )


def test_old_parameter_cache_generation_is_rejected():
    old_payload = ir.save_json(tvm_ffi.convert({"schema_version": 1, "params": []}))
    with pytest.raises(ValueError, match="cache schema"):
        load_kernel_params(old_payload)
    with pytest.raises(ValueError):
        load_kernel_params("\x80\x04not a json document")


def test_invalid_parameter_kind_and_type_are_rejected():
    with pytest.raises(ValueError, match="Only tensor"):
        KernelParam("int32", [4], "scalar")
    with pytest.raises(ValueError, match="Unknown kernel parameter kind"):
        KernelParam("float32", [], "invalid")
    with pytest.raises(TypeError, match="Unsupported kernel parameter type"):
        KernelParam.from_var(ir.Var("tuple", ir.TupleType([ir.PrimType("float32")])))
