"""Native parameter expressions must not be invoked as annotation factories."""

import numpy as np
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T


def test_native_pointer_annotation_retains_element_type_and_scope():
    @T.prim_func
    def sample(pointer: T.handle("uint8x128", "global")):
        T.evaluate(0)

    assert tirx.analysis.check_well_formed(sample)
    (pointer,) = sample.params
    assert pointer.name == "pointer"
    assert pointer.ty == ir.PtrType(ir.PrimType("uint8x128"), "global")


def test_native_scalar_annotation_retains_parameter_identity():
    @T.prim_func
    def sample(value: T.int32()):
        T.evaluate(value)

    assert tirx.analysis.check_well_formed(sample)
    (value,) = sample.params
    assert value.name == "value"
    assert value.ty == ir.PrimType("int32")
    assert sample.body[0].value.same_as(value)


def test_callable_handle_factory_remains_supported():
    @T.prim_func
    def sample(pointer: T.handle):
        T.evaluate(0)

    assert tirx.analysis.check_well_formed(sample)
    (pointer,) = sample.params
    assert pointer.ty == ir.PtrType(ir.PrimType("void"), "global")


def test_user_annotation_factory_is_evaluated_once():
    calls = []

    def pointer_factory():
        calls.append("annotation")
        return T.handle("float32", "global")

    @T.prim_func
    def sample(pointer: pointer_factory):
        T.evaluate(0)

    assert calls == ["annotation"]
    assert tirx.analysis.check_well_formed(sample)
    assert sample.params[0].ty == ir.PtrType(ir.PrimType("float32"), "global")


def test_dtype_factory_keeps_a_scalar_parameter():
    @T.prim_func
    def sample(value: T.int64):
        T.evaluate(value)

    assert tirx.analysis.check_well_formed(sample)
    assert sample.params[0].ty == ir.PrimType("int64")


def test_native_scalar_annotation_executes_with_explicit_runtime_value():
    @T.prim_func
    def sample(output: T.Tensor((4,), "int32"), value: T.int32()):
        for i in T.serial(4):
            output[i] = value * 3 + i

    assert tirx.analysis.check_well_formed(sample)
    target = "c"
    kernel = tilelang.compile(sample, target=target, target_host=target, execution_backend="tvm_ffi")
    output = torch.empty(4, dtype=torch.int32)
    kernel(output, -7)
    np.testing.assert_array_equal(output.numpy(), [-21, -20, -19, -18])
