"""Explicit binding retains one evaluation independently of assignment aliases."""

import inspect

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.ir as tl_ir
import tilelang.language as T


def _bindings(function):
    result = []
    tvm_ffi.structural_walk(function, (ir.Bind, result.append), order="pre")
    return result


def _sample(observations):
    def once(value):
        observations.append(value)
        return value

    @T.prim_func
    def sample(A: T.Tensor((1,), "int32"), B: T.Tensor((4,), "int32"), n: T.int32):
        scalar = T.bind(once(n))
        constant = T.bind(once(T.int32(7)))
        snapshot = T.bind(once(A[0]))
        alias = n
        A[0] = 99
        B[0] = scalar
        B[1] = constant
        B[2] = snapshot
        B[3] = snapshot + alias

    return sample


def test_explicit_bind_materializes_var_constant_and_load_once():
    observations = []
    function = _sample(observations)
    assert tirx.analysis.check_well_formed(function)
    assert len(observations) == 3
    binds = _bindings(function)
    assert len(binds) == 3
    for binding, value in zip(binds, observations):
        assert binding.value.same_as(value)
        assert not binding.var.same_as(value)
        ir.assert_structural_equal(binding.var.ty, value.ty)
    assert isinstance(binds[0].value, ir.Var)
    assert isinstance(binds[1].value, ir.IntImm)
    assert isinstance(binds[2].value, ir.TensorLoad)
    assert binds[0].value.same_as(function.params[2])


def test_explicit_bind_has_callsite_location_and_preserves_rhs_identity():
    observations = []
    function = _sample(observations)
    source, first_line = inspect.getsourcelines(_sample)
    lines = [first_line + index for index, line in enumerate(source) if "= T.bind(" in line]
    binds = _bindings(function)
    assert len(binds) == len(lines) == 3
    for binding, line in zip(binds, lines):
        assert tl_ir.span_to_location(binding.loc) == (__file__, line)
        assert tl_ir.span_to_location(binding.var.loc) == (__file__, line)
    # An explicit alias declaration owns a fresh Var. It must not rename or
    # relocate the original scalar parameter to the binding's source line.
    assert binds[0].value.same_as(function.params[2])
    assert tl_ir.span_to_location(function.params[2].loc) != (__file__, lines[0])


def test_bound_load_keeps_its_value_across_a_write_and_repeated_uses():
    function = _sample([])
    target = {"kind": "c"}
    kernel = tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")
    source = torch.tensor([17], dtype=torch.int32)
    output = torch.zeros(4, dtype=torch.int32)
    kernel(source, output, 13)
    np.testing.assert_array_equal(source.numpy(), [99])
    np.testing.assert_array_equal(output.numpy(), [13, 7, 17, 30])


def test_type_annotation_forces_a_fresh_typed_binding():
    @T.prim_func
    def sample(n: T.int32, B: T.Tensor((1,), "int32")):
        value = T.bind(n, type_annotation=T.int32)
        B[0] = value

    assert tirx.analysis.check_well_formed(sample)
    (binding,) = _bindings(sample)
    assert binding.value.same_as(sample.params[0])
    assert not binding.var.same_as(sample.params[0])
    ir.assert_structural_equal(binding.var.ty, ir.PrimType("int32"))


def test_explicit_bind_accepts_an_existing_native_variable():
    requested = ir.Var("requested", ir.PrimType("int32"))

    @T.prim_func
    def sample(n: T.int32, B: T.Tensor((1,), "int32")):
        value = T.bind(n, var=requested)
        B[0] = value

    assert tirx.analysis.check_well_formed(sample)
    (binding,) = _bindings(sample)
    assert binding.var.same_as(requested)
    assert binding.value.same_as(sample.params[0])


def test_explicit_bind_rejects_a_non_variable_destination():
    with pytest.raises(TypeError, match="var.*ir.Var"):

        @T.prim_func
        def sample(B: T.Tensor((1,), "int32")):
            value = T.bind(7, var=8)
            B[0] = value
