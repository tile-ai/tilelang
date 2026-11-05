"""Tensor-valued bindings stay definitions at language consumer boundaries."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


def _view(function, name):
    (bind,) = [node for node in collect_nodes(function, ir.Bind) if node.var.name == name and isinstance(node.var.ty, tirx.TensorType)]
    assert bind.value.op.name == "tirx.decl_tensor"
    return bind.var


@pytest.mark.parametrize("view", [False, True])
def test_copy_fill_clear_and_pointer_keep_bound_view_identity(view):
    @T.prim_func
    def kernel(A: T.Tensor((64,), "float32"), B: T.Tensor((8, 8), "float32")):
        alias = T.view(A, (8, 8)) if view else T.reshape(A, (8, 8))
        T.copy(alias, B)
        T.fill(alias, 2.0)
        T.clear(alias)
        T.evaluate(T.access_ptr(alias, "r"))

    assert tirx.analysis.check_well_formed(kernel)
    alias = _view(kernel, "alias")
    (copy,) = collect_calls(kernel, op="tl.tileop.copy")
    assert copy.args[0].source.same_as(alias)
    fills = collect_calls(kernel, op="tl.tileop.fill")
    assert len(fills) == 2 and all(call.args[0].source.same_as(alias) for call in fills)
    (pointer,) = collect_calls(kernel, op="tl.access_ptr")
    assert pointer.args[0].source.same_as(alias)
    for call in [copy, *fills, pointer]:
        call.validate()


@pytest.mark.parametrize("lanes", [2, 4])
def test_vector_atomic_return_uses_view_dtype_and_pointer(lanes):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        alias = T.reshape(A, (2, 4))
        other = T.reshape(B, (2, 4))
        if lanes == 2:
            T.evaluate(T.atomic_addx2(alias, other, return_prev=True))
        else:
            T.evaluate(T.atomic_addx4(alias, other, return_prev=True))

    assert tirx.analysis.check_well_formed(kernel)
    (call,) = collect_calls(kernel, op=f"tl.atomic_addx{lanes}_ret_elem_op")
    assert call.ty == ir.PrimType(f"float32x{lanes}")
    for pointer, name in zip(call.args, ("alias", "other"), strict=True):
        assert pointer.op.name == "tl.access_ptr"
        assert pointer.args[0].source.same_as(_view(kernel, name))
    call.validate()


def test_dense_gemm_keeps_three_bound_tensor_operands():
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16"), C: T.Tensor((256,), "float32")):
        a_view = T.reshape(A, (16, 16))
        b_view = T.reshape(B, (16, 16))
        c_view = T.reshape(C, (16, 16))
        T.gemm(a_view, b_view, c_view)

    assert tirx.analysis.check_well_formed(kernel)
    (call,) = collect_calls(kernel, op="tl.tileop.gemm")
    for region, name in zip(call.args[:3], ("a_view", "b_view", "c_view"), strict=True):
        assert region.source.same_as(_view(kernel, name))
    call.validate()


def test_sparse_gemm_keeps_metadata_and_data_view_definitions():
    @T.prim_func
    def kernel(
        A: T.Tensor((256,), "float16"),
        E: T.Tensor((32,), "uint16"),
        B: T.Tensor((512,), "float16"),
        C: T.Tensor((256,), "float32"),
    ):
        a_view = T.reshape(A, (16, 16))
        e_view = T.reshape(E, (16, 2))
        b_view = T.reshape(B, (32, 16))
        c_view = T.reshape(C, (16, 16))
        T.gemm_sp(a_view, e_view, b_view, c_view)

    assert tirx.analysis.check_well_formed(kernel)
    (call,) = collect_calls(kernel, op="tl.tileop.gemm_sp")
    for region, name in zip(call.args[:4], ("a_view", "e_view", "b_view", "c_view"), strict=True):
        assert region.source.same_as(_view(kernel, name))
    call.validate()


def test_scalar_load_alias_still_resolves_to_its_original_slice():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "float32")):
        value = A[2]
        T.fill(value, 3.0)
        T.evaluate(T.access_ptr(value, "rw"))

    assert tirx.analysis.check_well_formed(kernel)
    fills = collect_calls(kernel, op="tl.tileop.fill")
    assert len(fills) == 1
    for call in fills:
        assert call.args[0].source.same_as(kernel.params[0])
        assert int(call.args[0].region[0].begin) == 2
        assert int(call.args[0].region[0].extent) == 1
    (pointer,) = collect_calls(kernel, op="tl.access_ptr")
    assert tvm_ffi.structural_equal(pointer.args[0], ir.TensorLoad(kernel.params[0], [2]))


def test_clear_slice_alias_keeps_extent_and_bare_scalar_refusal():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "float32")):
        region = A[2:4]
        T.clear(region)

    assert tirx.analysis.check_well_formed(kernel)
    (fill,) = collect_calls(kernel, op="tl.tileop.fill")
    assert fill.args[0].source.same_as(kernel.params[0])
    assert int(fill.args[0].region[0].begin) == 2
    assert int(fill.args[0].region[0].extent) == 2
    with pytest.raises(ValueError, match="Invalid buffer region"):

        @T.prim_func
        def invalid(A: T.Tensor((8,), "float32")):
            value = A[2]
            T.clear(value)
