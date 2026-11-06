"""Native tensor and pointer contracts for the public accumulator fence."""

from tilelang.language import _construction
from tilelang.language import _tensor_spec

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.language as T


def _fence_call(function):
    assert tirx.analysis.check_well_formed(function)
    calls = []
    tvm_ffi.structural_walk(function, (ir.Call, calls.append))
    for call in calls:
        call.validate()
    (fence,) = [call for call in calls if call.op.name == "tl.warpgroup_fence_operand"]
    assert int(fence.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.Opaque)
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(function)), function)
    return fence


def _integer(expression):
    return int(sym.Analyzer().simplify(expression))


@pytest.mark.parametrize("dtype", ["float32", "int32", "float16", "int8"])
@pytest.mark.parametrize("kind", ["tensor", "load", "region"])
@pytest.mark.parametrize("strides, element_offset", [(None, 10), ((10, 1), 12)])
def test_tensor_paths_preserve_element_offsets_and_register_counts(dtype, kind, strides, element_offset):
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("fence_" + kind)
        tensor = _tensor_spec.arg("fragment", _tensor_spec.buffer((4, 8), dtype, strides=strides))
        if kind == "tensor":
            T.warpgroup_fence_operand(tensor, offset=3)
            expected_offset, element_count = 3, 32
        elif kind == "load":
            T.warpgroup_fence_operand(tensor[1, 2], offset=3, num_regs=7)
            expected_offset, element_count = element_offset + 3, None
        else:
            T.warpgroup_fence_operand(tensor[1:3, 2:6], offset=3)
            expected_offset, element_count = element_offset + 3, 8
    fence = _fence_call(builder.get())
    assert fence.args[0].value == dtype
    assert fence.args[1].op.name == "tirx.tensor_data_ptr"
    assert fence.args[1].args[0].same_as(tensor)
    assert fence.args[1].ty.element_type == ir.PrimType(dtype)
    assert _integer(fence.args[2]) == expected_offset
    scalar_dtype = tvm_ffi.dtype(dtype)
    assert scalar_dtype.lanes == 1
    registers = 7 if element_count is None else (element_count * scalar_dtype.bits + 31) // 32
    assert _integer(fence.args[3]) == registers


@pytest.mark.parametrize("dtype, registers", [("float16", 5), ("int8", 3)])
def test_inferred_count_rounds_partial_registers_up(dtype, registers):
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("partial_register")
        tensor = _tensor_spec.arg("fragment", _tensor_spec.buffer((3, 3), dtype))
        T.warpgroup_fence_operand(tensor)
    assert _integer(_fence_call(builder.get()).args[3]) == registers


@pytest.mark.parametrize("dtype", ["float32", "int32", "float16", "int8"])
@pytest.mark.parametrize("kind", ["typed_pointer", "access_ptr", "address_of"])
def test_pointer_paths_infer_native_element_type_and_keep_operand_identity(dtype, kind):
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("fence_" + kind)
        if kind == "typed_pointer":
            pointer = _tensor_spec.arg("fragment", ir.PtrType(dtype, "local"))
        else:
            tensor = _tensor_spec.arg("fragment", _tensor_spec.buffer((4, 8), dtype, strides=(10, 1), scope="local"))
            pointer = T.access_ptr(tensor[1, 2], "rw", extent=8) if kind == "access_ptr" else T.address_of(tensor[1, 2])
        before = ir.save_json(pointer)
        T.warpgroup_fence_operand(pointer, offset=3, num_regs=7)
    fence = _fence_call(builder.get())
    assert fence.args[0].value == dtype
    assert fence.args[1].same_as(pointer)
    assert ir.save_json(pointer) == before
    assert _integer(fence.args[2]) == 3
    assert _integer(fence.args[3]) == 7
    assert pointer.ty.element_type == ir.PrimType(dtype)
    assert str(pointer.ty.scope) == "local"


@pytest.mark.parametrize("kind", ["tensor", "load", "region"])
def test_tensor_dtype_mismatch_still_rejects_before_emission(kind):
    with IRBuilder(), _construction.prim_func():
        tensor = _tensor_spec.arg("fragment", _tensor_spec.buffer((4, 8), "float32"))
        operand = tensor if kind == "tensor" else tensor[1, 2] if kind == "load" else tensor[1:3, 2:6]
        with pytest.raises(ValueError, match="dtype mismatch"):
            T.warpgroup_fence_operand(operand, num_regs=1, dtype="int32")


@pytest.mark.parametrize("kind", ["load", "pointer", "symbolic_tensor", "symbolic_region"])
def test_ambiguous_register_counts_still_require_the_caller(kind):
    with IRBuilder(), _construction.prim_func():
        n = _tensor_spec.arg("n", ir.PrimType("int32"))
        tensor = _tensor_spec.arg("fragment", _tensor_spec.buffer((n, 8), "float32"))
        pointer = _tensor_spec.arg("pointer", ir.PtrType("float32"))
        operand = (
            tensor[0, 0] if kind == "load" else pointer if kind == "pointer" else tensor if kind == "symbolic_tensor" else tensor[0:n, 0:4]
        )
        with pytest.raises(ValueError, match="num_regs"):
            T.warpgroup_fence_operand(operand)


def test_void_pointer_requires_explicit_dtype_and_preserves_it_when_supplied():
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("untyped_pointer")
        pointer = _tensor_spec.arg("pointer", ir.PtrType("void"))
        with pytest.raises(ValueError, match="dtype must be provided"):
            T.warpgroup_fence_operand(pointer, num_regs=3)
        T.warpgroup_fence_operand(pointer, num_regs=3, dtype="int32")
    fence = _fence_call(builder.get())
    assert fence.args[1].same_as(pointer)
    assert fence.args[0].value == "int32"


@pytest.mark.parametrize(
    "dtype, ctype, cast",
    [
        ("float32", "float", "float"),
        ("int32", "int", "uint32_t"),
        ("float16", "half_t", "uint32_t"),
        ("int8", "signed char", "uint32_t"),
    ],
)
def test_typed_pointer_reaches_real_cuda_source_emitter(dtype, ctype, cast, tmp_path):
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("native_fence")
        pointer = _tensor_spec.arg("fragment", ir.PtrType(dtype))
        T.warpgroup_fence_operand(pointer, offset=3, num_regs=7)
    function = builder.get()
    _fence_call(function)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"})
    function = function.with_attr("calling_conv", 2).with_attr("target", target)
    source = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        ir.IRModule({"native_fence": function}), target
    ).inspect_source("cuda")
    (tmp_path / (dtype + ".cu")).write_text(source)
    assert f"{ctype}*" in source or f"{ctype} *" in source
    assert f"reinterpret_cast<{cast}*>(fragment + 3), 7)" in source
    assert source.count("tl::warpgroup_fence_operand(") == 1
