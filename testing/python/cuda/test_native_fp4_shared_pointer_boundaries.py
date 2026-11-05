"""Independent pointer-scope, volatile-provenance, and packed-byte boundaries."""

import ctypes
import re

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, target, tirx

import tilelang
import tilelang.testing

DTYPE = "custom[float4_e2m1_unpacked]8"
CUDA = target.Target({"kind": "cuda", "arch": "sm_100a"})


def function(parameters, statements, name):
    result = tirx.Function(parameters, statements, ret_type=ir.PrimType("void"))
    return result.with_attrs({"global_symbol": name, "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH})


def emit(value, tmp_path, compile_device=False):
    assert tirx.analysis.check_well_formed(value)
    serialized = ir.save_json(value)
    suffix = "" if compile_device else "_without_compile"
    result = tvm_ffi.get_global_func("target.build.tilelang_cuda" + suffix)(ir.IRModule({str(value.attrs["global_symbol"]): value}), CUDA)
    source = result.inspect_source("cuda")
    assert ir.save_json(value) == serialized
    (tmp_path / "input.json").write_text(serialized)
    (tmp_path / "kernel.cu").write_text(source)
    return result, source


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("integer_bind", [False, True])
@pytest.mark.parametrize("qualifier", [False, True])
def test_custom_alias_keeps_volatile_provenance(scope, integer_bind, qualifier, tmp_path):
    storage_type = tirx.TensorType((64,), DTYPE, scope=scope)
    storage = ir.Var("storage", storage_type)
    raw_type = tirx.TensorType((64,), "uint8", scope=scope)
    raw = ir.Var("raw_bytes", raw_type)
    output_ptr = ir.Var("output_ptr", ir.PtrType("uint8", "global"))
    output_type = tirx.TensorType((1,), "uint8")
    output = ir.Var("output", output_type)
    statements = [
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[output_type])),
        ir.Bind(storage, ir.Call("tirx.alloc_tensor", [], ty_args=[storage_type], attrs=ir.DictAttrs({"tirx.volatile": qualifier}))),
        ir.Bind(raw, ir.Call("tirx.decl_tensor", [tirx.reinterpret(storage.data, ty=ir.PtrType("uint8", scope))], ty_args=[raw_type])),
        ir.TensorStore(raw, [ir.const(7, "int32")], ir.const(0x37, "uint8")),
    ]
    pointer = tirx.ptr_byte_offset(storage.data, ir.const(7, "int64"), ty=storage.data.ty)
    if integer_bind:
        encoded = ir.Var("encoded_address", "uint64")
        statements.append(ir.Bind(encoded, tirx.reinterpret(pointer, ty="uint64")))
        pointer = tirx.reinterpret(encoded, ty=storage.data.ty)
    bound = ir.Var("tagged_pointer", storage.data.ty)
    statements.append(ir.Bind(bound, pointer))
    view_type = tirx.TensorType((1,), DTYPE, scope=scope)
    view = ir.Var("custom_view", view_type)
    statements.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [bound], ty_args=[view_type])))
    byte_type = tirx.TensorType((1,), "uint8", scope=scope)
    byte_view = ir.Var("byte_view", byte_type)
    statements.append(
        ir.Bind(byte_view, ir.Call("tirx.decl_tensor", [tirx.reinterpret(view.data, ty=ir.PtrType("uint8", scope))], ty_args=[byte_type]))
    )
    statements.append(ir.TensorStore(output, [ir.const(0, "int32")], ir.TensorLoad(byte_view, [ir.const(0, "int32")])))
    value = function([output_ptr], statements, "volatile_custom")
    _, source = emit(value, tmp_path)
    output_line = next(line for line in source.splitlines() if re.search(r"output_ptr\[0\]\s*=", line))
    assert bool(re.search(r"\bvolatile\b", output_line)) == qualifier, output_line
    bound_line = next(line for line in source.splitlines() if "tagged_pointer =" in line)
    assert bool(re.search(r"\bvolatile\b", bound_line)) == qualifier, bound_line
    assert "uint8_t*" in bound_line
    assert " / 2" not in source


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_custom_vector_pointer_stays_rejected(scope):
    parameter = ir.Var("source", ir.PtrType("uint8", scope))
    pointer_type = ir.PtrType(DTYPE + "x2", scope)
    pointer = ir.Var("vector_pointer", pointer_type)
    value = function([parameter], [ir.Bind(pointer, tirx.reinterpret(parameter, ty=pointer_type))], "vector_custom")
    with pytest.raises(RuntimeError, match="SMEM/TMA storage tag"):
        tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"vector_custom": value}), CUDA)


def packed_byte_function(offset):
    source_ptr = ir.Var("packed_source", ir.PtrType("float4_e2m1fn", "global"))
    output_ptr = ir.Var("output_ptr", ir.PtrType("uint8", "global"))
    view_type = tirx.TensorType((3,), "uint8")
    view = ir.Var("source_bytes", view_type)
    output = ir.Var("output", view_type)
    pointer = tirx.ptr_byte_offset(source_ptr, ir.const(offset, "int64"), ty=source_ptr.ty)
    statements = [
        ir.Bind(view, ir.Call("tirx.decl_tensor", [tirx.reinterpret(pointer, ty=ir.PtrType("uint8", "global"))], ty_args=[view_type])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[view_type])),
    ]
    i = ir.Var("i", "int32")
    statements.append(ir.For(i, 0, 3, 1, ir.ForKind.DEFAULT, [ir.TensorStore(output, [i], ir.TensorLoad(view, [i]))]))
    return function([source_ptr, output_ptr], statements, "packed_bytes")


@pytest.mark.parametrize("offset", [0, 1, 7, 16])
def test_packed_global_pointer_displacement_counts_bytes(offset, tmp_path):
    _, source = emit(packed_byte_function(offset), tmp_path)
    assert "fp4_e2_t*" in source
    assert re.search(r"char\s*\*\)packed_source", source), source
    assert " / 2" not in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@tilelang.testing.requires_cuda_compute_version_lt(11, 0)
@pytest.mark.parametrize("offset", [0, 1, 7, 16])
def test_packed_global_pointer_exact_bytes_and_guards(offset, tmp_path):
    from tilelang.backend.module import get_backend

    get_backend("cuda")
    torch.cuda.set_device(0)
    module, _ = emit(packed_byte_function(offset), tmp_path, compile_device=True)
    source = (torch.arange(64, device="cuda") * 17 + 3).to(torch.uint8)
    original_source = source.clone()
    output = torch.full((8,), 0xA5, dtype=torch.uint8, device="cuda")
    module["packed_bytes"](ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(output.data_ptr() + 2))
    torch.cuda.synchronize()
    expected = torch.full_like(output, 0xA5)
    expected[2:5] = source[offset : offset + 3]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(source, original_source, atol=0, rtol=0)
