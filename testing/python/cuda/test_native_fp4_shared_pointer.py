"""Unpacked FP4 pointers keep shared bytes, offsets, and native type identity."""

import ctypes

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, prim, target, tirx

import tilelang
import tilelang.testing


DTYPE = "custom[float4_e2m1_unpacked]8"
CUDA = target.Target({"kind": "cuda", "arch": "sm_100a"})


def make_function(offset, bound, roundtrip, scope="shared"):
    storage_type = tirx.TensorType((64,), DTYPE, scope=scope)
    storage = ir.Var("storage", storage_type)
    raw_type = tirx.TensorType((64,), "uint8", scope=scope)
    raw = ir.Var("bytes", raw_type)
    output_ptr = ir.Var("output_ptr", ir.PtrType("uint8", "global"))
    output_type = tirx.TensorType((4,), "uint8", data_alignment=1)
    output = ir.Var("output", output_type)
    statements = [
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[output_type])),
        ir.Bind(storage, ir.Call("tirx.alloc_tensor", [], ty_args=[storage_type])),
        ir.Bind(raw, ir.Call("tirx.decl_tensor", [tirx.reinterpret(storage.data, ty=ir.PtrType("uint8", scope))], ty_args=[raw_type])),
    ]
    i = ir.Var("i", "int32")
    statements.append(ir.For(i, 0, 64, 1, ir.ForKind.DEFAULT, [ir.TensorStore(raw, [i], prim.Cast("uint8", i * 13 + 5))]))
    pointer = tirx.ptr_byte_offset(storage.data, ir.const(offset, "int64"), ty=storage.data.ty)
    if roundtrip:
        pointer = tirx.reinterpret(tirx.reinterpret(pointer, ty="uint64"), ty=storage.data.ty)
    if bound:
        variable = ir.Var("offset_pointer", pointer.ty)
        statements.append(ir.Bind(variable, pointer))
        pointer = variable
    # The custom view must remain a native custom view. Only the CUDA storage
    # representation becomes uint8_t*, including this separate declaration.
    view_type = tirx.TensorType((4,), DTYPE, scope=scope, data_alignment=1)
    view = ir.Var("tagged_view", view_type)
    statements.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view_type])))
    bytes_type = tirx.TensorType((4,), "uint8", scope=scope, data_alignment=1)
    bytes_view = ir.Var("byte_view", bytes_type)
    statements.append(
        ir.Bind(bytes_view, ir.Call("tirx.decl_tensor", [tirx.reinterpret(view.data, ty=ir.PtrType("uint8", scope))], ty_args=[bytes_type]))
    )
    j = ir.Var("j", "int32")
    statements.append(ir.For(j, 0, 4, 1, ir.ForKind.DEFAULT, [ir.TensorStore(output, [j], ir.TensorLoad(bytes_view, [j]))]))
    function = tirx.Function([output_ptr], statements, ret_type=ir.PrimType("void"))
    return function.with_attrs({"global_symbol": "shared_pointer", "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH})


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("bound,roundtrip", [(False, False), (True, False), (False, True), (True, True)])
def test_unpacked_pointer_emission_keeps_ir_and_shared_byte_offsets(scope, bound, roundtrip, tmp_path):
    function = make_function(7, bound, roundtrip, scope)
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"shared_pointer": function}), CUDA)
    source = module.inspect_source("cuda")
    (tmp_path / "kernel.cu").write_text(source)
    (tmp_path / "input.json").write_text(before)
    assert ir.save_json(function) == before
    assert "uint8_t storage[" in source
    assert "fp4_e2_" not in source
    assert "uint8_t*" in source
    assert " + (int64_t)7" in source or " + ((int64_t)7)" in source
    assert " / 2" not in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@tilelang.testing.requires_cuda_compute_version_lt(11, 0)
@pytest.mark.parametrize("offset", [0, 1, 7, 16])
@pytest.mark.parametrize("bound,roundtrip", [(False, False), (True, False), (False, True), (True, True)])
def test_unpacked_pointer_exact_bytes_on_cuda(offset, bound, roundtrip, tmp_path):
    from tilelang.backend.module import get_backend

    get_backend("cuda")
    torch.cuda.set_device(0)
    function = make_function(offset, bound, roundtrip)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda")(ir.IRModule({"shared_pointer": function}), CUDA)
    (tmp_path / "kernel.cu").write_text(module.inspect_source("cuda"))
    storage = torch.full((10,), 0xA5, dtype=torch.uint8, device="cuda")
    module["shared_pointer"](ctypes.c_void_p(storage.data_ptr() + 3))
    torch.cuda.synchronize()
    expected = torch.full_like(storage, 0xA5)
    expected[3:7] = (torch.arange(offset, offset + 4, device="cuda") * 13 + 5).to(torch.uint8)
    torch.testing.assert_close(storage, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scope", ["global", "local"])
def test_unpacked_pointer_does_not_gain_a_non_shared_representation(scope):
    value = ir.Var("source", ir.PtrType("uint8", scope))
    custom_type = ir.PtrType(DTYPE, scope)
    view = ir.Var("view", custom_type)
    function = tirx.Function([value], [ir.Bind(view, tirx.reinterpret(value, ty=custom_type))]).with_attrs(
        {"global_symbol": "unsupported", "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH}
    )
    with pytest.raises(RuntimeError, match="SMEM/TMA storage tag"):
        tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"unsupported": function}), CUDA)


@pytest.mark.parametrize("dtype", [DTYPE, DTYPE + "x2"])
def test_unpacked_scalar_and_vector_are_still_not_arithmetic_types(dtype):
    value = ir.Var("value", ir.PrimType(dtype))
    function = tirx.Function([value], []).with_attrs({"global_symbol": "unsupported", "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH})
    with pytest.raises(RuntimeError, match="SMEM/TMA storage tag"):
        tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"unsupported": function}), CUDA)
