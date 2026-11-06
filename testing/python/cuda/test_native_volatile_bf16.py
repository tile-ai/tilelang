"""Volatile BF16 memory accesses preserve payloads through views and vectors."""

import ctypes
import re

import pytest
import tilelang.testing
import torch
import tvm_ffi
from testing.python.cuda.test_native_vector_memory_alignment import _nvcc

from tvm_tirx import ir, prim, target, tirx


def _copy_function(lanes, stride, alias, volatile):
    dtype = ir.PrimType("bfloat16")
    source_ptr = ir.Var("source_ptr", ir.PtrType(dtype))
    output_ptr = ir.Var("output_ptr", ir.PtrType(dtype))
    base = ir.Var("base", "int32")
    global_ty = tirx.TensorType((64,), dtype, data_alignment=2)
    source, output = ir.Var("source", global_ty), ir.Var("output", global_ty)
    shared_ty = tirx.TensorType((64,), dtype, scope="shared", data_alignment=2)
    view = ir.Var("view", shared_ty)
    attributes = ir.DictAttrs({"tirx.volatile": volatile})
    statements = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_ptr], ty_args=[global_ty])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[global_ty])),
    ]
    if alias:
        owner_ty = tirx.TensorType((256,), "uint8", scope="shared", data_alignment=64)
        owner = ir.Var("arena", owner_ty)
        pointer = ir.Var("view_pointer", ir.PtrType(dtype, "shared"))
        statements.extend(
            [
                ir.Bind(owner, ir.Call("tirx.alloc_tensor", [], ty_args=[owner_ty], attrs=attributes)),
                ir.Bind(pointer, tirx.ptr_byte_offset(tirx.tensor_data_ptr(owner), 2, ty=pointer.ty)),
                ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[shared_ty])),
            ]
        )
    else:
        statements.append(ir.Bind(view, ir.Call("tirx.alloc_tensor", [], ty_args=[shared_ty], attrs=attributes)))
    index = base if lanes == 1 else prim.Ramp(base, stride, lanes)
    statements.extend(
        [
            ir.TensorStore(view, [index], ir.TensorLoad(source, [index])),
            ir.TensorStore(output, [index], ir.TensorLoad(view, [index])),
        ]
    )
    function = tirx.Function([source_ptr, output_ptr, base], statements, ret_type=ir.PrimType("void"))
    function = function.with_attr("global_symbol", "volatile_bf16_copy")
    function = function.with_attr("calling_conv", ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    assert tirx.analysis.check_well_formed(function)
    return function


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("lanes,stride", [(1, 1), (4, 3)])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("volatile", [False, True])
def test_volatile_bf16_raw_bits(lanes, stride, alias, volatile, tmp_path):
    from tilelang.backend.module import get_backend

    get_backend("cuda")  # Direct FFI builds need the owning compiler callback.
    torch.cuda.set_device(0)
    function = _copy_function(lanes, stride, alias, volatile)
    original = ir.save_json(function)
    (tmp_path / "input.py").write_text(function.script())
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda")(
        ir.IRModule({"volatile_bf16_copy": function}),
        target.Target({"kind": "cuda", "arch": "sm_100a"}),
    )
    assert ir.save_json(function) == original
    source = module.inspect_source("cuda")
    (tmp_path / "kernel.cu").write_text(source)
    if volatile and (lanes == 1 or stride != 1):
        assert "bfloat16_t::bitcast(" in source
        assert "volatile bfloat16_t*" in source
        assert re.search(r"\.storage\s*=", source)
    if not volatile:
        assert "volatile bfloat16_t*" not in source

    # Copy exactly, including both zeros, subnormals, infinities and NaN payloads.
    bits = [
        0,
        0x8000,
        1,
        0x8001,
        0x3F80,
        0xBF80,
        0x7F80,
        0xFF80,
        0x7FC1,
        0xFFC2,
        0x7F81,
        0xFF83,
        0x1234,
        0x4321,
        0x3F81,
        0xBF81,
    ]
    values = [(bits[i % len(bits)] + 0x8000) % 0x10000 - 0x8000 for i in range(64)]
    raw = torch.tensor(values, dtype=torch.int16, device="cuda")
    actual = torch.full_like(raw, 0x5A5A)
    for base in range(1, 17):
        actual.fill_(0x5A5A)
        module["volatile_bf16_copy"](ctypes.c_void_p(raw.data_ptr()), ctypes.c_void_p(actual.data_ptr()), base)
        torch.cuda.synchronize()
        expected = torch.full_like(raw, 0x5A5A)
        expected[base : base + lanes * stride : stride] = raw[base : base + lanes * stride : stride]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if volatile and alias:
        ptx = _nvcc(source, tmp_path, ptx=True).read_text()
        width = "16"
        assert re.search(r"ld\.volatile\.shared\.(?:b|u)" + width, ptx), ptx
        assert re.search(r"st\.volatile\.shared\.(?:b|u)" + width, ptx), ptx
