"""Wide integer addresses use typed scalar lanes, without a CUDA int64x8 value."""

import ctypes

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, prim, target, tirx

import tilelang
import tilelang.testing


CASES = [
    ("mixed", False),
    ("mixed", True),
    ("wrap", False),
    ("wrap", True),
    ("nested_shuffle", False),
    ("nested_shuffle", True),
    ("descending", False),
    ("descending", True),
    ("broadcast", False),
]


def _indices(base, case):
    order = [6, 1, 7, 0, 3, 5, 2, 4]
    if case == "mixed":
        vector = prim.Shuffle([base * 8 + i for i in range(8)], order)
        wide = prim.Cast("int64x8", vector)
        return prim.Add(prim.Broadcast(ir.const(3, "int64"), 8), prim.Mul(wide, prim.Broadcast(ir.const(2, "int64"), 8)))
    if case == "wrap":
        vector = prim.Shuffle([base + i for i in range(8)], list(range(8)))
        return prim.Cast("int64x8", prim.Cast("uint8x8", vector))
    if case == "nested_shuffle":
        vector = prim.Shuffle([prim.Ramp(base, 2, 4), prim.Ramp(base + 8, 2, 4)], order)
        return prim.Cast("int64x8", vector)
    if case == "descending":
        return prim.Ramp(prim.Cast("int64", base) + ir.const(14, "int64"), ir.const(-2, "int64"), 8)
    if case == "broadcast":
        return prim.Broadcast(prim.Cast("int64", base), 8)
    raise AssertionError(case)


def _oracle(base, case):
    order = [6, 1, 7, 0, 3, 5, 2, 4]
    if case == "mixed":
        return [(base * 8 + i) * 2 + 3 for i in order]
    if case == "wrap":
        return [(base + i) & 255 for i in range(8)]
    if case == "nested_shuffle":
        return [base + 2 * i for i in order]
    if case == "descending":
        return [base + 14 - 2 * i for i in range(8)]
    if case == "broadcast":
        return [base] * 8
    raise AssertionError(case)


def _function(case, scatter=False):
    pointer_type = ir.PtrType("float32", "global")
    source_ptr, output_ptr = ir.Var("source_ptr", pointer_type), ir.Var("output_ptr", pointer_type)
    base = ir.Var("base", "int32")
    tensor_type = tirx.TensorType((256,), "float32", data_alignment=1)
    source, output = ir.Var("source", tensor_type), ir.Var("output", tensor_type)
    index = _indices(base, case)
    contiguous = prim.Ramp(0, 1, 8)
    load_index, store_index = (contiguous, index) if scatter else (index, contiguous)
    body = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_ptr], ty_args=[tensor_type])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[tensor_type])),
        ir.TensorStore(output, [store_index], ir.TensorLoad(source, [load_index])),
    ]
    function = tirx.Function([source_ptr, output_ptr, base], body, ret_type=ir.PrimType("void"))
    return function.with_attr(
        {
            "global_symbol": "wide_address",
            "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH,
        }
    )


def _build(function, compile_device=False):
    from tilelang.backend.module import get_backend

    get_backend("cuda")
    build = "target.build.tilelang_cuda" + ("" if compile_device else "_without_compile")
    return tvm_ffi.get_global_func(build)(ir.IRModule({"wide_address": function}), target.Target({"kind": "cuda", "arch": "sm_100a"}))


@pytest.mark.parametrize("case,scatter", CASES)
def test_wide_memory_index_source_preserves_native_ir(case, scatter):
    function = _function(case, scatter)
    before = ir.save_json(function)
    assert tirx.analysis.check_well_formed(function)
    module = _build(function)
    source = module.inspect_source("cuda")
    assert "longlong8" not in source
    assert "int64_t" in source
    assert ir.save_json(function) == before


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
@pytest.mark.parametrize("case,scatter", CASES)
def test_wide_memory_indices_execute_exact_lanes(case, scatter, tmp_path):
    torch.cuda.set_device(0)
    module = _build(_function(case, scatter), compile_device=True)
    (tmp_path / "kernel.cu").write_text(module.inspect_source("cuda"))
    source = torch.arange(256, device="cuda", dtype=torch.float32) * 2 - 101
    output = torch.full_like(source, -999)
    bases = [0, 1, 7] if case != "wrap" else [0, 252, 255]
    for base in bases:
        output.fill_(-999)
        module["wide_address"](ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(output.data_ptr()), base)
        torch.cuda.synchronize()
        expected = torch.full_like(output, -999)
        indices = torch.tensor(_oracle(base, case), device="cuda", dtype=torch.int64)
        if scatter:
            expected[indices] = source[:8]
        else:
            expected[:8] = source[indices]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_effectful_wide_index_is_not_duplicated_into_scalar_calls():
    function = _function("broadcast")
    effect = tirx.call_extern("next_index", ty="int64")

    def replace_load(load):
        return ir.TensorLoad(load.source, [prim.Broadcast(effect, 8)])

    function = tvm_ffi.structural_map(function, (ir.TensorLoad, replace_load), order="post")
    before = ir.save_json(function)
    with pytest.raises(tilelang.errors.InternalError, match="Cannot convert type .*int64x8 to CUDA type"):
        _build(function)
    assert ir.save_json(function) == before


def test_bound_effectful_value_is_not_reevaluated_per_index_lane():
    function = _function("broadcast")
    # An explicit scalar Bind snapshots state before the pure lane expansion.
    # Do not use Broadcast(scalar RNG) or a vector-return extern call: TileLang
    # preserves their per-lane call convention in the existing CUDA emitter.
    effect = tirx.call_extern("next_index", ty="int64")
    bound = ir.Var("bound_index", "int64")

    def replace_load(load):
        return ir.TensorLoad(load.source, [prim.Broadcast(bound, 8)])

    function = tvm_ffi.structural_map(function, (ir.TensorLoad, replace_load), order="post")
    function = tirx.Function(function.params, [ir.Bind(bound, effect), *function.body], ret_type=function.ret_type, attrs=function.attrs)
    assert tirx.analysis.check_well_formed(function)
    source = _build(function).inspect_source("cuda")
    assert source.count("next_index()") == 1, source


@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_narrow_vector_arithmetic_is_not_promoted_by_lane_projection(dtype):
    function = _function("broadcast")
    base = function.params[2]
    narrow = prim.Broadcast(prim.Cast(dtype, base), 8)
    offset = prim.Cast(f"{dtype}x8", prim.Ramp(0, 1, 8))
    wide_index = prim.Cast("int64x8", prim.Add(narrow, offset))

    def replace_load(load):
        return ir.TensorLoad(load.source, [wide_index])

    function = tvm_ffi.structural_map(function, (ir.TensorLoad, replace_load), order="post")
    with pytest.raises(tilelang.errors.InternalError, match="Cannot convert type .*int64x8 to CUDA type"):
        _build(function)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
@pytest.mark.parametrize("narrow_wrap", [False, True])
def test_literal_four_lane_indices_keep_vector_access_and_finite_cast_values(narrow_wrap):
    function = _function("broadcast")
    source, output = [binding.var for binding in function.body.seq[:2]]
    values = [254, 255, 256, 257] if narrow_wrap else [3, 4, 5, 6]
    index = prim.Shuffle([ir.const(value) for value in values], list(range(4)))
    if narrow_wrap:
        index = prim.Cast("uint8x4", index)
    index = prim.Cast("int64x4", index)
    zero, one = ir.Var("zero", "int64"), ir.Var("one", "int64")
    index = prim.Add(prim.Broadcast(zero, 4), prim.Mul(index, prim.Broadcast(one, 4)))
    store = ir.TensorStore(output, [prim.Ramp(ir.const(11, "int64"), ir.const(1, "int64"), 4)], source[index])
    function = function.with_body([*function.body.seq[:2], ir.Bind(zero, ir.const(0, "int64")), ir.Bind(one, ir.const(1, "int64")), store])
    before = ir.save_json(function)
    module = _build(function, compile_device=True)
    code = module.inspect_source("cuda")
    assert code.count("*(float4*)") == (1 if narrow_wrap else 2)
    assert ir.save_json(function) == before
    # The source slice plus index 3 and destination slice plus index 11
    # both give 16-byte-aligned vector addresses. The wrapping case remains
    # a scalar gather and must preserve the uint8 cast values.
    data = torch.arange(257, device="cuda", dtype=torch.float32)[1:]
    actual_storage = torch.full((257,), -999, device="cuda", dtype=torch.float32)
    actual = actual_storage[1:]
    module["wide_address"](ctypes.c_void_p(data.data_ptr()), ctypes.c_void_p(actual.data_ptr()), 0)
    torch.cuda.synchronize()
    expected = torch.full_like(actual, -999)
    expected[11:15] = data[torch.tensor([value & 255 for value in values], device="cuda")]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual_storage[0].item() == -999
