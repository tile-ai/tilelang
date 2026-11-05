"""Logical TF32 keeps float32 storage without weakening packed tensor checks."""

from functools import lru_cache

import numpy as np
import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.testing
from tilelang.testing.packed import compile_packed_function

from test_native_packed_subbyte_abi import _descriptor, _invoke


def _tf32(lanes=1):
    return "custom[tfloat32]32" + (f"x{lanes}" if lanes != 1 else "")


@lru_cache(None)
def _shape_entry(dtype, tilelang_owned=True, fixed_extent=None):
    size = ir.Var("size", "int64") if fixed_extent is None else ir.const(fixed_extent, "int64")
    tensor = ir.Var("input", tirx.TensorType([size], dtype))
    function = tirx.Function([tensor], ir.Return(size), ir.PrimType("int64"))
    function = function.with_attr("global_symbol", "main")
    return compile_packed_function(function, tilelang_owned=tilelang_owned)


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_tf32_accepts_float32_with_the_same_lane_count_and_shape(lanes):
    storage = "float32" + (f"x{lanes}" if lanes != 1 else "")
    for size in (0, 1, 7):
        assert _invoke(_shape_entry(_tf32(lanes)), _descriptor((size,), storage)) == size


@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("tilelang_owned", [False, True])
def test_exact_tf32_storage_is_still_accepted(lanes, tilelang_owned):
    logical = _tf32(lanes)
    assert _invoke(_shape_entry(logical, tilelang_owned), _descriptor((7,), logical)) == 7


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_tf32_alias_does_not_change_the_upstream_binder(lanes):
    storage = "float32" + (f"x{lanes}" if lanes != 1 else "")
    with pytest.raises(TypeError, match="dtype"):
        _invoke(_shape_entry(_tf32(lanes), False), _descriptor((7,), storage))


@pytest.mark.parametrize("storage", ["float16", "float64", "bfloat16", "int32", "uint32"])
def test_tf32_alias_rejects_other_storage_dtypes(storage):
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_shape_entry(_tf32()), _descriptor((7,), storage))


@pytest.mark.parametrize("field,value", [("bits", 16), ("bits", 64), ("lanes", 0), ("lanes", 2), ("code", 131), ("code", 239)])
def test_tf32_alias_keeps_exact_bits_lanes_and_named_storage_codes(field, value):
    descriptor = _descriptor((7,), "float32")
    setattr(descriptor.dtype, field, value)
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_shape_entry(_tf32()), descriptor)


def test_float32_does_not_accept_a_reverse_tf32_alias():
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_shape_entry("float32"), _descriptor((7,), _tf32()))


def test_tf32_alias_does_not_relax_shape_or_compact_stride_checks():
    entry = _shape_entry(_tf32(), fixed_extent=7)
    with pytest.raises(RuntimeError, match="shape"):
        _invoke(entry, _descriptor((8,), "float32"))
    with pytest.raises(RuntimeError, match="stride"):
        _invoke(entry, _descriptor((7,), "float32", strides=(2,)))


@pytest.mark.parametrize("callee", [False, True])
def test_tf32_input_storage_alias_reaches_allocated_output_worker(callee):
    import tilelang.cpu.language as T

    size = T.dynamic("size", "int64")

    @T.prim_func
    def fill(A: T.Tensor((size,), T.tfloat32), B: T.Tensor((size,), "int32")):
        with T.Kernel(1):
            for index in T.serial(size):
                B[index] = T.int32(index) + 2

    if callee:
        fill = fill.with_attr("tilelang_callee_allocated_outputs", True)
    kernel = tilelang.compile(fill, out_idx=[1], target="c", target_host="c", execution_backend="tvm_ffi")
    for extent in (0, 1, 17):
        source = torch.zeros(extent + 3, dtype=torch.float32)[3:]
        actual = kernel(source)
        torch.testing.assert_close(actual, torch.arange(extent, dtype=torch.int32) + 2, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(torch.zeros(7, dtype=torch.int32))


@tilelang.testing.requires_cuda
def test_tf32_copy_uses_float32_storage_without_rounding_or_conversion():
    import tilelang.language as T

    @T.prim_func
    def copy(A: T.Tensor((8,), T.tfloat32), B: T.Tensor((8,), T.tfloat32)):
        with T.Kernel(1, threads=32):
            T.copy(A, B)

    assert str(copy.params[0].ty.dtype) == str(T.tfloat32)
    kernel = tilelang.compile(copy, out_idx=[1], target="cuda", target_host="c", execution_backend="tvm_ffi")
    bits = np.array([0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x3F800001, 0xBF800FFF, 0x00000001, 0], dtype="uint32")
    source = torch.from_numpy(bits).cuda().view(torch.float32)
    result = kernel(source)
    torch.cuda.synchronize()
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(result.view(torch.uint32).cpu().numpy(), bits)
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(torch.zeros(8, dtype=torch.float16, device="cuda"))
