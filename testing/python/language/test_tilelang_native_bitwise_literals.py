"""Separate native-object literals from unchanged TileLang source operators."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir

import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def test_native_explicit_mask_retains_ir_contract():
    value = T.uint32()
    result = value & T.uint32(0xFFFFFFFF)
    assert isinstance(value, ir.Var)
    assert result.ty == ir.PrimType("uint32")


@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
def test_source_bare_bitwise_masks(dtype):
    mask = (1 << int(dtype[4:])) - 1

    @T.prim_func
    def masks(A: T.Tensor((4,), dtype), B: T.Tensor((3, 4), dtype)):
        for i in T.serial(4):
            B[0, i] = A[i] & mask
            B[1, i] = mask | A[i]
            B[2, i] = A[i] ^ mask

    executable = compile_packed_function(masks)
    inputs = np.asarray([0, 1, 42, mask], dtype=dtype)
    output = np.zeros((3, 4), dtype=dtype)
    executable(tvm_ffi.from_dlpack(inputs), tvm_ffi.from_dlpack(output))
    expected = np.stack([inputs & mask, mask | inputs, inputs ^ mask])
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
def test_source_augmented_bitwise_mask(dtype):
    mask = (1 << int(dtype[4:])) - 1

    @T.prim_func
    def augmented(A: T.Tensor((4,), dtype), B: T.Tensor((4,), dtype)):
        for i in T.serial(4):
            B[i] = A[i]
            B[i] &= mask
            B[i] ^= mask

    executable = compile_packed_function(augmented)
    inputs = np.asarray([0, 1, 42, mask], dtype=dtype)
    output = np.zeros_like(inputs)
    executable(tvm_ffi.from_dlpack(inputs), tvm_ffi.from_dlpack(output))
    np.testing.assert_array_equal(output, inputs ^ mask)
