# Copyright 2018 The apache/tvm Authors. All Rights Reserved.
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# The code below is mostly copied from mlc.ai quantization.py in mlc-llm.
# pylint: disable=invalid-name,missing-function-docstring,unused-variable
"""TIR computation utilities for quantization."""

from tilelang import language as T
from tilelang import tvm as tvm
from tvm import tir


# fmt: off
def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr,
                          dtype: str):
    """
        Convert a packed 4-bit field stored in a uint8 into a bfloat16 value using an exponent scale.

        This function expects a storage field of width `nbit == 4` packed into the 8-bit input `val` and returns
        a bfloat16 constructed from the unpacked sign, a scaled exponent, and the 1-bit mantissa.

        Behavior:
        - Validates `nbit == 4`, `dtype == T.bfloat16`, and `val.dtype == T.uint8` (AssertionError if violated).
        - Extracts the 4-bit field at position `pos` (fields are packed consecutively in `val`).
        - Interprets the 4-bit field as: sign = bit3, exponent = bits1-2, mantissa = bit0.
        - Converts the 2-bit exponent to bf16 exponent space by adding a bias of 126, adds `scale` to that exponent,
        and clamps the result to the 8-bit exponent range (0..255).
        - Assembles a 16-bit bfloat16 bit pattern from (sign, biased-and-scaled-exponent, mantissa) and
        returns it reinterpreted as `bfloat16`.

        Parameters:
        - nbit: must be 4 (width of the packed field).
        - val: uint8 expression containing packed fields.
        - pos: index of the field within `val` (0-based); used to compute the bit shift.
        - scale: exponent-scale to add to the converted exponent (treated as an unsigned integer expression).
        - dtype: must be T.bfloat16.

        Returns:
        - A tir.PrimExpr of dtype "bfloat16" representing the decoded and scaled value.
        """
    assert nbit == 4
    assert dtype == T.bfloat16
    assert val.dtype == T.uint8
    mask = tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)
    e_f4 = (f4 & tir.const(6, T.uint16)) >> tir.const(1, T.uint16)
    # Exponential bias between f4 and bf16 is 2^(8-1) - 2^(2-1) = 126
    e_bf16 = e_f4 + tir.const(126, T.uint16)
    # Scale is the exponential part, within the representation of uint8
    # To handle the overflow, we use the max function to limit the exponential part to 8 bits
    e_bf16 = min(e_bf16 + scale, tir.const((1 << 8) - 1, T.uint16))
    m_f4 = f4 & tir.const(1, T.uint16)
    val_bf16 = tir.reinterpret(T.bfloat16,
                               ((((s << tir.const(8, T.uint16)) | e_bf16) << tir.const(7, T.uint16))
                                | (m_f4 << tir.const(6, T.uint16))).astype(T.uint16))
    return val_bf16

def _tir_f32x2_to_bf16x2_to_u32(v0: tir.PrimExpr, v1: tir.PrimExpr, round_to_even: bool = True):
    """
    Convert two float32 values to bfloat16 and pack them into a single uint32.

    The two inputs v0 and v1 (float32 PrimExpr) are reinterpreted as uint32 bit patterns, optionally rounded to nearest-even
    by adding a rounding bias, then truncated to their upper 16 bits (bfloat16 representation). The two 16-bit results are
    packed into a uint32 with v0 in the lower 16 bits and v1 in the upper 16 bits.

    Parameters:
        v0 (tir.PrimExpr): First float32 value to convert and pack.
        v1 (tir.PrimExpr): Second float32 value to convert and pack.
        round_to_even (bool): If True, apply round-to-nearest-even bias before truncation (default True).

    Returns:
        tir.PrimExpr: A uint32 PrimExpr containing the packed bfloat16 representations (v0 low 16 bits, v1 high 16 bits).
    """
    mask = tir.const((1 << 16) - 1, T.uint32)
    res = []
    for data in [v0, v1]:
        u32_val = tir.reinterpret(T.uint32, data)
        if round_to_even:
            rounding_bias = ((u32_val >> tir.const(16, T.uint32))
                             & tir.const(1, T.uint32)) + tir.const(0x7FFF, T.uint32)
            u32_val += rounding_bias
        res.append((u32_val >> tir.const(16, T.uint32)) & mask)
    return res[0] | (res[1] << tir.const(16, T.uint32))


def _tir_u32_to_bf16x2_to_f32x2(x: tir.PrimExpr):
    mask = tir.const((1 << 16) - 1, T.uint32)
    x0 = x & mask
    x1 = (x >> 16) & mask
    return (tir.reinterpret(T.float32, x << tir.const(16, T.uint32)) for x in [x0, x1])


def _tir_u32_to_int_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert val.dtype == T.uint32
    mask = tvm.tir.const((1 << nbit) - 1, T.uint32)
    return tir.Cast(dtype, (val >> (pos * nbit).astype(T.uint32)) & mask)


def _tir_packed_uint_to_uint_to_float(storage_nbit: int):
    storage_dtype = "uint" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        max_int_value = (1 << (nbit - 1)) - 1
        return ((val >> (pos.astype(T.uint32) * tir.const(nbit, T.uint32))) & tir.const(
            (1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

    return f_convert


def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tir.const((1 << nbit) - 1, T.int32)
        unextended = (val >> (pos.astype(T.int32) * tir.const(nbit, T.int32))) & mask
        return tir.Cast(
            dtype, (unextended << tir.const(32 - nbit, T.int32)) >> tir.const(32 - nbit, T.int32))

    return f_convert


def _tir_f32_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == T.float32
    val_u32 = tir.reinterpret(T.uint32, val)
    # e_f32 >  120 -> e_f4 = min(e_f32 - 120 + M_h, 7)
    # e_f32 == 120 -> e_f4 = 1
    # e_f32 < 120 -> e_f4 = 0
    m_h = (val_u32 >> tir.const(22, T.uint32)) & tir.const(1, T.uint32)
    e_f32 = (val_u32 >> tir.const(23, T.uint32)) & tir.const(255, T.uint32)
    s = (val_u32 >> tir.const(31, T.uint32))
    e_f4 = tir.Select(
        e_f32 > tir.const(120, T.uint32),
        tir.Min(e_f32 - tir.const(120, T.uint32) + m_h, tir.const(7, T.uint32)),
        tir.Select(e_f32 == tir.const(120, T.uint32), tir.const(1, T.uint32),
                   tir.const(0, T.uint32)))
    return (s << tir.const(3, T.uint32)) | e_f4


def _tir_f16_to_uint_to_f4(val: tir.PrimExpr):
    assert val.dtype == T.float16
    val_u32 = tir.Cast(T.uint32, tir.reinterpret(T.uint16, val))
    m_h = (val_u32 >> tir.const(9, T.uint32)) & tir.const(1, T.uint32)
    e_f16 = (val_u32 >> tir.const(10, T.uint32)) & tir.const(31, T.uint32)
    s = (val_u32 >> tir.const(15, T.uint32))
    e_f4 = tir.Select(
        e_f16 > tir.const(8, T.uint32),
        tir.Min(e_f16 - tir.const(8, T.uint32) + m_h, tir.const(7, T.uint32)),
        tir.Select(e_f16 == tir.const(8, T.uint32), tir.const(1, T.uint32), tir.const(0, T.uint32)))
    return (s << tir.const(3, T.uint32)) | e_f4


def _tir_u32_to_f4_to_f32(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.float32
    assert val.dtype == T.uint32
    # e_f4 == 0 -> e_f32 = 0
    # e_f4 != 0 -> e_f32 = e_f4 + 120 = e_f4 | (1111000)_2
    mask = tvm.tir.const((1 << nbit) - 1, T.uint32)
    f4 = (val >> (pos.astype(T.uint32) * tir.const(nbit, T.uint32))) & mask
    s = f4 >> tir.const(3, T.uint32)
    e_f4 = f4 & tir.const(7, T.uint32)
    e_f32 = e_f4 | tir.const(120, T.uint32)
    val_f32 = tir.reinterpret(T.float32,
                              (e_f32 | (s << tir.const(8, T.uint32))) << tir.const(23, T.uint32))
    return tir.Select(e_f4 == tir.const(0, T.uint32), tir.const(0, T.float32), val_f32)


def _tir_packed_to_fp4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.float16
    assert val.dtype == T.uint32
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + 8 = e_f4 | (1000)_2
    mask = tvm.tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)
    e_f4 = f4 & tir.const(7, T.uint16)
    e_f16 = e_f4 | tir.const(8, T.uint16)
    val_f16 = tir.reinterpret(T.float16,
                              ((e_f16 | (s << tir.const(5, T.uint16))) << tir.const(10, T.uint16)).astype(T.uint16))
    return tir.Select(e_f4 == tir.const(0, T.uint16), tir.const(0, T.float16), val_f16)

def _tir_packed_to_fp4_to_f16(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        f4 = ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(storage_dtype)
        f4 = (val >> (pos.astype(storage_dtype) * tir.const(nbit, storage_dtype))) & mask
        s = f4 >> tir.const(3, storage_dtype)
        e_f4 = f4 & tir.const(7, storage_dtype)
        e_f16 = e_f4 | tir.const(8, storage_dtype)
        val_f16 = tir.reinterpret(T.float16,
                                ((e_f16 | (s << tir.const(5, storage_dtype))) << tir.const(10, storage_dtype)).astype(T.uint16))
        return tir.Select(e_f4 == tir.const(0, storage_dtype), tir.const(0, T.float16), val_f16)

    return f_convert

def _tir_u8_to_f8_e4m3_to_f16_naive(nbit: int, val: tir.PrimExpr, dtype: str):
    assert nbit == 8
    assert dtype == T.float16
    s_f16 = (val >> tir.const(7, T.uint16)) << tir.const(15, T.uint16)
    e4 = val & tir.const(0x40, T.uint16)
    prefix = tir.Select(e4 == tir.const(0, T.uint16), tir.const(0x2000, T.uint16),
                        tir.const(0x4000, T.uint16))
    e_f16 = ((val & tir.const(63, T.uint16)) << tir.const(7, T.uint16)) | prefix
    return tir.reinterpret(T.float16, s_f16 | e_f16)


def _tir_u8_to_f8_e4m3_to_f16(nbit: int, val: tir.PrimExpr, dtype: str):
    assert nbit == 8
    assert dtype == T.float16
    s_f16 = (val >> tir.const(7, T.uint16)) << tir.const(15, T.uint16)
    e4 = val & tir.const(0x40, T.uint16)
    e_f16 = ((val & tir.const(63, T.uint16)) << tir.const(7, T.uint16)) | (e4 << tir.const(8, T.uint16)) | (e4 << tir.const(7, T.uint16))
    e_f16 = e_f16 ^ tir.const(0x2000, T.uint16)
    return tir.reinterpret(T.float16, s_f16 | e_f16)


def _tir_u8_to_f8_e5m2_to_f16(nbit: int, val: tir.PrimExpr, dtype: str):
    assert nbit == 8
    assert dtype == T.float16
    return tir.reinterpret("float8_e5m2", val).astype(T.float16)


def _tir_packed_to_signed_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        max_int_value = (1 << (nbit - 1))
        return ((val >> (pos.astype(T.uint32) * tir.const(nbit, T.uint32))) & tir.const(
            (1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)

    return f_convert


def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)

    return f_convert


def _tir_packed_to_unsigned_convert_with_zeros(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, zero: tvm.tir.PrimExpr,
                  dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return (((val >> (pos * nbit).astype(storage_dtype)) & mask) - zero).astype(dtype)

    return f_convert


def _tir_packed_int_to_int_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tir.const((1 << nbit) - 1, T.int32)
        unextended = (val >> (pos.astype(T.int32) * tir.const(nbit, T.int32))) & mask
        return tir.Cast(
            dtype, (unextended << tir.const(32 - nbit, T.int32)) >> tir.const(32 - nbit, T.int32))

    return f_convert


# fmt: on
