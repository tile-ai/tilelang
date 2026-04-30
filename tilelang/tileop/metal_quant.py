from __future__ import annotations

from dataclasses import dataclass

from tilelang import language as T


FP32 = "float32"


@dataclass(frozen=True)
class QuantSimdgroupTile:
    block_m: int
    block_n: int
    block_k: int
    wm: int
    wn: int


SMALL_TILE = QuantSimdgroupTile(block_m=16, block_n=32, block_k=32, wm=1, wn=1)
LARGE_TILE = QuantSimdgroupTile(block_m=32, block_n=32, block_k=32, wm=1, wn=2)


def use_large_simdgroup_tile(m: int, n: int, *, mixed_fp4_weight: bool = False) -> bool:
    """Shape-only quant contraction selector for Metal packed uint8 probes.

    fp8 x fp8 starts winning once there is enough row and output-column work to
    amortize threadgroup staging.  The mixed fp8/fp4 path has more decode and
    scale traffic, so keep the middle ``N=256`` band on scalar/GEMV schedules
    until a better mixed simdgroup tile lands.
    """
    if mixed_fp4_weight:
        return m >= 64 and (n == 128 or n >= 512)
    return m >= 64 and n >= 256


def selected_simdgroup_tile(m: int, n: int, *, mixed_fp4_weight: bool = False) -> QuantSimdgroupTile:
    return LARGE_TILE if use_large_simdgroup_tile(m, n, mixed_fp4_weight=mixed_fp4_weight) else SMALL_TILE


def use_small_m_gemv(m: int, n: int, *, mixed_fp4_weight: bool = False) -> bool:
    """Shape-only selector for promoted small-M packed quant GEMV schedules."""
    if mixed_fp4_weight:
        if 1 <= m <= 16:
            return n >= 64
        if 17 <= m <= 24:
            return n >= 128
        if 25 <= m <= 32:
            return n == 256
        if 33 <= m <= 48:
            return n >= 128
        return False
    if 1 <= m <= 32:
        return n >= 128
    return False


def fp8_e4m3fn_to_float(bits):
    """Decode packed uint8 e4m3fn to fp32 inside TileLang/Metal kernels."""
    bits_u = T.Cast("uint32", bits)
    abs_bits = bits_u & T.uint32(0x7F)
    sign = (bits_u >> T.uint32(7)) & T.uint32(1)
    exp_bits = (bits_u >> T.uint32(3)) & T.uint32(0xF)
    mant_bits = bits_u & T.uint32(0x7)

    mant = T.Cast(FP32, mant_bits)
    subnormal = mant * T.float32(1.0 / 512.0)
    normal = (T.float32(1.0) + mant * T.float32(1.0 / 8.0)) * T.exp2(
        T.Cast(FP32, T.Cast("int32", exp_bits) - T.int32(7))
    )
    value = T.if_then_else(exp_bits == T.uint32(0), subnormal, normal)
    value = T.if_then_else(abs_bits == T.uint32(0x7F), T.float32(0.0), value)
    return T.if_then_else(sign != T.uint32(0), -value, value)


def fp4_e2m1fn_to_float(bits, nibble_index):
    """Decode one e2m1fn nibble from packed uint8 storage."""
    bits_u = T.Cast("uint32", bits)
    shift = T.Cast("uint32", nibble_index) * T.uint32(4)
    nibble = (bits_u >> shift) & T.uint32(0xF)
    sign = (nibble >> T.uint32(3)) & T.uint32(1)
    mag = nibble & T.uint32(0x7)
    value = T.if_then_else(
        mag == T.uint32(0),
        T.float32(0.0),
        T.if_then_else(
            mag == T.uint32(1),
            T.float32(0.5),
            T.if_then_else(
                mag == T.uint32(2),
                T.float32(1.0),
                T.if_then_else(
                    mag == T.uint32(3),
                    T.float32(1.5),
                    T.if_then_else(
                        mag == T.uint32(4),
                        T.float32(2.0),
                        T.if_then_else(
                            mag == T.uint32(5),
                            T.float32(3.0),
                            T.if_then_else(mag == T.uint32(6), T.float32(4.0), T.float32(6.0)),
                        ),
                    ),
                ),
            ),
        ),
    )
    return T.if_then_else(sign != T.uint32(0), -value, value)


def e8m0_to_float(bits):
    """Decode torch.float8_e8m0fnu-compatible scale byte to fp32."""
    bits_i = T.Cast("int32", bits)
    value = T.exp2(T.Cast(FP32, bits_i - T.int32(127)))
    return T.if_then_else(bits_i == T.int32(255), T.float32(0.0), value)
