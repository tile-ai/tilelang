from tvm import tir
from typing import Any
import tilelang.language as T
from tilelang.language.utils import index_to_coordinates

@T.macro
def umulhi_uint32(a, b):
    return T.Cast("uint32", (T.Cast('uint64', a) * T.Cast('uint64', b)) >> tir.const(32, "uint64"))

@T.macro
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds):
    PHILOX_KEY_A = tir.const(0x9E3779B9, "uint32")
    PHILOX_KEY_B = tir.const(0xBB67AE85, "uint32")
    PHILOX_ROUND_A = tir.const(0xD2511F53, "uint32")
    PHILOX_ROUND_B = tir.const(0xCD9E8D57, "uint32")

    c0_var = T.alloc_var("uint32", init=c0)
    c1_var = T.alloc_var("uint32", init=c1)
    c2_var = T.alloc_var("uint32", init=c2)
    c3_var = T.alloc_var("uint32", init=c3)
    k0_var = T.alloc_var("uint32", init=k0)
    k1_var = T.alloc_var("uint32", init=k1)
    
    for _ in T.serial(n_rounds):
        _c0 = c0_var
        _c2 = c2_var
        A = PHILOX_ROUND_A
        B = PHILOX_ROUND_B
        c0_var = umulhi_uint32(B, _c2) ^ c1_var ^ k0_var
        c2_var = umulhi_uint32(A, _c0) ^ c3_var ^ k1_var
        c1_var = T.Cast("uint32", T.Cast("uint64", B) * T.Cast("uint64", _c2))
        c3_var = T.Cast("uint32", T.Cast("uint64", A) * T.Cast("uint64", _c0))
        k0_var = T.Cast("uint32", T.Cast("uint64", k0_var) + PHILOX_KEY_A)
        k1_var = T.Cast("uint32", T.Cast("uint64", k1_var) + PHILOX_KEY_B)
    return c0_var, c1_var, c2_var, c3_var

@T.macro
def uint32_to_uniform_float(x: tir.PrimExpr) -> tir.PrimExpr:
    assert x.dtype == 'uint32' or x.dtype == "int32", f"x.dtype {x.dtype} is not supported"
    x_int32 = T.reinterpret('int32', x)
    scale = tir.const(4.6566127342e-10, "float32")

    x_abs = T.if_then_else(x_int32 < 0, -x_int32 - 1, x_int32)

    return T.Cast("float32", x_abs) * scale

@T.macro
def _rand_parallel_impl(buffer: T.Buffer, seed_lo, seed_hi, total_elems, n_rounds):
    for i in T.Parallel(total_elems):
        coords = index_to_coordinates(i, buffer.shape)
        offset = T.Cast("uint32", i)
        offset_lo = offset
        offset_hi = tir.const(0, "uint32")

        c0, c1, c2, c3 = philox_impl(offset_lo, offset_hi, tir.const(0, "uint32"), tir.const(0, "uint32"),
        seed_lo, seed_hi, n_rounds)

        rand_float = uint32_to_uniform_float(c0)
        buffer[coords] = rand_float


def rand(buffer: T.Buffer, seed, n_rounds: int = 10):
    seed = T.Cast("uint64", seed)
    seed_lo = T.Cast("uint32", seed & tir.const(0xffffffff, "uint64"))
    seed_hi = T.Cast("uint32", (seed >> 32) & tir.const(0xffffffff, "uint64"))
    
    total_elems = 1
    for dim in buffer.shape:
        total_elems *= dim
    
    _rand_parallel_impl(buffer, seed_lo, seed_hi, total_elems, n_rounds)

