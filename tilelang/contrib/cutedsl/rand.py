"""Philox4x32-10 random numbers for the CuTeDSL backend.

This is a port of the ``curandStatePhilox4_32_10_t`` part of
``curand_kernel.h``. The generator state lives in a 16-word register array
because it has to survive across conditional code, which is how curand keeps
it in a local struct.

Uniform integers, uniform floats and both normal siblings reproduce curand
bit for bit, including the Box-Muller transform and its cached second sample.
"""

from __future__ import annotations

__all__ = ["rng_state", "rng_init", "rng_rand", "rng_rand_float", "rng_uniform2_double"]

import struct

import cutlass.cute as cute
from cutlass._mlir.dialects import arith as _arith_raw
from cutlass._mlir_helpers.dialect_proxy import DialectAutoConvertProxy
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.cute.typing import Float32, Float64, Uint32, Uint64

_arith = DialectAutoConvertProxy(_arith_raw)

# Word offsets inside the state: counter, current output block, key, position
# in the output block, then the Box-Muller caches.
_CTR = 0
_OUT = 4
_KEY = 8
_STATE = 10
_NORMAL_FLAG = 11
_NORMAL_EXTRA = 12
_NORMAL_DOUBLE_FLAG = 13
_NORMAL_DOUBLE_EXTRA = 14
STATE_WORDS = 16

_PHILOX_W32_0 = 0x9E3779B9
_PHILOX_W32_1 = 0xBB67AE85
_PHILOX_M4x32_0 = 0xD2511F53
_PHILOX_M4x32_1 = 0xCD9E8D57


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


# curand_globals.h, with the float products rounded the way the compiler sees
# them.
_CURAND_2POW32_INV = _float32(2.3283064e-10)
_CURAND_2POW32_INV_HALF = _float32(_CURAND_2POW32_INV / 2)
_CURAND_2POW32_INV_2PI = _float32(_CURAND_2POW32_INV * _float32(6.2831855))
_CURAND_2POW32_INV_2PI_HALF = _float32(_CURAND_2POW32_INV_2PI / 2)
_CURAND_2POW32_INV_DOUBLE = 2.3283064365386963e-10
_CURAND_2POW53_INV_DOUBLE = 1.1102230246251565e-16


def _select(cond, true_value, false_value):
    """``arith.select`` on two scalars of the same type."""
    return type(true_value)(
        _arith.select(
            cond.ir_value() if hasattr(cond, "ir_value") else cond,
            true_value.ir_value() if hasattr(true_value, "ir_value") else true_value,
            false_value.ir_value() if hasattr(false_value, "ir_value") else false_value,
        )
    )


def _mulhilo(a, b):
    product = Uint64(a) * Uint64(b)
    return Uint32(product >> Uint64(32)), Uint32(product)


def _philox_round(ctr, key):
    hi0, lo0 = _mulhilo(Uint32(_PHILOX_M4x32_0), ctr[0])
    hi1, lo1 = _mulhilo(Uint32(_PHILOX_M4x32_1), ctr[2])
    return (hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0)


def _philox4x32(ctr, key):
    kx, ky = key
    for _ in range(9):
        ctr = _philox_round(ctr, (kx, ky))
        kx = kx + Uint32(_PHILOX_W32_0)
        ky = ky + Uint32(_PHILOX_W32_1)
    return _philox_round(ctr, (kx, ky))


def _counter(state):
    return (state[_CTR], state[_CTR + 1], state[_CTR + 2], state[_CTR + 3])


def _key(state):
    return (state[_KEY], state[_KEY + 1])


def _refresh_output(state):
    output = _philox4x32(_counter(state), _key(state))
    for i in range(4):
        state[_OUT + i] = output[i]


def _incr_counter(state, n):
    """Philox_State_Incr: add ``n`` (a Uint64) to the 128-bit counter."""
    nlo = Uint32(n)
    nhi = Uint32(n >> Uint64(32))
    x = state[_CTR]
    nx = x + nlo
    nhi = nhi + Uint32(nx < x)
    y = state[_CTR + 1]
    ny = y + nhi
    carry = Uint32(ny < nhi)
    z = state[_CTR + 2]
    nz = z + carry
    carry = Uint32(nz < carry)
    state[_CTR] = nx
    state[_CTR + 1] = ny
    state[_CTR + 2] = nz
    state[_CTR + 3] = state[_CTR + 3] + carry


def _incr_counter_hi(state, n):
    """Philox_State_Incr_hi: advance the counter by ``n`` subsequences."""
    nlo = Uint32(n)
    nhi = Uint32(n >> Uint64(32))
    z = state[_CTR + 2]
    nz = z + nlo
    state[_CTR + 2] = nz
    state[_CTR + 3] = state[_CTR + 3] + nhi + Uint32(nz < nlo)


def _skipahead(state, n):
    position = state[_STATE] + Uint32(n & Uint64(3))
    blocks = n >> Uint64(2)
    wrapped = position > Uint32(3)
    blocks = blocks + Uint64(_select(wrapped, Uint32(1), Uint32(0)))
    state[_STATE] = _select(wrapped, position - Uint32(4), position)
    _incr_counter(state, blocks)
    _refresh_output(state)


def _skipahead_sequence(state, n):
    _incr_counter_hi(state, n)
    _refresh_output(state)


@dsl_user_op
def rng_state(*, loc=None, ip=None):
    """Allocate the generator state as a register array."""
    return cute.make_rmem_tensor(STATE_WORDS, Uint32, loc=loc, ip=ip)


@dsl_user_op
def rng_init(state, seed, seq, off, generator="curandStatePhilox4_32_10_t", *, loc=None, ip=None) -> None:
    """Initialize the generator, like ``curand_init(seed, seq, off, &state)``."""
    if generator != "curandStatePhilox4_32_10_t":
        raise NotImplementedError(
            f"The CuTeDSL backend implements the Philox4x32-10 generator only; got generator={generator!r}. "
            "Pass generator='curandStatePhilox4_32_10_t' to T.rng_init."
        )
    for i in range(4):
        state[_CTR + i] = Uint32(0)
    seed = Uint64(seed)
    state[_KEY] = Uint32(seed)
    state[_KEY + 1] = Uint32(seed >> Uint64(32))
    state[_STATE] = Uint32(0)
    state[_NORMAL_FLAG] = Uint32(0)
    state[_NORMAL_EXTRA] = Uint32(0)
    state[_NORMAL_DOUBLE_FLAG] = Uint32(0)
    state[_NORMAL_DOUBLE_EXTRA] = Uint32(0)
    state[_NORMAL_DOUBLE_EXTRA + 1] = Uint32(0)
    _skipahead_sequence(state, Uint64(seq))
    _skipahead(state, Uint64(off))


@dsl_user_op
def rng_rand(state, *, loc=None, ip=None):
    """Return the next 32-bit unsigned value, like ``curand(&state)``."""
    position = state[_STATE]
    output = (state[_OUT], state[_OUT + 1], state[_OUT + 2], state[_OUT + 3])
    ret = _select(
        position == Uint32(3),
        output[3],
        _select(
            position == Uint32(2),
            output[2],
            _select(position == Uint32(1), output[1], output[0]),
        ),
    )
    position = position + Uint32(1)
    wrapped = position == Uint32(4)
    state[_STATE] = _select(wrapped, Uint32(0), position)

    def _advance():
        _incr_counter(state, Uint64(1))
        _refresh_output(state)

    if_generate(wrapped, _advance)
    return ret


def _box_muller32(x, y):
    """``_curand_box_muller``: one normal sample and its companion."""
    u = Float32(x) * Float32(_CURAND_2POW32_INV) + Float32(_CURAND_2POW32_INV_HALF)
    v = Float32(y) * Float32(_CURAND_2POW32_INV_2PI) + Float32(_CURAND_2POW32_INV_2PI_HALF)
    scale = cute.math.sqrt(Float32(-2.0) * cute.math.log(u))
    # curand's __sincosf is the fast approximation; the precise sin/cos differ
    # from it in the last bits.
    sin_v, cos_v = cute.math.sincos(v, fastmath=True)
    return sin_v * scale, cos_v * scale


def _box_muller64(x0, x1, y0, y1):
    """``_curand_box_muller_double``: one normal sample of 64-bit precision."""
    zx = Uint64(x0) ^ (Uint64(x1) << Uint64(21))
    u = Float64(zx) * Float64(_CURAND_2POW53_INV_DOUBLE) + Float64(_CURAND_2POW53_INV_DOUBLE / 2.0)
    zy = Uint64(y0) ^ (Uint64(y1) << Uint64(21))
    v = Float64(zy) * Float64(_CURAND_2POW53_INV_DOUBLE * 2.0) + Float64(_CURAND_2POW53_INV_DOUBLE)
    scale = cute.math.sqrt(Float64(-2.0) * cute.math.log(u))
    sin_v, cos_v = cute.math.sincos(v * Float64(3.141592653589793), fastmath=True)
    return sin_v * scale, cos_v * scale


def _rand4(state):
    return tuple(rng_rand(state) for _ in range(4))


def _uniform_double_hq(x0, x1):
    """``_curand_uniform_double_hq``: 53 random bits per double."""
    z = Uint64(x0) ^ (Uint64(x1) << Uint64(21))
    return Float64(z) * Float64(_CURAND_2POW53_INV_DOUBLE) + Float64(_CURAND_2POW53_INV_DOUBLE / 2.0)


@dsl_user_op
def rng_uniform2_double(state, out, *, loc=None, ip=None) -> None:
    """Fill two doubles with ``curand_uniform2_double``, which draws one block.

    The 64-bit uniform store is the one place where curand's vector helper is
    not the elementwise composition of its scalar form: it takes 53-bit samples
    from a whole Philox block instead of one 32-bit word per result.
    """
    x0, x1, y0, y1 = _rand4(state)
    out[0] = _uniform_double_hq(x0, x1)
    out[1] = _uniform_double_hq(y0, y1)


@dsl_user_op
def rng_rand_float(state, bit=32, dist="uniform", *, loc=None, ip=None):
    """Return the next random float, like ``curand_uniform`` / ``curand_normal``."""
    if dist == "uniform":
        x = rng_rand(state)
        if bit == 64:
            return Float64(x) * Float64(_CURAND_2POW32_INV_DOUBLE) + Float64(_CURAND_2POW32_INV_DOUBLE)
        return Float32(x) * Float32(_CURAND_2POW32_INV) + Float32(_CURAND_2POW32_INV_HALF)

    if dist != "normal":
        raise ValueError(f"Unsupported distribution {dist!r}; expected 'uniform' or 'normal'.")

    if bit == 64:

        def _fresh():
            x0, x1, y0, y1 = _rand4(state)
            first, second = _box_muller64(x0, x1, y0, y1)
            bits = Float64(second).bitcast(Uint64)
            state[_NORMAL_DOUBLE_EXTRA] = Uint32(bits)
            state[_NORMAL_DOUBLE_EXTRA + 1] = Uint32(bits >> Uint64(32))
            state[_NORMAL_DOUBLE_FLAG] = Uint32(1)
            return first

        def _cached():
            state[_NORMAL_DOUBLE_FLAG] = Uint32(0)
            bits = Uint64(state[_NORMAL_DOUBLE_EXTRA]) | (Uint64(state[_NORMAL_DOUBLE_EXTRA + 1]) << Uint64(32))
            return Float64(bits.bitcast(Float64))

        return if_generate(state[_NORMAL_DOUBLE_FLAG] != Uint32(1), _fresh, _cached, return_types=[Float64])

    def _fresh():
        first, second = _box_muller32(rng_rand(state), rng_rand(state))
        state[_NORMAL_EXTRA] = Float32(second).bitcast(Uint32)
        state[_NORMAL_FLAG] = Uint32(1)
        return first

    def _cached():
        state[_NORMAL_FLAG] = Uint32(0)
        return Float32(state[_NORMAL_EXTRA].bitcast(Float32))

    return if_generate(state[_NORMAL_FLAG] != Uint32(1), _fresh, _cached, return_types=[Float32])
