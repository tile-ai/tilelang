"""Philox4x32-10 RNG helper for TileLang PTO SIMT kernels.

Mirrors the Ascend SIMT reference implementation
(src/tl_templates/ascend/philox_rng.h and random_kernel_base.h) so PTO and
AscendC runs share the same Philox counter/consumption rules and the same
Box-Muller normal transform.

The helper is a trace-time Python object: counter/buffer fields are per-lane
runtime scalar values, while the buffer index and the Box-Muller cache flag are
trace-time Python state. Draws are therefore only well-defined in straight-line
code and statically unrolled loops (pto.static_range). The TileLang PTO codegen
rejects RNG draws inside dynamic loops and branches up front.
"""

from ptodsl import pto, scalar

# Philox4x32-10 constants, identical to random_kernel_base.h.
_PHILOX_M4X32_A = 0xD2511F53
_PHILOX_M4X32_B = 0xCD9E8D57
_PHILOX_W32_A = 0x9E3779B9
_PHILOX_W32_B = 0xBB67AE85

# Uniform mapping constants: u = rand() * 2^-32 + 2^-33.
_RAND_2POW32_INV = 2.3283064e-10
_RAND_2POW32_INV_HALF = 1.1641532182693481e-10

_TWO_PI = 6.2831854820251465  # 2*pi in float32
_NORMAL_EPS = 1.0e-7


def _i32(value):
    # Internal Philox state uses plain signless i32 values: pto.mulhi's
    # verifier rejects ui32 operands (signedness is an op attribute), and the
    # arithmetic below only relies on wrap-around semantics plus sign-agnostic
    # equality compares.
    return pto.const(value & 0xFFFFFFFF, dtype=pto.i32)


def _split_i64_bits(value):
    """Return the low/high i32 words of a Python or runtime 64-bit value."""
    if isinstance(value, int):
        return _i32(value), _i32(value >> 32)

    value64 = scalar.cast(value, pto.i64)
    low = scalar.cast(value64, pto.i32)
    high = pto.mulhi(
        value64,
        pto.const(1 << 32, dtype=pto.i64),
        signedness="unsigned",
    )
    return low, scalar.cast(high, pto.i32)


class PhiloxRNG:
    """Per-lane Philox4x32-10 state for PTO SIMT kernels.

    ``seed``, ``seq``, and ``off`` may be Python integers or runtime integer
    scalars. ``off`` follows the nonnegative int64 contract of the Ascend
    reference and is converted to a Philox block offset with ``ceil(off / 4)``.
    """

    def __init__(self, seed, seq, off=0):
        if isinstance(seed, int) and not 0 <= seed < (1 << 64):
            raise ValueError("PhiloxRNG seed must fit in 64 bits")
        if isinstance(off, int) and not 0 <= off < (1 << 64):
            raise ValueError("PhiloxRNG off must fit in 64 bits")

        self._key0, self._key1 = _split_i64_bits(seed)

        # Counter starts at zero; SkipLo adds ceil(off/4) to the low 64 bits
        # and SkipHi places seq in the high 64 bits. Adding to zero counters
        # cannot carry, so the initial values are direct assignments.
        block_offset = (off + 3) // 4
        self._ctr0, self._ctr1 = _split_i64_bits(block_offset)
        self._ctr2, self._ctr3 = _split_i64_bits(seq)

        self._buf0 = _i32(0)
        self._buf1 = _i32(0)
        self._buf2 = _i32(0)
        self._buf3 = _i32(0)
        self._idx = 4  # 4 == exhausted; force generation on the first draw

        self._normal_cache = None
        self._has_normal = False

    @staticmethod
    def _mulhi(a, b):
        return pto.mulhi(a, b, signedness="unsigned")

    def _round(self, c0, c1, c2, c3, k0, k1):
        mul_a = _i32(_PHILOX_M4X32_A)
        mul_b = _i32(_PHILOX_M4X32_B)
        lo0 = mul_a * c0
        hi0 = self._mulhi(mul_a, c0)
        lo1 = mul_b * c2
        hi1 = self._mulhi(mul_b, c2)
        return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0

    def _generate(self):
        # PhiloxRandomSimt: 10 rounds on temporaries; the key is bumped after
        # every round. State fields are updated only after the block finishes.
        c0, c1, c2, c3 = self._ctr0, self._ctr1, self._ctr2, self._ctr3
        k0, k1 = self._key0, self._key1
        for _ in range(10):
            c0, c1, c2, c3 = self._round(c0, c1, c2, c3, k0, k1)
            k0 = k0 + _i32(_PHILOX_W32_A)
            k1 = k1 + _i32(_PHILOX_W32_B)
        self._buf0, self._buf1, self._buf2, self._buf3 = c0, c1, c2, c3

    def _skip_one(self):
        # 128-bit increment of (ctr0..ctr3) with explicit carry propagation,
        # matching SkipOne in the Ascend reference. Written branch-free:
        # counter values are runtime scalars, so data-dependent Python branches
        # cannot be traced.
        one = _i32(1)
        zero = _i32(0)
        self._ctr0 = self._ctr0 + one
        k1 = scalar.select(self._ctr0 == 0, one, zero)
        self._ctr1 = self._ctr1 + k1
        k2 = scalar.select(self._ctr1 == 0, k1, zero)
        self._ctr2 = self._ctr2 + k2
        k3 = scalar.select(self._ctr2 == 0, k2, zero)
        self._ctr3 = self._ctr3 + k3

    def _rand_i32(self):
        if self._idx >= 4:
            self._generate()
            self._skip_one()
            self._idx = 0
        out = (self._buf0, self._buf1, self._buf2, self._buf3)[self._idx]
        self._idx += 1
        return out

    def rand(self):
        """Draw one raw uint32 from the lane's stream, advancing the state."""
        # Reinterpret as ui32 at the boundary so the value matches uint32
        # buffers; the bit pattern is unchanged.
        return scalar.cast(self._rand_i32(), pto.ui32)

    def rand_uniform(self):
        """Draw a float32 uniformly distributed in [0, 1)."""
        value = pto.convert(
            self._rand_i32(),
            pto.f32,
            rounding="r",
            saturation="nosat",
            signedness="unsigned",
        )
        return value * _RAND_2POW32_INV + _RAND_2POW32_INV_HALF

    def rand_normal(self):
        """Draw a float32 from N(0, 1) via Box-Muller (two draws cached)."""
        if not hasattr(pto, "sin") or not hasattr(pto, "cos"):
            raise RuntimeError(
                "PhiloxRNG.rand_normal requires pto.sin/pto.cos (A5 SIMT "
                "sin/cos SoftLib, PTOAS PR #1193); the current ptodsl build "
                "does not provide them"
            )
        if self._has_normal:
            self._has_normal = False
            return self._normal_cache
        u1 = pto.fmax(self.rand_uniform(), _NORMAL_EPS)
        u2 = self.rand_uniform()
        r = pto.sqrt(pto.log(u1) * -2.0)
        v = u2 * _TWO_PI
        z0 = r * pto.sin(v)
        z1 = r * pto.cos(v)
        self._normal_cache = z1
        self._has_normal = True
        return z0
