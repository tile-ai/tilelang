from __future__ import annotations

from ptodsl import pto


class PhiloxRNG:
    """Philox 4x32-10 PRNG for PTO SIMT path.

    Each SIMT thread creates its own instance with a unique (seed, seq) pair.
    The seed and initial 128-bit counter are immutable. The caller owns a
    logical draw index and threads it through each method explicitly, so
    PTODSL can carry the ordinary Python SSA name through runtime control
    flow. ``rand()`` returns a uint32; ``rand_uniform()`` returns float32 in
    [0, 1); ``rand_normal()`` uses the same cached Box-Muller construction as
    the AscendC backend.
    """

    _PHILOX_M4X32_A = 0xD2511F53
    _PHILOX_M4X32_B = 0xCD9E8D57
    _PHILOX_W32_A = 0x9E3779B9
    _PHILOX_W32_B = 0xBB67AE85
    _RAND_2POW32_INV = 2.3283064e-10
    _RAND_2POW32_INV_HALF = _RAND_2POW32_INV / 2.0
    _BOX_MULLER_EPS = 1.0e-7
    _BOX_MULLER_TWO = 2.0
    _BOX_MULLER_TWO_PI = 6.283185307179586

    @staticmethod
    def _as_ui64(value):
        return pto.cast(value, pto.ui64)

    @staticmethod
    def _split_ui64(value):
        value = pto.cast(value, pto.ui64)
        low = pto.cast(value, pto.i32)
        high = pto.cast(
            value >> pto.const(32, dtype=pto.ui64),
            pto.i32,
        )
        return low, high

    def __init__(self, seed, seq, off):
        seed = self._as_ui64(seed)
        seq = self._as_ui64(seq)
        off = self._as_ui64(off)
        self.key0, self.key1 = self._split_ui64(seed)

        # Offsets count 32-bit draws while Philox advances in 4x32 blocks.
        # Keep the intra-block lane so non-aligned offsets select the exact
        # requested draw without overflowing an `(off + 3)` ceil-division.
        four = pto.const(4, dtype=pto.ui64)
        self.base_counter_lo = off // four
        self.base_counter_hi = seq
        self.base_lane = off % four

    def rand(self, draw_index):
        four = pto.const(4, dtype=pto.ui64)
        block_delta = draw_index // four
        lane_offset = self.base_lane + draw_index % four
        lane_block_delta = lane_offset // four
        lane = pto.cast(lane_offset % four, pto.i32)

        counter_before_lane = self.base_counter_lo + block_delta
        carry = pto.cast(counter_before_lane < self.base_counter_lo, pto.ui64)
        counter_lo = counter_before_lane + lane_block_delta
        carry = carry + pto.cast(counter_lo < counter_before_lane, pto.ui64)
        counter_hi = self.base_counter_hi + carry
        c0, c1 = self._split_ui64(counter_lo)
        c2, c3 = self._split_ui64(counter_hi)
        k0 = self.key0
        k1 = self.key1
        _m0 = pto.const(self._PHILOX_M4X32_A, dtype=pto.i32)
        _m1 = pto.const(self._PHILOX_M4X32_B, dtype=pto.i32)
        _k0 = pto.const(self._PHILOX_W32_A, dtype=pto.i32)
        _k1 = pto.const(self._PHILOX_W32_B, dtype=pto.i32)
        for _r in range(10):
            _lo0 = c0 * _m0
            _hi0 = pto.mulhi(c0, _m0, signedness="unsigned")
            _lo1 = c2 * _m1
            _hi1 = pto.mulhi(c2, _m1, signedness="unsigned")
            c0 = _hi1 ^ c1 ^ k0
            c1 = _lo1
            c2 = _hi0 ^ c3 ^ k1
            c3 = _lo0
            k0 = k0 + _k0
            k1 = k1 + _k1

        one = pto.const(1, dtype=pto.i32)
        two = pto.const(2, dtype=pto.i32)
        result = pto.select(lane == one, c1, c0)
        result = pto.select(lane == two, c2, result)
        result = pto.select(lane == pto.const(3, dtype=pto.i32), c3, result)
        return result, draw_index + pto.const(1, dtype=pto.ui64)

    def rand_uniform(self, draw_index):
        _u, draw_index = self.rand(draw_index)
        # Philox words are carried as i32 bit patterns. Re-author the value as
        # unsigned before converting so high-bit words map to positive floats.
        _u = pto.cast(_u, pto.ui32)
        _f = pto.cast(
            _u,
            pto.f32,
            rounding="to_nearest_even",
            saturation="nosat",
        )
        result = _f * pto.const(self._RAND_2POW32_INV, dtype=pto.f32) + pto.const(self._RAND_2POW32_INV_HALF, dtype=pto.f32)
        return result, draw_index

    def rand_normal(self, draw_index, normal_cache, has_normal):
        """Return one normal draw and the updated cached Box-Muller state."""
        initial_draw_index = draw_index
        u1, draw_index = self.rand_uniform(draw_index)
        u2, draw_index = self.rand_uniform(draw_index)

        eps = pto.const(self._BOX_MULLER_EPS, dtype=pto.f32)
        u1 = pto.select(u1 < eps, eps, u1)
        angle = pto.const(self._BOX_MULLER_TWO_PI, dtype=pto.f32) * u2
        radius = pto.sqrt(pto.const(-self._BOX_MULLER_TWO, dtype=pto.f32) * pto.log(u1))
        # AscendC calls sincosf(angle, &normal0, &normal1), whose output
        # pointer order is sine followed by cosine.
        normal0 = radius * pto.sin(angle)
        normal1 = radius * pto.cos(angle)

        zero = pto.const(0, dtype=pto.i32)
        one = pto.const(1, dtype=pto.i32)
        use_cache = has_normal != zero
        result = pto.select(use_cache, normal_cache, normal0)
        # pto.select strips integer signedness; re-author as ui64 so the
        # loop-carried draw index keeps its type.
        next_draw_index = self._as_ui64(pto.select(use_cache, initial_draw_index, draw_index))
        next_normal_cache = pto.select(use_cache, normal_cache, normal1)
        next_has_normal = pto.select(use_cache, zero, one)
        return (
            result,
            next_draw_index,
            next_normal_cache,
            next_has_normal,
        )
