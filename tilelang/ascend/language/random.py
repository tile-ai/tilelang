"""Ascend dialect RNG ops.

Same tl.rng_* emission as the common implementation, but rng_init's default
sequence id derives from the thread context — and on Ascend the thread domain
lives in the enclosing T.SimtVF scope, so the derivation must go through this
dialect's thread accessors rather than the launch-frame ones the CUDA module
consults.
"""

from __future__ import annotations

from tvm import tirx

from tilelang.cuda.language.random import (
    rng_init as _common_rng_init,
    rng_rand,
    rng_rand_float,
)

from .kernel import get_block_binding, get_thread_binding, get_thread_extent

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]


def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> tirx.PrimExpr:
    """Initialize the random number generator state.

    See :func:`tilelang.cuda.language.random.rng_init`; the only difference is
    that the default ``seq`` (one stream per lane) is derived from the Ascend
    thread scope: the SimtVF lane id plus the AI-core index times the lane
    count.
    """
    if seq is None:
        bx = get_block_binding()
        ex = get_thread_extent()
        tx = get_thread_binding()
        seq = tx + bx * ex
    return _common_rng_init(seed, seq=seq, off=off, generator=generator)
