from tilelang.language import _intrinsics
import tvm_ffi as _ffi
from tvm_tirx import ir as _ir

import tilelang.language.common as T

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> _ir.Expr:
    """Initialize CUDA curand random number generator state

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr
        Sequence number for parallel random number generation.
    off : PrimExpr
        Offset number for parallel random number generation.
    generator : StringImm
        Set random generator.
        See https://docs.nvidia.com/cuda/curand/group__DEVICE.html

    Returns
    -------
    state : PrimExpr
        The random number generator state handle.
    """
    assert generator in ["curandStateMRG32k3a_t", "curandStatePhilox4_32_10_t", "curandStateXORWOW_t"]
    seed = _ffi.convert(seed)
    if seq is None:
        bx = T.get_block_binding()
        ex = T.get_thread_extent()
        tx = T.get_thread_binding()
        id = tx + bx * ex
        seq = _ffi.convert(id)
    else:
        seq = _ffi.convert(seq)
    off = _ffi.convert(off)
    return _intrinsics.call_intrin("void", _ir.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> _ir.Expr:
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return _intrinsics.call_intrin("uint32", _ir.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> _ir.Expr:
    """Generate a random float

    Parameters
    ----------
    bit : int = [32, 64]
        Bitwidth of random float.
    dist : StringImm = ["uniform", "normal"]
        Random distribution.

    Returns
    -------
    random_value : PrimExpr
        A random float.
    """
    assert bit in [32, 64]
    assert dist in ["uniform", "normal"]
    return _intrinsics.call_intrin("float" + str(bit), _ir.Op.get("tl.rng_rand_float"), dist)
