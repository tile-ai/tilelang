from tvm import tir


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq, off):
    """Initialize CUDA curand random number generator state

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr
        Sequence number for parallel random number generation.
    off : PrimExpr
        Offset number for parallel random number generation.

    Returns
    -------
    state : PrimExpr
        The random number generator state handle.
    """
    seed = tir.convert(seed)
    seq = tir.convert(seq)
    off = tir.convert(off)
    return tir.call_intrin("handle", tir.op.Op.get("tl.rng_init"), seed, seq, off)


def rng_rand():
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tir.call_intrin("uint32", tir.op.Op.get("tl.rng_rand"))
