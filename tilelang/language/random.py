from tvm import tir
import tilelang.language as T


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> tir.PrimExpr:
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
    seed = tir.convert(seed)
    if seq is None:
        bx = T.get_block_binding()
        ex = T.kernel.get_thread_extent()
        tx = T.get_thread_binding()
        id = tx + bx * ex
        seq = tir.convert(id)
    else:
        seq = tir.convert(seq)
    off = tir.convert(off)
    return tir.call_intrin("void", tir.op.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> tir.PrimExpr:
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tir.call_intrin("uint32", tir.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> tir.PrimExpr:
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
    return tir.call_intrin("float" + str(bit), tir.op.Op.get("tl.rng_rand_float"), dist)
