from tvm import tir
import tilelang.language as T


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t"):
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


def rng_rand():
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tir.call_intrin("uint32", tir.op.Op.get("tl.rng_rand"))


def rng_rand_uniform():
    """Generate a uniformly distributed float

    Returns
    -------
    random_value : PrimExpr
        A 32-bit uniformly distributed float.
    """
    return tir.call_intrin("float32", tir.op.Op.get("tl.rng_rand_uniform"))


def rng_rand_uniform_double():
    """Generate a uniformly distributed double

    Returns
    -------
    random_value : PrimExpr
        A 64-bit uniformly distributed double.
    """
    return tir.call_intrin("float64", tir.op.Op.get("tl.rng_rand_uniform_double"))


def rng_rand_normal():
    """Generate a normally distributed float

    Returns
    -------
    random_value : PrimExpr
        A 32-bit normally distributed float.
    """
    return tir.call_intrin("float32", tir.op.Op.get("tl.rng_rand_normal"))


def rng_rand_normal_double():
    """Generate a normally distributed double

    Returns
    -------
    random_value : PrimExpr
        A 64-bit normally distributed double.
    """
    return tir.call_intrin("float64", tir.op.Op.get("tl.rng_rand_normal_double"))
