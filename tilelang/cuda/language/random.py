from tvm import tirx
import tilelang.language.common as T

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]


def _default_rng_sequence() -> tirx.PrimExpr:
    """Linear id of the calling thread over the complete logical launch.

    The sequence key decorrelates parallel curand streams. Deriving it from
    ``threadIdx.x``/``blockIdx.x`` alone gives every thread that differs only
    in ``y``/``z`` the same subsequence, so every launched dimension is folded
    in here in row-major order. Dimensions that were not launched have extent 1
    and therefore contribute nothing, which keeps the historical 1-D id
    ``threadIdx.x + blockIdx.x * blockDim.x`` unchanged.
    """
    linear_thread = 0
    threads_per_block = 1
    for binding, extent in zip(T.get_thread_bindings(), T.get_thread_extents()):
        if extent > 1:
            linear_thread = linear_thread + threads_per_block * binding
        threads_per_block = threads_per_block * extent

    linear_block = 0
    blocks_per_grid = 1
    for binding, extent in zip(T.get_block_bindings(), T.get_block_extents()):
        if extent > 1:
            linear_block = linear_block + blocks_per_grid * binding
        blocks_per_grid = blocks_per_grid * extent

    return tirx.convert(linear_thread + threads_per_block * linear_block)


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> tirx.PrimExpr:
    """Initialize CUDA curand random number generator state

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr
        Sequence number for parallel random number generation. When omitted, a
        deterministic default is derived that is unique for every thread of the
        complete launch: it folds in ``threadIdx.y``/``threadIdx.z`` and
        ``blockIdx.y``/``blockIdx.z``, not just their ``x`` components.
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
    seed = tirx.convert(seed)
    if seq is None:
        seq = _default_rng_sequence()
    else:
        seq = tirx.convert(seq)
    off = tirx.convert(off)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> tirx.PrimExpr:
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> tirx.PrimExpr:
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
    return tirx.call_intrin("float" + str(bit), tirx.op.Op.get("tl.rng_rand_float"), dist)
