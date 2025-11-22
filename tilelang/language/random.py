import tilelang.language as T
import tvm.tir as tir


@T.macro
def _rand_parallel_impl(buffer: T.Buffer,
                        seed,
                        total_elems,
                        block_m,
                        block_n,
                        n_rounds,
                        dtype="float32"):
    T.call_intrin(
        "handle",
        tir.op.Op.get("tl.philox_rand"),
        buffer.access_ptr("r"),
        total_elems,
        block_m,
        block_n,
        seed,
        n_rounds,
    )


def rand(buffer: T.Buffer, seed, n_rounds: int = 10):
    total_elems = 1
    for dim in buffer.shape:
        total_elems *= dim

    if len(buffer.shape) == 2:
        block_m = buffer.shape[0]
        block_n = buffer.shape[1]
    elif len(buffer.shape) == 1:
        block_m = buffer.shape[0]
        block_n = 1
    else:
        raise ValueError(
            f"Only support 1D or 2D buffer, but got {len(buffer.shape)}: {buffer.shape}")

    _rand_parallel_impl(buffer, seed, total_elems, block_m, block_n, n_rounds)
