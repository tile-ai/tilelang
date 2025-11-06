import tilelang.language as T

@T.macro
def _rand_parallel_impl(buffer: T.Buffer, seed, total_elems, n_rounds, dtype="float32"):
    T.call_extern(
        "tl::philox_rand_kernel",
        T.address_of(buffer[0]),
        total_elems,
        seed,
        n_rounds,
        dtype=dtype,
    )


def rand(buffer: T.Buffer, seed, n_rounds: int = 10):
    total_elems = 1
    for dim in buffer.shape:
        total_elems *= dim

    _rand_parallel_impl(buffer, seed, total_elems, n_rounds)
