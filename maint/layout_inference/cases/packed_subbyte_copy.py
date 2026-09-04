"""A packed int4 copy must assign both nibbles of a byte to one thread.

With 128 logical elements and 128 threads, the minimum-padding heuristic would
normally select one element per thread. The inferred copy loop must instead
retain a two-element vector so each participating thread owns one writable
byte.
"""

import tilelang.language as T


def _copy():
    @T.prim_func
    def main(A: T.Tensor((128,), T.int4), B: T.Tensor((128,), T.int4)):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


VARIANTS = {
    "int4_n128_t128": _copy,
}


VECTOR_ANCHOR = {
    "int4_n128_t128": {"A": 2, "B": 2},
}
