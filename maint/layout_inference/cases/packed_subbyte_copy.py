"""Packed 4-bit copies must assign both nibbles of a byte to one thread.

With 128 logical elements and 128 threads, the minimum-padding heuristic would
normally select one element per thread. Packed 4-bit stores cannot use that
layout: adjacent elements share one writable byte, and scalar stores are
byte-level read-modify-writes. The inferred copy loop must therefore retain a
two-element vector on each active thread.
"""

import tilelang.language as T


def _copy(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


def _fragment_pipeline(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((128,), dtype)
            B_local = T.alloc_fragment((128,), dtype)
            T.copy(A, A_local)
            for i in T.Parallel(128):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    return main


VARIANTS = {
    "int4_n128_t128": lambda: _copy(T.int4),
    "uint4_n128_t128": lambda: _copy(T.dtype("uint4")),
    "fp4_n128_t128": lambda: _copy(T.float4_e2m1fn),
    "int4_fragment_pipeline_n128_t128": lambda: _fragment_pipeline(T.int4),
}


VECTOR_ANCHOR = {
    "int4_n128_t128": {"A": 2, "B": 2},
    "uint4_n128_t128": {"A": 2, "B": 2},
    "fp4_n128_t128": {"A": 2, "B": 2},
    "int4_fragment_pipeline_n128_t128": {"A": 2, "B": 2},
}
