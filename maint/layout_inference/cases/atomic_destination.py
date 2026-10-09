"""Atomic destination legality must participate in the loop layout plan."""

from common import T


def make_atomic(repeated):
    @T.prim_func
    def main(A: T.Tensor((1024,), T.float16), B: T.Tensor((1024,), T.float16)):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((1024,), T.float16)
            for i in T.Parallel(1024):
                values[i] = A[i]
                if repeated:
                    T.atomic_add(B[i // 2], values[i])
                else:
                    T.atomic_add(B[i], values[i])

    return main


VARIANTS = {
    "contiguous": lambda: make_atomic(False),
    "repeated": lambda: make_atomic(True),
}

VECTOR_ANCHOR = {
    "contiguous": {"A": 2},
    "repeated": {"A": 1},
}
