"""Row reduction feeding a broadcast subtract: the softmax-shaped pattern.

One component holding a copy-in, a T.reduce_max, a broadcast-consuming
parallel loop, and a copy-out. The reduced fragment ``row_max`` has one
element per row while every column lane needs it, so its replication is
the interesting degree of freedom; the elementwise fragments must stay
unreplicated. This is the most common real-kernel shape (softmax /
layernorm epilogues) the free-mode search faces.
"""

import tilelang.language as T


def _softmaxish(M, N, threads):

    @T.prim_func
    def main(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=threads):
            x = T.alloc_fragment((M, N), T.float32)
            row_max = T.alloc_fragment((M,), T.float32)
            y = T.alloc_fragment((M, N), T.float32)
            T.copy(A, x)
            T.reduce_max(x, row_max, dim=1)
            for i, j in T.Parallel(M, N):
                y[i, j] = x[i, j] - row_max[i]
            T.copy(y, B)

    return main


VARIANTS = {
    "64x128_t128": lambda: _softmaxish(64, 128, 128),
    "128x256_t256": lambda: _softmaxish(128, 256, 256),
}


def check(variant, model, result):
    for name in ("x", "y"):
        frag = result["buffers"][name]
        assert frag["replicate"] == 1, f"elementwise fragment {name} needs no replication, got: {frag}"
