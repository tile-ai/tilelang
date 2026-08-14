"""fp16 load -> fp32 compute -> fp32 store chain.

One connected component with two fragments of different element widths.
The free-mode search must keep the pair consistent (same logical point on
the same thread) while the copy at each end wants vectorization sized by
its own dtype (8 x fp16 = 4 x fp32 = 16B).  Anchors that the chain stays
coalesced end-to-end and that neither fragment picks up replication.
"""

import tilelang.language as T


def _chain(M, N, threads):

    @T.prim_func
    def main(A: T.Tensor((M, N), T.float16), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=threads):
            x16 = T.alloc_fragment((M, N), T.float16)
            x32 = T.alloc_fragment((M, N), T.float32)
            T.copy(A, x16)
            for i, j in T.Parallel(M, N):
                x32[i, j] = x16[i, j].astype(T.float32) * 2.0
            T.copy(x32, B)

    return main


VARIANTS = {
    "128x128_t128": lambda: _chain(128, 128, 128),
    "64x512_t256": lambda: _chain(64, 512, 256),
}


def check(variant, model, result):
    for name in ("x16", "x32"):
        frag = result["buffers"][name]
        assert frag["replicate"] == 1, f"{name} needs no replication in a pure elementwise chain, got: {frag}"
