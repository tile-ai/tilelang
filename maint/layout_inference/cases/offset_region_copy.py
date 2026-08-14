"""Tiled copies with block-index region offsets: the multi-block shape.

Every real kernel copies ``A[by*BM, bx*BN]``-style sub-regions, so the
global-side region mins carry block indices — the "foreign vars" the cost
model's address evaluation treats as a uniform shift. Anchors that offset
regions score and rank exactly like their zero-offset counterparts (the
layout must match elementwise_copy's shape family, coalesced and
unreplicated).
"""

import tilelang.language as T


def _tiled(M, N, BM, BN, dtype, threads):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            frag = T.alloc_fragment((BM, BN), dtype)
            T.copy(A[by * BM, bx * BN], frag)
            T.copy(frag, B[by * BM, bx * BN])

    return main


VARIANTS = {
    "fp16_tile64x64_t128": lambda: _tiled(256, 256, 64, 64, T.float16, 128),
    "fp32_tile32x128_t128": lambda: _tiled(128, 512, 32, 128, T.float32, 128),
}


def check(variant, model, result):
    frag = result["buffers"]["frag"]
    assert frag["replicate"] == 1, f"tiled roundtrip copy needs no replication, got: {frag}"
