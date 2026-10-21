"""fp8 (1-byte dtype) roundtrip: the widest vectorization the model knows.

With 1-byte elements the 128-bit lane cap allows 16-element vectors, the
widest candidate the shared width policy admits. Anchors that both models
keep the fragment unreplicated and that the io-aware model's width
handling doesn't regress at the extreme end of the dtype range.
"""

import tilelang.language as T


def _roundtrip(M, N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), T.float8_e4m3), B: T.Tensor((M, N), T.float8_e4m3)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((M, N), T.float8_e4m3)
            T.copy(A, frag)
            T.copy(frag, B)

    return main


VARIANTS = {
    "128x256_t128": lambda: _roundtrip(128, 256, 128),
    "64x128_t64": lambda: _roundtrip(64, 128, 64),
}


def check(variant, model, result):
    frag = result["buffers"]["frag"]
    assert frag["replicate"] == 1, f"roundtrip copy needs no replication, got: {frag}"


# 1-byte dtype reaches the 128-bit lane cap: 16 elements per access.
VECTOR_ANCHOR = {
    "128x256_t128": {"A": 16, "B": 16},
    "64x128_t64": {"A": 16, "B": 16},
}
