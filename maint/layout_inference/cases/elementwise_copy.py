"""Global -> fragment -> global roundtrip: the baseline sanity case.

Both directions of the roundtrip want the same thing — the fragment's
last logical axis split into (vector run per thread, coalesced lanes) —
so both cost models must land on the same coalesced, vectorized layout.
This case anchors the invariant that turning the io-aware model ON does
not perturb layouts the register-count model already gets right, and it
is the primary equal-score anchor for any future fast-path/slow-path
split inside the cost model.
"""

import tilelang.language as T


def _roundtrip(M, N, dtype, threads):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((M, N), dtype)
            T.copy(A, frag)
            T.copy(frag, B)

    return main


VARIANTS = {
    "fp16_128x128_t128": lambda: _roundtrip(128, 128, T.float16, 128),
    "fp32_64x256_t256": lambda: _roundtrip(64, 256, T.float32, 256),
}


def check(variant, model, result):
    frag = result["buffers"]["frag"]
    assert frag["replicate"] == 1, f"roundtrip copy needs no replication, got: {frag}"
