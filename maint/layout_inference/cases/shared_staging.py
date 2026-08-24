"""Global -> shared -> fragment -> global staging chain.

The shared buffer picks up a (possibly swizzled) layout and the
shared->fragment copy is deliberately OUTSIDE the io model (only
fragment<->global statements are charged), so the fragment's layout is
decided by the final copy-out alone. Goldens document both the shared
layout and the fragment pick; a model change that suddenly starts charging
shared traffic would surface here first.
"""

import tilelang.language as T


def _staged(M, N, dtype, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, threads=threads):
            smem = T.alloc_shared((M, N), dtype)
            frag = T.alloc_fragment((M, N), dtype)
            T.copy(A, smem)
            T.copy(smem, frag)
            T.copy(frag, B)

    return main


VARIANTS = {
    "fp16_64x64_t128": lambda: _staged(64, 64, T.float16, 128),
    "fp32_128x128_t128": lambda: _staged(128, 128, T.float32, 128),
}


def check(variant, model, result):
    frag = result["buffers"]["frag"]
    assert frag["replicate"] == 1, f"staged roundtrip needs no replication, got: {frag}"


VECTOR_ANCHOR = {
    "fp16_64x64_t128": {"A": 8, "B": 8},
    "fp32_128x128_t128": {"A": 4, "B": 4},
}
