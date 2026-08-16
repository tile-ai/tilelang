"""Coalesced load vs transposed store: the two sides pull apart.

The copy ``A -> frag`` wants the fragment coalesced along j (A's last
axis); the loop stores ``B[j, i]``, which is coalesced along i instead.
No layout satisfies both, so this is where the two selection policies
can legitimately disagree: register count cannot see the difference
(every non-replicated candidate holds the same slot count), while the
io-aware model weighs the actual transaction counts of both statements.

There is no hand-written invariant — the goldens ARE the expectation,
and a diff between the two models' goldens is the interesting signal.
"""

import tilelang.language as T


def _transposed(M, N, dtype, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((N, M), dtype)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((M, N), dtype)
            T.copy(A, frag)
            for i, j in T.Parallel(M, N):
                B[j, i] = frag[i, j]

    return main


VARIANTS = {
    "fp32_128x128_t128": lambda: _transposed(128, 128, T.float32, 128),
    "fp16_64x256_t128": lambda: _transposed(64, 256, T.float16, 128),
}


# The trade-off the io-aware model made, visible in real codegen: the fp32
# pick sacrifices the load (A scalar) for vectorized stores (B 4-wide); the
# fp16 variant trades the opposite way (A 8-wide, B scalar). If either side
# of the asymmetry drifts, the model's belief and the vectorizer diverged.
VECTOR_ANCHOR = {
    "fp32_128x128_t128": {"A": 1, "B": 4},
    "fp16_64x256_t128": {"A": 8, "B": 1},
}
