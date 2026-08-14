"""Broadcast read of a tiny fragment inside a wide parallel loop (#1729).

``s_frag`` has fewer elements than there are threads, and every thread
reads it while streaming ``Out``.  This is THE case the two cost models
disagree on, and the disagreement is the point:

  - register-count picks a 1-slot partially-replicated ``s_frag`` (fewer
    registers) whose induced loop layout collapses the j axis onto each
    thread — every thread serially walks all of j, with the whole loop
    redundantly replicated across threads.  That is the issue #1729
    pathology, preserved in the golden as the documented legacy baseline.
  - io-aware must reject it via the issue term: ``s_frag`` fully
    replicated (replicate == thread count) and a non-replicated,
    coalesced loop layout over ``Out``.  Enforced by ``check`` below.
"""

import tilelang.language as T


def _broadcast(rows, cols, threads):

    @T.prim_func
    def main(S: T.Tensor((rows,), T.float32), Out: T.Tensor((rows, cols), T.float32)):
        with T.Kernel(1, threads=threads):
            s_frag = T.alloc_fragment((rows,), T.float32)
            T.copy(S, s_frag)
            for i, j in T.Parallel(rows, cols):
                Out[i, j] = s_frag[i] * 2.0

    return main


VARIANTS = {
    # The shape from issue #1729: (2,) fragment under T.Parallel(2, 2560).
    "issue1729_2x2560_t256": lambda: _broadcast(2, 2560, 256),
    "rows4_cols1024_t128": lambda: _broadcast(4, 1024, 128),
}

_THREADS = {
    "issue1729_2x2560_t256": 256,
    "rows4_cols1024_t128": 128,
}


def check(variant, model, result):
    if model != "io-aware":
        return  # legacy model keeps the #1729 pathology; golden documents it
    frag = result["buffers"]["s_frag"]
    threads = _THREADS[variant]
    assert frag["replicate"] == threads, f"broadcast-read fragment must be fully replicated over {threads} threads, got: {frag}"
    (loop,) = result["loops"].values()
    assert loop["replicate"] == 1, f"loop layout must not replicate execution, got: {loop}"
