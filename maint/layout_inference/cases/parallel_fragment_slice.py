"""A pointwise fragment slice must not seed an invalid full-buffer layout.

The partial ``T.Parallel`` write is exact and may remain legal when another
operator supplies a complete fragment layout.  Free inference must discard the
cheap slice-derived candidate whose forward thread map escapes the block, then
fall back to the full-copy layout.

The non-rectangular ``f[8:24, :]`` boundary is covered by the issue regression
test because this harness records successful inference results only.
"""

import tilelang.language as T


M, N, THREADS = 32, 64, 128


def _overlay(off, rows):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float16),
        Patch: T.Tensor((rows, N), T.float16),
        B: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(1, threads=THREADS):
            frag = T.alloc_fragment((M, N), T.float32)
            T.copy(A, frag, coalesced_width=8)
            for i, j in T.Parallel(rows, N):
                frag[i + off, j] = Patch[i, j]
            T.copy(frag, B, coalesced_width=8)

    return main


VARIANTS = {
    "offset3_rows8": lambda: _overlay(3, 8),
    "offset16_rows8": lambda: _overlay(16, 8),
}


def check(variant, model, result):
    frag = result["buffers"]["frag"]
    assert frag["threads"] <= THREADS, f"fragment escapes the thread range: {frag}"
    assert frag["thread_range"] == [0, THREADS]
