"""Compilation entry point for FP8 main and optional FP8/FP4 extra KV caches."""

import tilelang

from examples.ascend.flashmla.kernel import sparse_attention


def compile_decode(nq, nk, topk, **kwargs):
    return tilelang.compile(
        sparse_attention(nq, nk, topk, kv_format="fp8", **kwargs),
        target="ascend",
        out_idx=[8, 9],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
