"""Compilation entry point for BF16 sparse prefill."""

import tilelang

from examples.ascend.flashmla.kernel import sparse_attention


def compile_prefill(nq, nk, topk, **kwargs):
    return tilelang.compile(
        sparse_attention(nq, nk, topk, kv_format="bf16", **kwargs),
        target="ascend",
        out_idx=[5, 6, 7],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
