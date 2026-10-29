"""Compilation entry point for BF16 sparse prefill."""

import tilelang
from tilelang import tvm

from examples.ascend.flashmla.kernel import sparse_attention


def compile_prefill(nq, nk, topk, **kwargs):
    # UB ND-to-NZ copy layout inference currently needs an active target context.
    with tvm.target.Target("ascend"):
        return tilelang.compile(
            sparse_attention(nq, nk, topk, kv_format="bf16", **kwargs),
            target="ascend",
            out_idx=[5, 6, 7],
            compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
        )
