"""BF16 prefill entry point for the explicitly scheduled sparse attention core."""

import tilelang
from tilelang import tvm
from examples.ascend.flashmla.core import PASS_CONFIG, sparse_attention


def prefill(nq, nk, topk, **kwargs):
    func = sparse_attention(nq, nk, topk, **kwargs)
    # The BF16 specialization never references the three decode-only inputs.
    # Remove them from its public ABI; the pipeline implementation stays shared.
    params = list(func.params[:5]) + list(func.params[8:])
    param_set = set(params)
    return tvm.tirx.PrimFunc(
        params, func.body, func.ret_type, {var: buf for var, buf in func.buffer_map.items() if var in param_set}, func.attrs
    )


def compile_prefill(nq, nk, topk, **kwargs):
    return tilelang.compile(
        prefill(nq, nk, topk, **kwargs),
        target="ascend",
        out_idx=[5, 6, 7],
        pass_configs=PASS_CONFIG,
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
