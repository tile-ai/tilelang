"""Explicitly scheduled FP8 main cache / FP8 or FP4 extra cache attention."""

import torch
import tilelang
from tilelang import tvm
from examples.ascend.flashmla.core import PASS_CONFIG, sparse_attention


def prepare_decode(param, inputs):
    """Prepare the PR's packed KV inputs without changing their quantization."""
    nq = param.decode.b * param.s_q
    q = inputs.q.reshape(nq, 64, 512)
    main = inputs.kv_scope
    extra = inputs.extra_kv_scope
    kv = main.get_kvcache_for_flash_mla()
    if not kv.is_contiguous():
        raise ValueError("The example currently requires contiguous packed KV pages")
    indices = main.indices_in_kvcache.reshape(nq, param.topk)
    sink = inputs.attn_sink if inputs.attn_sink is not None else torch.zeros(64, device=q.device, dtype=torch.float32)
    lengths = (
        main.topk_length.repeat_interleave(param.s_q)
        if main.topk_length is not None
        else torch.full((nq,), param.topk, device=q.device, dtype=torch.int32)
    )
    if extra is not None:
        ekv = extra.get_kvcache_for_flash_mla()
        if not ekv.is_contiguous():
            raise ValueError("The example currently requires contiguous packed extra KV pages")
        etopk = extra.indices_in_kvcache.shape[-1]
        eindices = extra.indices_in_kvcache.reshape(nq, etopk)
        elengths = (
            extra.topk_length.repeat_interleave(param.s_q)
            if extra.topk_length is not None
            else torch.full((nq,), etopk, device=q.device, dtype=torch.int32)
        )
        eformat = "fp4" if ekv.shape[-1] == 288 else "fp8"
    else:
        ekv = torch.empty(1, device=q.device, dtype=torch.uint8)
        etopk = 0
        eindices = torch.zeros((nq, 1), device=q.device, dtype=torch.int32)
        elengths = torch.zeros(nq, device=q.device, dtype=torch.int32)
        eformat = "fp8"
    nbytes = kv.numel() + (ekv.numel() if extra is not None else 0)
    ntokens = kv.numel() // 528 + (ekv.numel() // (288 if eformat == "fp4" else 528) if extra is not None else 0)
    reuse_cache = (
        nbytes <= 8 * 1024**2
        and nq * (param.topk + etopk) >= 4 * ntokens
        and main.topk_length is None
        and (extra is None or extra.topk_length is None)
    )
    func = sparse_attention(
        nq,
        ntokens,
        param.topk,
        sink=param.have_attn_sink,
        variable_lengths=main.topk_length is not None,
        scale=inputs.sm_scale,
        q_strides=tuple(q.stride()),
        index_stride=indices.stride(0),
        kv_format="fp8",
        extra_topk=etopk,
        extra_format=eformat,
        extra_index_stride=eindices.stride(0),
        variable_extra_lengths=extra is not None and extra.topk_length is not None,
        kv_storage_bytes=kv.numel(),
        extra_storage_bytes=ekv.numel(),
        cache_hint=int(reuse_cache),
    )
    params = list(func.params[:9]) + list(func.params[10:])
    param_set = set(params)
    func = tvm.tirx.PrimFunc(
        params, func.body, func.ret_type, {var: buf for var, buf in func.buffer_map.items() if var in param_set}, func.attrs
    )
    kernel = tilelang.compile(
        func,
        target="ascend",
        out_idx=[8, 9],
        pass_configs=PASS_CONFIG,
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
    kv_bytes = kv.view(torch.uint8).view(-1)
    extra_bytes = ekv.view(torch.uint8).view(-1)

    def run():
        # Match the PR wrapper's per-invocation flattening. Non-contiguous
        # [batch, query, ...] tensors can require a device copy here, which
        # affects the attention kernel's initial cache state even when the
        # profiler only times the attention kernel.
        query = inputs.q.reshape(nq, 64, 512)
        main_indices = main.indices_in_kvcache.reshape(nq, param.topk)
        extra_indices = extra.indices_in_kvcache.reshape(nq, etopk) if extra is not None else eindices
        main_lengths = main.topk_length.repeat_interleave(param.s_q) if main.topk_length is not None else lengths
        extra_lengths = extra.topk_length.repeat_interleave(param.s_q) if extra is not None and extra.topk_length is not None else elengths
        out, lse = kernel(query, kv_bytes, main_indices, sink, main_lengths, extra_bytes, extra_indices, extra_lengths)
        return out.reshape(param.decode.b, param.s_q, 64, 512), lse.reshape(param.decode.b, param.s_q, 64).transpose(1, 2).contiguous()

    return kernel, run
