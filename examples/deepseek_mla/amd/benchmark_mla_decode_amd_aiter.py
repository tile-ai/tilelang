# This benchmark script is modified based on: https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py
# ruff: noqa
import argparse
import math
import random
import torch

import triton
import triton.language as tl

import tilelang
from tilelang.profiler import do_bench

try:
    from aiter.mla import mla_decode_fwd
except ImportError:
    print("aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device.")


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q,
                h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_torch, lse_torch = ref_mla()
    t = triton.testing.do_bench(ref_mla)
    return out_torch, lse_torch, t


@torch.inference_mode()
def run_mla_aiter(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    assert d > dv, "mla with rope dim should be larger than no rope dim"

    qo_indptr = torch.zeros(b + 1, dtype=torch.int)
    kv_indptr = torch.zeros(b + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(b, dtype=torch.int)
    seq_lens_qo.fill_(1)
    max_seqlen_qo = seq_lens_qo.max().item()

    kv_indptr[1 : b + 1] = torch.cumsum(cache_seqlens, dim=0)
    qo_indptr[1 : b + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()

    # set block_size to 1
    page_size = 1
    kv_buffer = blocked_k.view(-1, page_size, h_kv, d)

    flat_indices = []
    for i in range(b):
        start = i * max_seqlen_pad
        end = start + cache_seqlens[i]
        flat_indices.append(torch.arange(start, end, dtype=torch.int))

    kv_indices = torch.cat(flat_indices)

    kv_last_page_lens = torch.ones(b, dtype=torch.int)

    sm_scale = 1.0 / (d**0.5)

    def mla_aiter():
        out_aiter = torch.empty((total_q, h_q, dv), dtype=dtype).fill_(-1)
        attn_logits_aiter, attn_lse_aiter = mla_decode_fwd(
            q.view((total_q, h_q, d)),
            kv_buffer,
            out_aiter,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
        )
        return out_aiter.view([b, s_q, h_q, dv])

    out_aiter = mla_aiter()
    t = triton.testing.do_bench(mla_aiter)
    return out_aiter, None, t


FUNC_TABLE = {
    "torch": run_torch_mla,
    "mla_aiter": run_mla_aiter,
}


def compare_ab(baseline, target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(
        f"comparing {baseline} vs {target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}"
    )
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert baseline in FUNC_TABLE
    assert target in FUNC_TABLE
    baseline_func = FUNC_TABLE[baseline]
    target_func = FUNC_TABLE[target]

    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)

    out_a, lse_a, perf_a = baseline_func(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )
    out_b, lse_b, perf_b = target_func(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )

    torch.testing.assert_close(out_b.float(), out_a.float(), atol=1e-2, rtol=1e-2), "out"
    if target not in ["mla_aiter"]:
        # flash_mla_triton doesn't return lse
        torch.testing.assert_close(lse_b.float(), lse_a.float(), atol=1e-2, rtol=1e-2), "lse"

    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"perf {baseline}: {perf_a:.3f} ms, {FLOPS / 10**9 / perf_a:.3f} TFLOPS, {bytes / 10**6 / perf_a:.3f} GB/s")
    print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10**9 / perf_b:.3f} TFLOPS, {bytes / 10**6 / perf_b:.3f} GB/s")
    return bytes / 10**6 / perf_a, bytes / 10**6 / perf_b


def compare_a(target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(f"{target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert target in FUNC_TABLE, f"target {target} not in {FUNC_TABLE}"
    target_func = FUNC_TABLE[target]

    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)

    out_b, lse_b, perf_b = target_func(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype
    )

    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10**9 / perf_b:.3f} TFLOPS, {bytes / 10**6 / perf_b:.3f} GB/s")
    return bytes / 10**6 / perf_b


available_targets = [
    "torch",
    "mla_aiter",
]

shape_configs = [
    {
        "b": batch,
        "s_q": 1,
        "cache_seqlens": torch.tensor([seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device="cuda"),
        "h_q": head,
        "h_kv": 1,
        "d": 512 + 64,
        "dv": 512,
        "causal": True,
        "dtype": torch.bfloat16,
    }
    for batch in [64, 128]
    for seqlen in [1024, 2048, 4096, 8192, 16384]
    for head in [128]
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="torch")
    parser.add_argument("--target", type=str, default="mla_aiter")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    benchmark_type = "all" if args.all else f"{args.baseline}_vs_{args.target}" if args.compare else args.target
    with open(f"{benchmark_type}_perf.csv", "w") as fout:
        fout.write("name,batch,seqlen,head,bw\n")
        for shape in shape_configs:
            if args.all:
                for target in available_targets:
                    perf = compare_a(
                        target,
                        shape["b"],
                        shape["s_q"],
                        shape["cache_seqlens"],
                        shape["h_q"],
                        shape["h_kv"],
                        shape["d"],
                        shape["dv"],
                        shape["causal"],
                        shape["dtype"],
                    )
                    fout.write(
                        f"{target},{shape['b']},{shape['cache_seqlens'].float().mean().cpu().item():.0f},{shape['h_q']},{perf:.0f}\n"
                    )
            elif args.compare:
                perfa, prefb = compare_ab(
                    args.baseline,
                    args.target,
                    shape["b"],
                    shape["s_q"],
                    shape["cache_seqlens"],
                    shape["h_q"],
                    shape["h_kv"],
                    shape["d"],
                    shape["dv"],
                    shape["causal"],
                    shape["dtype"],
                )
                fout.write(
                    f"{args.baseline},{shape['b']},{shape['cache_seqlens'].float().mean().cpu().item():.0f},{shape['h_q']},{perfa:.0f}\n"
                )
                fout.write(
                    f"{args.target},{shape['b']},{shape['cache_seqlens'].float().mean().cpu().item():.0f},{shape['h_q']},{prefb:.0f}\n"
                )
            elif args.one:
                perf = compare_a(
                    args.target,
                    shape["b"],
                    shape["s_q"],
                    shape["cache_seqlens"],
                    shape["h_q"],
                    shape["h_kv"],
                    shape["d"],
                    shape["dv"],
                    shape["causal"],
                    shape["dtype"],
                )
                fout.write(
                    f"{args.target},{shape['b']},{shape['cache_seqlens'].float().mean().cpu().item():.0f},{shape['h_q']},{perf:.0f}\n"
                )
