import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import itertools
from functools import lru_cache
from typing import Tuple, Dict

torch.random.manual_seed(0)


def get_configs():
    block_N = [64, 128]
    block_H = [64]
    num_split = [1, 2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{"block_N": c[0], "block_H": c[1], "num_split": c[2], "num_stages": c[3], "threads": c[4]} for c in _configs]
    return configs


@lru_cache(maxsize=1)
def get_heuristic_config() -> Tuple[Dict, int]:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version == 89:
        cfg = dict(block_N=128, block_H=64, num_split=1, num_stages=0, threads=128)
    else:
        cfg = dict(block_N=128, block_H=64, num_split=8, num_stages=2, threads=128)
    return cfg, sm_version


# TODO(lei): fix warp specialized and tma lower pass
def get_pass_configs():
    return {tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[6], pass_configs=get_pass_configs())
def flashattn(batch, heads, groups, seqlen_kv, dim, block_N, block_H, num_split, num_stages, threads):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // groups

    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    valid_block_N = min(block_N, seqlen_kv // num_split)

    @T.prim_func
    def flashattn_gqa_decode_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        # split
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            mask_local = T.alloc_fragment([block_N], "uint8")
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                        :,
                    ],
                    K_shared,
                )
                T.copy(
                    mask[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                    ],
                    mask_local,
                )
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else((mask_local[j] != 0) & (j < seqlen_kv // num_split), acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(
                    V[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                        :,
                    ],
                    V_shared,
                )
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            for i in T.Parallel(block_H):
                if i < valid_block_H:
                    glse[bid, hid * valid_block_H + i, sid] = logsum[i]
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :])

        # combine
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local = T.alloc_fragment([num_split, 128], dtype)
            lse_logsum_local = T.alloc_fragment([128], accum_dtype)
            lse_max_local = T.alloc_fragment([128], accum_dtype)
            scale_local = T.alloc_fragment([128], accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            for k, j in T.Parallel(num_split, 128):
                lse_local[k, j] = glse[bz, by, k]
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
            for k in T.serial(num_split):
                for j in T.Parallel(128):
                    lse_logsum_local[j] += T.exp2(lse_local[k, j] - lse_max_local[j])
            for j in T.Parallel(128):
                lse_logsum_local[j] = T.log2(lse_logsum_local[j]) + lse_max_local[j]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                for j in T.Parallel(128):
                    scale_local[j] = T.exp2(lse_local[k, j] - lse_logsum_local[j])
                # Note: Pay attention to dim and the number of threads in Parallel
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[i]
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def flashattn_gqa_decode_no_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], "uint8"),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            mask_local = T.alloc_fragment([block_N], "uint8")
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.copy(mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    if num_split > 1:
        return flashattn_gqa_decode_split
    else:
        return flashattn_gqa_decode_no_split


def ref_program(query, key, value, mask, glse, Output_partial):
    #     """
    #     Inputs:
    #     - query (Tensor): [batch, heads, dim]
    #     - key (Tensor): [batch, seqlen_kv, groups, dim]
    #     - value (Tensor): [batch, seqlen_kv, groups, dim]
    #     - mask (Tensor): [batch, seqlen_kv, groups]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]
    value = rearrange(value, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]

    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    scores = einsum(query, key, "b g h d, b h s d -> b g h s")  # [batch_size, num_head_groups, groups, seqlen_kv]
    if mask is not None:
        mask = rearrange(mask, "b s h -> b h s")
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention = F.softmax(scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, value, "b g h s, b h s d -> b g h d")  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
    return out


def flash_split_ref(Q, K, V, mask):
    num_split = 16
    batch = Q.size(0)
    nheads = Q.size(1)
    groups = K.size(2)
    dim = Q.size(-1)
    block_N = 32
    seqlen_kv = K.size(1)
    num_head_groups = nheads // groups

    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, num_head_groups, groups, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, num_head_groups, groups, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, num_head_groups, groups, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads), device="cuda", dtype=torch.float)

    Q_ = Q * scale
    Q_ = rearrange(Q_, "b (h g) d -> b g h d", g=num_head_groups)

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float("-inf"))
        scores_max_prev.fill_(float("-inf"))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum(
                "bghd,bkhd->bghk",
                Q_,
                K[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )  # [batch, nheads, block_N]
            if mask is not None:
                mask_local = mask[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :]
                mask_local = rearrange(mask_local, "b s h -> b h s")
                mask_local = mask_local.unsqueeze(1)
                acc_s = acc_s.masked_fill(mask_local == 0, float("-inf"))
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
            scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
            acc_o *= scores_scale[:, :, :, None]
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)  # [batch, nheads, block_N]
            acc_o += torch.einsum(
                "bghk,bkhd->bghd",
                acc_s_cast,
                V[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o_out = rearrange(acc_o, "b g h d->b (h g) d")
        logsum_out = rearrange(logsum, "b g h->b (h g)")
        acc_o_out /= logsum_out[:, :, None]
        logsum_out = torch.log2(logsum_out) + rearrange(scores_max, "b g h->b (h g)")
        gacc_o[ks, :, :, :] = acc_o_out
        glogsum[ks, :, :] = logsum_out

    return glogsum.to(torch.float16).permute(1, 2, 0), gacc_o.to(torch.float16).permute(1, 2, 0, 3)


def reduce_ref(Q, K, V, mask, glse, Output_partial):
    num_split = 16
    o = torch.empty_like(Output_partial[:, :, 0, :]).fill_(0)
    lse_logsum = torch.empty_like(glse[:, :, 0]).fill_(0)  # [batch, heads]
    lse_max = glse.max(dim=2, keepdim=False).values
    for ks in range(num_split):
        lse = glse[:, :, ks]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glse[:, :, ks]
        scale = torch.exp2(lse - lse_logsum)  # [batch, heads]
        o += Output_partial[:, :, ks, :] * scale[:, :, None]
    return o.to(torch.float16)


def ref_split_program(Q, K, V, mask, glse=None, Output_partial=None):
    glse_, Output_partial_ = flash_split_ref(Q, K, V, mask)
    return reduce_ref(Q, K, V, mask, glse_, Output_partial_)


def print_red_warning(msg):
    print(f"\033[91m{msg}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f"{name} all zero")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-2, name="tensor", assert_=False, print_=True):
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        if assert_:
            raise AssertionError(f"{name} Error: {diff}")
    else:
        if print_:
            print(f"passed: {name} diff={diff}")


def main(batch: int = 1, heads: int = 32, groups: int = 8, kv_seqlen: int = 8192, dim: int = 128, tune: bool = False):
    batch, heads, groups, kv_seqlen, dim = batch, heads, groups, kv_seqlen, dim
    qk_flops = 2 * batch * heads * kv_seqlen * dim
    pv_flops = 2 * batch * heads * kv_seqlen * dim
    total_flops = qk_flops + pv_flops

    if not tune:
        config, sm_version = get_heuristic_config()
        kernel = flashattn(batch, heads, groups, kv_seqlen, dim, **config)
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        mask = torch.randint(0, 2, (batch, kv_seqlen, groups), device="cuda", dtype=torch.uint8)
        split = config["num_split"]
        glse = torch.empty(batch, heads, split, device="cuda", dtype=torch.float16)
        Output_partial = torch.empty(batch, heads, split, dim, device="cuda", dtype=torch.float16)
        o = kernel(q, k, v, mask, glse, Output_partial)
        o_ref = ref_program(q, k, v, mask, glse, Output_partial)
        o_ref_split = ref_split_program(q, k, v, mask, glse, Output_partial)

        print(o)
        print(o_ref)

        assert_similar(o, o_ref, name="o_ref")
        assert_similar(o, o_ref_split, name="o_ref_split")

        print("All checks pass.")
        latency = profiler.do_bench(ref_program, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        kernel = flashattn(batch, heads, groups, kv_seqlen, dim)
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


def run_regression_perf(batch: int = 1, heads: int = 32, groups: int = 8, kv_seqlen: int = 8192, dim: int = 128):
    batch, heads, groups, kv_seqlen, dim = batch, heads, groups, kv_seqlen, dim
    config, _ = get_heuristic_config()
    kernel = flashattn(batch, heads, groups, kv_seqlen, dim, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(warmup=500, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--groups", type=int, default=8, help="groups")
    parser.add_argument("--kv_seqlen", type=int, default=8192, help="kv sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.kv_seqlen, args.dim, args.tune)
