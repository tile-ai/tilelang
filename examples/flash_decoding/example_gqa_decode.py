import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum

num_split = 8


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, block_N, block_H):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, kv_head_num, dim]
    shape_v = [batch, seqlen_kv, kv_head_num, dim]
    shape_o = [batch, heads, dim]
    part_shape = [batch, heads, num_split, dim]
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    valid_block_H = min(block_H, kv_group_num)

    @T.macro
    def flash_attn_split(
            Q: T.Buffer(shape_q, dtype),  # [batch, heads, dim]
            K: T.Buffer(shape_k, dtype),
            V: T.Buffer(shape_v, dtype),
            mask: T.Buffer([batch, seqlen_kv, kv_head_num], "uint8"),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=128) as (bx, by, bz):
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

            # T.annotate_layout({
            #     O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            # })

            T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(
                    K[bid, (seqlen_kv // num_split) * sid +
                      k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                      cur_kv_head, :], K_shared)
                T.copy(
                    mask[bid, (seqlen_kv // num_split) * sid +
                         k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                         cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                 -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
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
                    V[bid, (seqlen_kv // num_split) * sid +
                      k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                      cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum[:valid_block_H], glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid])
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                            sid, :])

    @T.macro
    def combine(
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),  # [batch, heads, num_split, dim]
            Output: T.Buffer(shape_o, dtype),  # [batch, heads, dim]
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local = T.alloc_fragment([num_split, 128], dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_fragment([128], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout({
                lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                lse_max_local: T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                lse_local: T.Fragment(lse_local.shape, forward_thread_fn=lambda i, j: j),
            })

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            for k in T.Parallel(num_split):
                lse_local[k, 0] = glse[bz, by, k]
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, by, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split[0] = glse[bz, by, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def main(
            Q: T.Buffer(shape_q, dtype),
            K: T.Buffer(shape_k, dtype),
            V: T.Buffer(shape_v, dtype),
            mask: T.Buffer([batch, seqlen_kv, kv_head_num], "uint8"),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),  # [batch, heads, num_split, dim]
            Output: T.Buffer(shape_o, dtype),
    ):
        flash_attn_split(Q, K, V, mask, glse, Output_partial)
        combine(glse, Output_partial, Output)

    return main


def ref_program(query, key, value, mask, glse, Output_partial):
    #     """
    #     Inputs:
    #     - query (Tensor): [batch, heads, dim]
    #     - key (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - value (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - mask (Tensor): [batch, seqlen_kv, kv_head_num]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, kv_head_num, seqlen_kv, dim]
    value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, kv_head_num, seqlen_kv, dim]

    query = rearrange(
        query, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, kv_head_num, dim]

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, kv_head_num, seqlen_kv]
    if mask is not None:
        mask = rearrange(mask, 'b s h -> b h s')
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, kv_head_num, seqlen_kv]

    out = einsum(attention, value,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, kv_head_num, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out


def flash_split_ref(Q, K, V, mask):
    batch = Q.size(0)
    nheads = Q.size(1)
    kv_head_num = K.size(2)
    dim = Q.size(-1)
    block_N = 32
    seqlen_kv = K.size(1)
    num_head_groups = nheads // kv_head_num

    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, num_head_groups, kv_head_num, block_N),
                        device="cuda",
                        dtype=torch.float)
    acc_s_cast = torch.empty((batch, num_head_groups, kv_head_num, block_N),
                             device="cuda",
                             dtype=torch.float16)
    acc_o = torch.empty((batch, num_head_groups, kv_head_num, dim),
                        device="cuda",
                        dtype=torch.float)
    scores_max = torch.empty((batch, num_head_groups, kv_head_num),
                             device="cuda",
                             dtype=torch.float)
    scores_max_prev = torch.empty((batch, num_head_groups, kv_head_num),
                                  device="cuda",
                                  dtype=torch.float)
    scores_scale = torch.empty((batch, num_head_groups, kv_head_num),
                               device="cuda",
                               dtype=torch.float)
    scores_sum = torch.empty((batch, num_head_groups, kv_head_num),
                             device="cuda",
                             dtype=torch.float)
    logsum = torch.empty((batch, num_head_groups, kv_head_num), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads), device="cuda", dtype=torch.float)

    Q_ = Q * scale
    Q_ = rearrange(Q_, 'b (h g) d -> b g h d', g=num_head_groups)

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float('-inf'))
        scores_max_prev.fill_(float('-inf'))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum('bghd,bkhd->bghk', Q_,
                                 K[:, (seqlen_kv // num_split) * ks +
                                   i * block_N:(seqlen_kv // num_split) * ks +
                                   (i + 1) * block_N, :, :])  # [batch, nheads, block_N]
            if mask is not None:
                mask_local = mask[:, (seqlen_kv // num_split) * ks +
                                  i * block_N:(seqlen_kv // num_split) * ks + (i + 1) * block_N, :]
                mask_local = rearrange(mask_local, 'b s h -> b h s')
                mask_local = mask_local.unsqueeze(1)
                acc_s = acc_s.masked_fill(mask_local == 0, float('-inf'))
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
            scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
            acc_o *= scores_scale[:, :, :, None]
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)  # [batch, nheads, block_N]
            acc_o += torch.einsum(
                'bghk,bkhd->bghd', acc_s_cast,
                V[:, (seqlen_kv // num_split) * ks + i * block_N:(seqlen_kv // num_split) * ks +
                  (i + 1) * block_N, :, :])
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o_out = rearrange(acc_o, 'b g h d->b (h g) d')
        logsum_out = rearrange(logsum, 'b g h->b (h g)')
        acc_o_out /= logsum_out[:, :, None]
        logsum_out = torch.log2(logsum_out) + rearrange(scores_max, 'b g h->b (h g)')
        gacc_o[ks, :, :, :] = acc_o_out
        glogsum[ks, :, :] = logsum_out

    return glogsum.to(torch.float16).permute(1, 2, 0), gacc_o.to(torch.float16).permute(1, 2, 0, 3)


def reduce_ref(Q, K, V, glse, Output_partial):
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


if __name__ == "__main__":
    # BATCH, H_Q, KV_H, KV_CTX, D_HEAD = 64, 256, 4, 8192, 128
    # BATCH, H_Q, KV_H, KV_CTX, D_HEAD = 1, 16, 1, 64, 128
    BATCH, H_Q, KV_H, KV_CTX, D_HEAD = 1, 32, 8, 8192, 128
    qk_flops = 2 * BATCH * H_Q * KV_CTX * D_HEAD
    pv_flops = 2 * BATCH * H_Q * KV_CTX * D_HEAD
    total_flops = qk_flops + pv_flops
    BLOCK_N = 128  # if D_HEAD <= 128 else 32
    BLOCK_H = 64

    program = flashattn(BATCH, H_Q, KV_H, KV_CTX, D_HEAD, BLOCK_N, BLOCK_H)
    mod, params = tilelang.lower(program)
    mod = tilelang.Profiler(mod, params, [6], tilelang.TensorSupplyType.Auto)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks passed!")
    latency = mod.do_bench(mod.func, warmup=500, profiler="auto")
    print("Tile-lang: {:.4f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
