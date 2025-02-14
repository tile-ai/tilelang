import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T

num_split = 4


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, block_N, block_H):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_kv = [batch, seqlen_kv, kv_head_num, dim]
    part_shape = [batch, heads, num_split, dim]
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.macro
    def flash_attn_split(
            Q: T.Buffer(shape_q, dtype),
            K: T.Buffer(shape_kv, dtype),
            V: T.Buffer(shape_kv, dtype),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),
    ):
        with T.Kernel(
                batch, heads // min(block_H, kv_group_num), num_split, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // block_H)

            T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(
                    K[bid, (seqlen_kv // num_split) * sid +
                      k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                      cur_kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
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
                      k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :],
                    V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum, glse[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, sid])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H,
                                            sid, :])

    @T.macro
    def combine(
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),
            Output: T.Buffer(shape_q, dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local = T.alloc_fragment([num_split, 1], dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_fragment([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout({
                lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
            })

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            for k in T.Parallel(num_split):
                lse_local[k, 0] = glse[bz, by, k]
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
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
            K: T.Buffer(shape_kv, dtype),
            V: T.Buffer(shape_kv, dtype),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer(part_shape, dtype),  # [batch, heads, num_split, dim]
            Output: T.Buffer(shape_q, dtype),
    ):
        flash_attn_split(Q, K, V, glse, Output_partial)
        combine(glse, Output_partial, Output)

    return main


def ref_program(query, key, value, glse, Output_partial):
    #     """
    #     Inputs:
    #     - query (Tensor): [batch, heads, dim]
    #     - key (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - value (Tensor): [batch, seqlen_kv, kv_head_num, dim]

    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    from einops import rearrange
    batch_size, query_heads, dim = query.shape  # [batch_size, query_heads, dim]
    _, seqlen_kv, kv_heads, _ = key.shape  # [batch_size, seqlen_kv, kv_heads, kv_dim]
    assert kv_heads == 1, "kv_heads must be 1"

    query_expanded = rearrange(query, 'b h d -> b h 1 d')  # [batch_size, query_heads, 1, dim]
    key_expanded = key.expand(-1, -1, query_heads, -1)  # [batch_size, query_heads, seqlen_kv, dim]
    value_expanded = value.expand(-1, -1, query_heads,
                                  -1)  # [batch_size, query_heads, seqlen_kv, dim]
    key_expanded = rearrange(key_expanded,
                             'b n h d -> b h n d')  # [batch_size, kv_head_num, seqlen_kv, dim]
    value_expanded = rearrange(value_expanded,
                               'b n h d -> b h n d')  # [batch_size, query_heads, seqlen_kv, dim]

    scores = torch.matmul(query_expanded,
                          key_expanded.transpose(-1, -2))  # [batch_size, query_heads, 1, seqlen_kv]
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    attention_weights = F.softmax(scores, dim=-1)  # [batch_size, query_heads, 1, seqlen_kv]
    output = torch.matmul(attention_weights, value_expanded)  # [batch_size, query_heads, 1, dim]
    return output.view(batch_size, query_heads, dim)


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
    BATCH, H, KV_H, Q_CTX, KV_CTX, D_HEAD = 1, 128, 1, 128, 128 * 1024, 128
    flops_per_matmul = 2.0 * BATCH * H * Q_CTX * KV_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    BLOCK_N = 64  # if D_HEAD <= 128 else 32
    BLOCK_H = 128

    program = flashattn(BATCH, H, KV_H, KV_CTX, D_HEAD, BLOCK_N, BLOCK_H)
    mod, params = tilelang.lower(program)
    mod = tilelang.Profiler(mod, params, [5], tilelang.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks passed!")
