import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum

num_split = 1


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.macro
    def flash_attn_split(
            Q: T.Buffer([batch, heads, dim], dtype),
            Q_pe: T.Buffer([batch, heads, pe_dim], dtype),
            KV: T.Buffer([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Buffer([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer([batch, heads, num_split, dim], dtype),
    ):
        with T.Kernel(
                batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_0 = T.alloc_fragment([block_H, block_N], accum_dtype)
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

            T.annotate_layout({
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })

            T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = (seqlen_kv // num_split) * sid + k * block_N
                kv_end = (seqlen_kv // num_split) * sid + (k + 1) * block_N
                
                T.copy(
                    KV[bid, kv_start:kv_end, cur_kv_head, :], 
                    KV_shared
                )
                T.copy(
                    K_pe[bid, kv_start:kv_end, cur_kv_head, :], 
                    K_pe_shared
                )
                
                T.clear(acc_s_0)
                T.gemm(Q_shared, KV_shared, acc_s_0, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s_0, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(acc_s_0, S_shared)
                T.copy(S_shared, acc_s)
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
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
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
            Output_partial: T.Buffer([batch, heads, num_split, dim], dtype),
            Output: T.Buffer([batch, heads, dim], dtype),
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
            Q: T.Buffer([batch, heads, dim], dtype),
            Q_pe: T.Buffer([batch, heads, pe_dim], dtype), 
            KV: T.Buffer([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Buffer([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Buffer([batch, heads, num_split], dtype),
            Output_partial: T.Buffer([batch, heads, num_split, dim], dtype),
            Output: T.Buffer([batch, heads, dim], dtype),
    ):
        flash_attn_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
        combine(glse, Output_partial, Output)

    return main



def ref_program(q, q_pe, kv, k_pe, glse, Output_partial):
    #     """
    #     Inputs:
    #     - q (Tensor): [batch, heads, dim]
    #     - q_pe (Tensor): [batch, heads, pe_dim]
    #     - kv (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - k_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
    #     - glse (Tensor): [batch, heads, num_split]
    #     - Output_partial (Tensor): [batch, heads, num_split, dim]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim) ** 0.5
    q = rearrange(
        q, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    q_pe = rearrange(
        q_pe, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

    kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    k_pe = rearrange(k_pe, 'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, kv,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out


def flash_split_ref(Q, K, V):
    dim = 512
    pe_dim = 64
    batch = Q.size(0)
    nheads = Q.size(1)
    assert Q.size(2) == dim + pe_dim, "dim must be 576=512+64"
    block_N = 32
    seqlen_kv = K.size(1)

    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, nheads, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, nheads, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, nheads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, nheads), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, nheads), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, nheads), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, nheads), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, nheads), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads), device="cuda", dtype=torch.float)

    Q_ = Q * scale
    K_ = K.expand(-1, -1, nheads, -1)
    V_ = V.expand(-1, -1, nheads, -1)

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float('-inf'))
        scores_max_prev.fill_(float('-inf'))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum('bhd,bkhd->bhk', Q_,
                                 K_[:, (seqlen_kv // num_split) * ks +
                                    i * block_N:(seqlen_kv // num_split) * ks +
                                    (i + 1) * block_N, :, :])  # [batch, nheads, block_N]
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
            scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
            acc_o *= scores_scale[:, :, None]
            acc_s = torch.exp2(acc_s - scores_max[:, :, None])
            acc_s_cast = acc_s.to(torch.float16)  # [batch, nheads, block_N]
            acc_o += torch.einsum(
                'bhk,bkhd->bhd', acc_s_cast,
                V_[:, (seqlen_kv // num_split) * ks + i * block_N:(seqlen_kv // num_split) * ks +
                   (i + 1) * block_N, :, :])
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o /= logsum[:, :, None]
        logsum = torch.log2(logsum) + scores_max
        gacc_o[ks, :, :, :] = acc_o
        glogsum[ks, :, :] = logsum

    return glogsum.to(torch.float16).permute(1, 2, 0), gacc_o.to(torch.float16).permute(1, 2, 0, 3)


def reduce_ref(Q, K, V, glse, Output_partial):
    o = torch.empty_like(Output_partial[:, :, 0, :]).fill_(0)
    lse_logsum = torch.empty_like(glse[:, :, 0]).fill_(0)
    lse_max = glse.max(dim=2, keepdim=False).values
    for ks in range(num_split):
        lse = glse[:, :, ks]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glse[:, :, ks]
        scale = torch.exp2(lse - lse_logsum)
        o += Output_partial[:, :, ks, :] * scale[:, :, None]
    return o.to(torch.float16)


if __name__ == "__main__":
    BATCH, H_Q, KV_H, KV_CTX, D_HEAD, DPE = 128, 128, 1, 8192, 512, 64
    qk_flops = 2 * BATCH * H_Q * KV_CTX * (D_HEAD + DPE)
    pv_flops = 2 * BATCH * H_Q * KV_CTX * D_HEAD
    total_flops = qk_flops + pv_flops
    BLOCK_N = 32  # if D_HEAD <= 128 else 32
    BLOCK_H = 64

    program = flashattn(BATCH, H_Q, KV_H, KV_CTX, D_HEAD, DPE, BLOCK_N, BLOCK_H)
    mod, params = tilelang.lower(program)
    mod = tilelang.Profiler(mod, params, [6], tilelang.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All close")
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))