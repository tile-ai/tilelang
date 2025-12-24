import torch

num_split = 1


def flash_split_ref(Q, Q_pe, KV, K_pe):
    dim = Q.shape[-1]
    pe_dim = Q_pe.shape[-1]
    batch = Q.size(0)
    nheads = Q.size(1)
    block_N = 64
    seqlen_kv = KV.size(1)

    scale = (1.0 / (dim + pe_dim)) ** 0.5 * 1.44269504  # log2(e)
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
    Q_pe_ = Q_pe * scale
    KV_ = KV.expand(-1, -1, nheads, -1)
    K_pe_ = K_pe.expand(-1, -1, nheads, -1)

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float("-inf"))
        scores_max_prev.fill_(float("-inf"))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum(
                "bhd,bkhd->bhk",
                Q_,
                KV_[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )  # [batch, nheads, block_N]
            acc_s += torch.einsum(
                "bhd,bkhd->bhk",
                Q_pe_,
                K_pe_[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
            scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
            acc_o *= scores_scale[:, :, None]
            acc_s = torch.exp2(acc_s - scores_max[:, :, None])
            acc_s_cast = acc_s.to(torch.float16)  # [batch, nheads, block_N]
            acc_o += torch.einsum(
                "bhk,bkhd->bhd",
                acc_s_cast,
                KV_[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o /= logsum[:, :, None]
        logsum = torch.log2(logsum) + scores_max
        gacc_o[ks, :, :, :] = acc_o
        glogsum[ks, :, :] = logsum

    return glogsum.to(torch.float16).permute(1, 2, 0), gacc_o.to(torch.float16).permute(1, 2, 0, 3)


def reduce_ref(Q, Q_pe, KV, K_pe, glse, Output_partial):
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
