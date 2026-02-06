import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang as tl
import tilelang.language as T
from typing import Optional
from index import prepare_token_indices

from utils import get_abs_err, get_err_ratio

BF16 = T.bfloat16
FP32 = T.float32
INT32 = T.int32

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tl.jit(pass_configs=pass_configs)
def tl_indexer_topk_reducesum_impl(
    heads: int,
    dim: int,
    topk: int,
    sm_scale: Optional[float] = None,
    block_K: int = 32,
    dtype: str = FP32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0
    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    index_q_shape = [seq_len, heads, dim]
    weights_shape = [seq_len, heads]
    index_k_shape = [seq_len, dim]
    topk_indices_shape = [seq_len, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    if sm_scale is None:
        sm_scale = dim**-0.5

    @T.macro
    def bitonic_sort(
        topk_index_shared: T.SharedBuffer([N], dtype=INT32),
        topk_value_shared: T.SharedBuffer([N], dtype=FP32),
    ):
        T.sync_threads()
        for i1 in T.serial(num_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and (
                        (ascending and topk_value_shared[i] > topk_value_shared[j])
                        or (not ascending and topk_value_shared[i] < topk_value_shared[j])
                    ):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def tl_indexer_topk_reducesum_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        Weights: T.Tensor(weights_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        ReduceSum: T.Tensor(topk_indices_shape, FP32),
        Offsets: T.Tensor(offsets_shape, INT32),
        TokenIndices: T.Tensor(token_indices_shape, INT32),
    ):
        with T.Kernel(seq_len, threads=num_threads) as (bx):
            i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
            bos, eos = Offsets[i_b], Offsets[i_b + 1]
            num_blocks = T.ceildiv(i_t + 1, block_K)

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)

            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
            T.sync_threads()

            weights_frag = T.alloc_shared([heads], dtype=dtype)
            T.copy(Weights[bos + i_t, :], weights_frag)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_st = bk_i * block_K
                k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                for i, j in T.Parallel(block_K, dim):
                    index_k_shared[i, j] = T.if_then_else(k_st + i < k_ed, IndexK[bos + k_st + i, j], 0)
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                T.sync_threads()

                logits_sum = T.alloc_fragment(block_K, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.sync_threads()

                offset = T.alloc_var(INT32)
                if k_st >= topk:
                    offset = topk + (k_st % topk)
                else:
                    offset = k_st
                T.sync_threads()
                for i in T.Parallel(block_K):
                    if k_st + i > i_t:
                        logits_sum[i] = float("-inf")
                    j = offset + i
                    topk_index_shared[j] = k_st + i
                    topk_value_shared[j] = logits_sum[i]
                T.sync_threads()

                if k_ed > topk and k_ed % topk == 0:
                    bitonic_sort(topk_index_shared, topk_value_shared)

            bitonic_sort(topk_index_shared, topk_value_shared)

            logits_max_frag = T.alloc_fragment([1], dtype=FP32)
            logits_frag = T.alloc_fragment([topk], dtype=FP32)
            reducesum_shared = T.alloc_shared([topk], dtype=FP32)

            T.copy(topk_value_shared[:topk], logits_frag)
            T.sync_threads()

            T.reduce_max(logits_frag, logits_max_frag, dim=-1)
            T.sync_threads()

            for i in T.Parallel(topk):
                logits_frag[i] = T.exp(logits_frag[i] - logits_max_frag[0])
            T.sync_threads()

            lse_frag = T.alloc_fragment([1], dtype=FP32)
            T.reduce_sum(logits_frag, lse_frag)
            T.sync_threads()

            for i in T.Parallel(topk):
                reducesum_shared[i] = logits_frag[i] / lse_frag[0]
            T.sync_threads()

            # for i in T.Parallel(topk):
            #     reducesum_shared[i] = logits_frag[i]
            # T.sync_threads()

            for i in T.Parallel(topk):
                if topk_index_shared[i] > i_t:
                    topk_index_shared[i] = -1
            T.sync_threads()

            T.copy(topk_index_shared[:topk], TopkIndices[bos + i_t, :])
            T.copy(reducesum_shared[:topk], ReduceSum[bos + i_t, :])

    return tl_indexer_topk_reducesum_kernel


def indexer_topk_reducesum_interface(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    offsets: torch.Tensor,
    dtype: str = BF16,
):
    seq_len, heads, dim = q.shape
    kernel = tl_indexer_topk_reducesum_impl(heads=heads, dim=dim, topk=topk, dtype=dtype)
    token_indices = prepare_token_indices(offsets)
    topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
    topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
    kernel(q, weights, k, topk_indices, topk_score, offsets, token_indices)
    return topk_indices, topk_score


def ref_index_score(Q: torch.Tensor, Weights: torch.Tensor, K: torch.Tensor, topk: int, offsets: torch.Tensor) -> torch.Tensor:
    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= topk
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(q, k, "s1 h k, s2 k -> s1 h s2")
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return topk_indices, topk_score


def test_kernel(
    B=1,
    S=2048,
    H=64,
    D=128,
    topk=64,
):
    torch.manual_seed(42)

    q = torch.randn((S, H, D)).cuda().bfloat16()
    weights = torch.randn((S, H)).cuda().bfloat16()
    k = torch.randn((S, D)).cuda().bfloat16()
    offsets = torch.tensor([0, S], dtype=torch.int32).cuda()

    ref_topk_indices, ref_topk_score = ref_index_score(q, weights, k, topk, offsets)

    topk_indices, topk_score = indexer_topk_reducesum_interface(q, weights, k, topk, offsets)

    for j in range(S):
        ref_np = ref_topk_indices[j].cpu().to(torch.int32).numpy()
        trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        ref_np_val = ref_topk_score[j]
        trt_np_val = topk_score[j]

        mask = (ref_np_val > 0).cpu().numpy()

        set_ref = set(ref_np[mask])
        set_trt = set(trt_np[mask])
        intersection = set_ref & set_trt

        print("idx:", j, "selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

        print(f"err: {get_abs_err(ref_np_val, trt_np_val):.6f} ratio: {get_err_ratio(ref_np_val, trt_np_val):.6f}")


if __name__ == "__main__":
    test_kernel()
