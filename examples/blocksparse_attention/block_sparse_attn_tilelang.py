# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch

import tilelang
import tilelang.language as T
import torch.nn.functional as F


def get_sparse_attn_mask_from_topk(x, topk, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    # N_CTX = downsample_len * BLOCK
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len],
                            False,
                            dtype=torch.bool,
                            device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


def blocksparse_flashattn(batch, heads, seq_len, dim, downsample_len, is_causal):
    block_M = 64
    block_N = 64
    num_stages = 0
    threads = 128
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    block_mask_shape = [batch, heads, downsample_len, downsample_len]

    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "int8"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Buffer(shape, dtype),
            Q_shared: T.Buffer([block_M, dim], dtype),
            K_shared: T.Buffer([block_N, dim], dtype),
            acc_s: T.Buffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
                V: T.Buffer(shape, dtype),
                V_shared: T.Buffer([block_M, dim], dtype),
                acc_s_cast: T.Buffer([block_M, block_N], dtype),
                acc_o: T.Buffer([block_M, dim], accum_dtype),
                k: T.int32,
                by: T.int32,
                bz: T.int32,
        ):
            T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.Buffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.Buffer([block_M, block_N], dtype),
                scores_max: T.Buffer([block_M], accum_dtype),
                scores_max_prev: T.Buffer([block_M], accum_dtype),
                scores_scale: T.Buffer([block_M], accum_dtype),
                scores_sum: T.Buffer([block_M], accum_dtype),
                logsum: T.Buffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.Buffer([block_M, dim], accum_dtype),
                scores_scale: T.Buffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Buffer(shape, dtype),
                K: T.Buffer(shape, dtype),
                V: T.Buffer(shape, dtype),
                BlockSparseMask: T.Buffer(block_mask_shape, block_mask_dtype),
                Output: T.Buffer(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_mask = T.alloc_local([downsample_len], block_mask_dtype)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for vj in T.serial(downsample_len):
                    block_mask[vj] = BlockSparseMask[bz, by, bx, vj]

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if block_mask[k] != 0:
                        MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                        Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                                scores_sum, logsum)
                        Rescale(acc_o, scores_scale)
                        MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return main

    return kernel_func(block_M, block_N, num_stages, threads)


def test_topk_sparse_attention():
    # Config
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 1, 1, 256, 64
    TOPK = 2  # Keep top 8 elements per row
    BLOCK = 64
    torch.manual_seed(0)

    # Create inputs
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)

    sm_scale = 1.0 / (D_HEAD**0.5)

    # Create sparse mask (downsampled to block level)
    downsample_factor = BLOCK
    downsample_len = math.ceil(SEQ_LEN / downsample_factor)
    x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len],
                       device='cuda',
                       dtype=torch.bfloat16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    # Run Triton kernel
    program = blocksparse_flashattn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, downsample_len, is_causal=True)
    kernel = tilelang.compile(program, out_idx=[4])
    print(kernel.get_kernel_source())
    tilelang_output = kernel(q, k, v, block_mask)

    # Compute reference
    # Expand block mask to full attention matrix
    full_mask = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device='cuda'))
    full_mask = full_mask[..., :SEQ_LEN, :SEQ_LEN].bool()
    full_mask = full_mask & torch.tril(torch.ones_like(full_mask))  # Apply causal

    # PyTorch reference implementation
    attn = torch.einsum('bhsd,bhtd->bhst', q, k) * sm_scale
    attn = attn.masked_fill(~full_mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    ref_output = torch.einsum('bhst,bhtd->bhsd', attn, v)

    print("ref_output", ref_output)
    print("tilelang_output", tilelang_output)

    # Verify accuracy
    assert torch.allclose(tilelang_output, ref_output, atol=1e-2, rtol=1e-2), \
        "TileLang output doesn't match reference"
    print("Pass topk sparse attention test with qlen == klen")


if __name__ == "__main__":
    test_topk_sparse_attention()
