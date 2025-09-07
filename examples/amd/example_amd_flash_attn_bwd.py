import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.primitives.gemm.base import GemmWarpPolicy
import itertools
import argparse
from functools import partial

def ref_program(Q, K, V, is_causal, groups=1):
    assert Q.size(
        2) == K.size(2) * groups, f"Q heads {Q.size(2)} K heads {K.size(2)} groups {groups}"
    assert Q.size(
        2) == V.size(2) * groups, f"Q heads {Q.size(2)} V heads {V.size(2)} groups {groups}"
    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def get_fwd_configs():
    block_M = [32, 64, 128, 256]
    block_N = [32, 64, 128, 256]
    threads = [128, 256, 512]
    num_split_q = [64, 128, 256]
    num_stages = [0,1]
    enable_rasterization = [True]
    k_pack = [2]
    panel_size = [7, 8, 9, 10]
    qk_coalesced_width = [8]
    v_coalesced_width = [4]

    valid_configs = []

    for m, n, s, t, stages, r, k, p, qkw, vw in itertools.product(block_M, block_N, num_split_q,
                                                                  threads, num_stages,
                                                                  enable_rasterization, k_pack,
                                                                  panel_size, qk_coalesced_width,
                                                                  v_coalesced_width):
        valid_configs.append({
            "block_M": m,
            "block_N": n,
            "num_split_q": s,
            "threads": t,
            "num_stages": stages,
            "enable_rasterization": r,
            "k_pack": k,
            "panel_size": p,
            "qk_coalesced_width": qkw,
            "v_coalesced_width": vw,
        })
    return valid_configs


@tilelang.autotune(configs=get_fwd_configs(), cache_input_tensors=True)
@tilelang.jit(out_idx=[3, 4])
def fast_flashattn(
    batch, heads, seq_len, dim, is_causal, groups,
    block_M: int, block_N: int, num_split_q: int, threads: int, num_stages: int,
    enable_rasterization: bool, k_pack: int, panel_size: int,
    qk_coalesced_width: int, v_coalesced_width: int,
):
    scale = (1.0 / dim)**0.5 * 1.44269504
    log2_e = 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    vec_size = qk_coalesced_width
    v_vec_size = v_coalesced_width

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype), K: T.Tensor(kv_shape, dtype), V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(panel_size, enable=enable_rasterization)
            bz = byz_combined // heads
            by = byz_combined % heads
            num_q_blocks = T.ceildiv(seq_len, block_M)
            bx = T.alloc_var("int32")
            bx = b_split
            with T.While(bx < num_q_blocks):
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                l_i = T.alloc_fragment([block_M], accum_dtype)
                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)
                current_bx = bx
                q_block_offset = current_bx * block_M
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                m_prev = T.alloc_fragment([block_M], accum_dtype)
                scale_factor = T.alloc_fragment([block_M], accum_dtype)
                T.copy(Q[bz, q_block_offset:q_block_offset + block_M, by, :], Q_shared, coalesced_width=vec_size)
                loop_end_k = T.ceildiv(q_block_offset + block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
                row_sum = T.alloc_fragment([block_M], accum_dtype)
                for k in T.Pipelined(loop_end_k, num_stages=num_stages):
                    kv_idx = k * block_N
                    T.copy(K[bz, kv_idx:kv_idx + block_N, by // groups, :], K_shared, coalesced_width=vec_size)
                    T.copy(V[bz, kv_idx:kv_idx + block_N, by // groups, :], V_shared, coalesced_width=v_vec_size)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(q_block_offset + i >= kv_idx + j, 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        sf = T.exp2(m_prev[i] * scale - m_i[i] * scale)
                        l_i[i] *= sf
                        scale_factor[i] = sf
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scale_factor[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_i[i] * scale)
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        l_i[i] += row_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)
                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l
                for i, j in T.Parallel(block_M, dim):
                    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]
                
                for i in T.Parallel(block_M):
                    if q_block_offset + i < seq_len:
                        m_natural = m_i[i] / log2_e
                        lse_val = m_natural + T.log(l_i[i])
                        LSE[bz, by, q_block_offset + i] = lse_val

                bx = current_bx + num_split_q
    return main


def get_bwd_configs():
    block_M = [16, 32, 64, 128, 256]
    block_N = [16, 32, 64, 128, 256]
    threads = [64, 128, 256, 512]
    num_stages = [0, 1]
    configs = []
    for m, n, stages, t in itertools.product(block_M, block_N, num_stages, threads):
        configs.append({"block_M": m, "block_N": n, "num_stages": stages, "threads": t})
    return configs


@tilelang.jit(out_idx=[2])
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    blk = 32
    @T.prim_func
    def flash_bwd_prep(O: T.Tensor(shape, dtype), dO: T.Tensor(shape, dtype), Delta: T.Tensor([batch, heads, seq_len], accum_dtype)):
        with T.Kernel(batch, heads, T.ceildiv(seq_len, blk)) as (bz, bx, by):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk): acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])
    return flash_bwd_prep


@tilelang.autotune(configs=get_bwd_configs(), cache_input_tensors=True)
@tilelang.jit(out_idx=[6, 7, 8])
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups,
    block_M: int, block_N: int, num_stages: int, threads: int):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504 # log2(e)
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    @T.prim_func
    def flash_bwd_main(Q: T.Tensor([batch, seq_len, heads, dim], dtype), 
                       K: T.Tensor([batch, seq_len, head_kv, dim], dtype), 
                       V: T.Tensor([batch, seq_len, head_kv, dim], dtype), 
                       dO: T.Tensor([batch, seq_len, heads, dim], dtype), 
                       lse: T.Tensor([batch, heads, seq_len], accum_dtype), 
                       Delta: T.Tensor([batch, heads, seq_len], accum_dtype), 
                       dQ: T.Tensor([batch, seq_len, heads, dim], dtype), 
                       dK: T.Tensor([batch, seq_len, head_kv, dim], dtype), 
                       dV: T.Tensor([batch, seq_len, head_kv, dim], dtype)):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            # Shared memory buffers
            K_shared = T.alloc_shared([block_M, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            q_shared = T.alloc_shared([block_N, dim], dtype)
            do_shared = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_shared = T.alloc_shared([block_N], accum_dtype)
            ds_shared = T.alloc_shared([block_M, block_N], dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            # Register fragments
            p_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dP = T.alloc_fragment([block_M, block_N], accum_dtype)
            dS_acc = T.alloc_fragment([block_M, block_N], accum_dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)

            T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q_shared)
                
                T.clear(qkT)
                T.gemm(K_shared, q_shared, qkT, transpose_B=True, policy=GemmWarpPolicy.FullRow)

                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                
                # We need a separate fragment for P because qkT will be reused for dP
                P_acc = T.alloc_fragment([block_M, block_N], accum_dtype)
                for i, j in T.Parallel(block_M, block_N):
                    P_acc[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        P_acc[i, j] = T.if_then_else(by * block_M + i >= k * block_N + j, P_acc[i, j], 0)
                
                T.copy(P_acc, p_cast)
                
                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do_shared)
                T.gemm(p_cast, do_shared, dv, policy=GemmWarpPolicy.FullRow)
                
                T.clear(dP)
                T.gemm(V_shared, do_shared, dP, transpose_B=True, policy=GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta_shared)
                
                for i, j in T.Parallel(block_M, block_N):
                    dS_acc[i, j] = p_cast[i, j] * (dP[i, j] - delta_shared[j]) * sm_scale
                
                # --- START OF FIX: Use Shared Memory to Reset Layout ---
                # 1. "Wash" the layout by copying dS to shared memory.
                T.copy(dS_acc, ds_shared)
                
                # 2. Use the "clean" shared memory as input for the transposed GEMM to calculate dQ.
                #    The compiler can now handle the transpose efficiently from the simple shared memory layout.
                T.clear(dq)
                T.gemm(ds_shared, K_shared, dq, transpose_A=True, policy=GemmWarpPolicy.FullRow)

                # 3. For dK, we still need a fragment. We can load it back from shared memory.
                #    This ensures its layout is fresh and not tied to the dS_acc calculation.
                ds_cast_for_dk = T.alloc_fragment([block_M, block_N], dtype)
                T.copy(ds_shared, ds_cast_for_dk)
                T.gemm(ds_cast_for_dk, q_shared, dk, policy=GemmWarpPolicy.FullRow)
                # --- END OF FIX ---x j

                # 直接写入dQ，避免使用atomic_add
                # 注意：这假设每个Q块只被一个线程块处理
                for i, j in T.Parallel(block_N, dim):
                    if k * block_N + i < seq_len:
                        dQ[bz, k * block_N + i, bx, j] = dq[i, j]

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx // groups, :])
            T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx // groups, :])
    return flash_bwd_main


def main(batch: int = 1, heads: int = 8, seq_len: int = 4096, dim: int = 128, is_causal: bool = False, groups: int = 1):
    device = "cuda"
    dtype = torch.float16
    
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    dO = torch.randn_like(q)

    print("Starting autotuning for FlashAttention-V2 Forward Pass...")
    fwd_kernel = fast_flashattn(batch, heads, seq_len, dim, is_causal, groups)

    if fwd_kernel is None or fwd_kernel.config is None:
        print("Forward pass auto-tuning failed. No valid kernel configuration was found.")
        return
    print(f"Autotuning finished. Best Forward Configuration: {fwd_kernel.config}")
    o_tl, lse_tl = fwd_kernel(q, k, v)

    bwd_prep = flashattn_bwd_preprocess(
        batch, heads, seq_len, dim,
    )
    delta_tl = bwd_prep(o_tl, dO)

    print("\nStarting autotuning for FlashAttention-V2 Backward Pass...")
    bwd_kernel = flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups)
    if bwd_kernel is None or bwd_kernel.config is None:
        print("Backward pass auto-tuning failed.")
        return
    print(f"Autotuning finished. Best Backward Configuration: {bwd_kernel.config}")
    
    dQ_tl = torch.empty_like(q)
    dK_tl = torch.empty_like(k)
    dV_tl = torch.empty_like(v)
    bwd_kernel(q, k, v, dO, lse_tl, delta_tl, dQ_tl, dK_tl, dV_tl)

    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()
    
    o_ref, lse_ref = ref_program(q_ref, k_ref, v_ref, is_causal, groups)
    o_ref.backward(dO)
    
    print("\nVerifying correctness...")
    torch.testing.assert_close(o_tl, o_ref, rtol=0.05, atol=0.05, msg="Forward output mismatch")
    print("Forward pass is correct.")
    torch.testing.assert_close(lse_tl, lse_ref, rtol=0.02, atol=0.02, msg="LSE mismatch")
    print("LSE is correct.")
    torch.testing.assert_close(dQ_tl, q_ref.grad, rtol=0.05, atol=0.05, msg="dQ mismatch")
    print("dQ is correct.")
    torch.testing.assert_close(dK_tl, k_ref.grad, rtol=0.05, atol=0.05, msg="dK mismatch")
    print("dK is correct.")
    torch.testing.assert_close(dV_tl, v_ref.grad, rtol=0.05, atol=0.05, msg="dV mismatch")
    print("dV is correct.")
    print("All checks pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=8, help='heads')
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--groups', type=int, default=1, help='groups')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups)