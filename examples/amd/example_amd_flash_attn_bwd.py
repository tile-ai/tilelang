import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.primitives.gemm.base import GemmWarpPolicy
import itertools
import argparse
from functools import partial
import numpy as np
import time

# --- Reference Program (从原始的fwd+bwd代码中保留，因为它返回LSE) ---
def ref_program(Q, K, V, is_causal, groups=1):
    assert Q.size(
        2) == K.size(2) * groups, f"Q heads {Q.size(2)} K heads {K.size(2)} groups {groups}"
    assert Q.size(
        2) == V.size(2) * groups, f"Q heads {Q.size(2)} V heads {V.size(2)} groups {groups}"
    dim = Q.size(-1)
    K_ref = K.repeat_interleave(groups, dim=2)
    V_ref = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K_ref)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V_ref)
    lse = torch.logsumexp(scores, dim=-1).float() # 这是返回LSE的关键
    return output, lse


# --- 新的前向传播配置 (来自你提供的新fwd代码，并重命名为get_fwd_configs) ---
def get_fwd_configs():
    """Generates configurations for the autotuner, tailored for FA-2 style parallelism."""
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


# --- 修改后的fast_flashattn，同时输出LSE ---
@tilelang.autotune(configs=get_fwd_configs(), cache_input_tensors=True) # 使用新的配置
@tilelang.jit(out_idx=[3, 4]) # 现在输出两个张量：Output 和 LSE
def fast_flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M: int,
    block_N: int,
    num_split_q: int,
    threads: int,
    num_stages: int,
    enable_rasterization: bool,
    k_pack: int,
    panel_size: int,
    qk_coalesced_width: int,
    v_coalesced_width: int,
):
    scale = (1.0 / dim)**0.5
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    vec_size = qk_coalesced_width
    v_vec_size = v_coalesced_width

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, heads, seq_len], accum_dtype), # 添加LSE输出
    ):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(panel_size, enable=enable_rasterization)

            bz = byz_combined // heads
            by = byz_combined % heads

            num_q_blocks = T.ceildiv(seq_len, block_M)

            # 将原始的bx重命名为bx_loop_var，以避免与T.Kernel的bx参数混淆
            bx_loop_var = T.alloc_var("int32")
            bx_loop_var = b_split

            with T.While(bx_loop_var < num_q_blocks):
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype) # current max
                l_i = T.alloc_fragment([block_M], accum_dtype) # current logsum_exp

                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)

                current_bx = bx_loop_var
                q_block_offset = current_bx * block_M

                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)

                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                m_prev = T.alloc_fragment([block_M], accum_dtype)
                scale_factor = T.alloc_fragment([block_M], accum_dtype)

                T.copy(
                    Q[bz, q_block_offset:q_block_offset + block_M, by, :],
                    Q_shared,
                    coalesced_width=vec_size)

                loop_end_k = (T.ceildiv(q_block_offset + block_M, block_N)
                              if is_causal else T.ceildiv(seq_len, block_N))

                row_sum = T.alloc_fragment([block_M], accum_dtype)

                for k in T.Pipelined(loop_end_k, num_stages=num_stages):
                    kv_idx = k * block_N

                    T.copy(
                        K[bz, kv_idx:kv_idx + block_N, by // groups, :],
                        K_shared,
                        coalesced_width=vec_size)
                    T.copy(
                        V[bz, kv_idx:kv_idx + block_N, by // groups, :],
                        V_shared,
                        coalesced_width=v_vec_size)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(q_block_offset + i >= kv_idx + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        k_pack=k_pack,
                        policy=GemmWarpPolicy.FullRow,
                    )

                    # 应用缩放因子 (为exp2计算准备)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = acc_s[i, j] * scale # 这里的scores已经包含了scale

                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False) # m_i 是缩放后scores的最大值

                    for i in T.Parallel(block_M):
                        if m_prev[i] == -T.infinity(accum_dtype):
                            scale_factor[i] = 0.0
                        else:
                            scale_factor[i] = T.exp(m_prev[i] - m_i[i]) # 注意这里m_prev和m_i已经是缩放后的值

                        l_i[i] *= scale_factor[i] # L_i 乘以之前的最大值差的指数

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scale_factor[i]

                    for i, j in T.Parallel(block_M, block_N):
                        if acc_s[i, j] == -T.infinity(acc_s.dtype): # 处理因mask产生的-inf
                            acc_s[i, j] = 0.0
                        else:
                            acc_s[i, j] = T.exp(acc_s[i, j] - m_i[i]) # scores 减去当前最大值

                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        l_i[i] += row_sum[i]

                    # 将 acc_s (accum_dtype) 转换为 dtype 并直接与 V 进行 GEMM
                    T.copy(acc_s, acc_s_cast)

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)

                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l

                for i, j in T.Parallel(block_M, dim):
                    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]

                # --- 计算并存储LSE ---
                for i in T.Parallel(block_M):
                    if q_block_offset + i < seq_len:
                        lse_val = T.if_then_else(
                            l_i[i] > 0,
                            T.log(l_i[i]) + m_i[i], # LSE = log(sum(exp(scaled_scores - max_scaled_scores))) + max_scaled_scores
                            -T.infinity(accum_dtype)
                        )
                        LSE[bz, by, q_block_offset + i] = lse_val

                bx_loop_var = current_bx + num_split_q # 更新循环变量

    return main

# --- 反向传播函数 (从原始fwd+bwd代码中保留) ---

def get_bwd_configs():
    """
    使用排列组合生成反向传播的自动调优配置，扩大搜索范围
    """
    block_M = [16, 32, 64, 128, 256]
    block_N = [16, 32, 64, 128, 256]
    threads = [64, 128, 256, 512, 1024]
    num_stages = [0, 1, 2]

    configs = []
    for m, n, stages, t in itertools.product(block_M, block_N, num_stages, threads):
        configs.append({
            "block_M": m,
            "block_N": n,
            "num_stages": stages,
            "threads": t
        })

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
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])
    return flash_bwd_prep


@tilelang.autotune(configs=get_bwd_configs(), cache_input_tensors=True)
@tilelang.jit
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups,
                  block_M: int, block_N: int, num_stages: int, threads: int):
    sm_scale = (1.0 / dim)**0.5
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_bwd_kernel(
            Q: T.Tensor(q_shape, dtype), K: T.Tensor(kv_shape, dtype), V: T.Tensor(kv_shape, dtype),
            dO: T.Tensor(q_shape, dtype), lse: T.Tensor([batch, heads, seq_len], accum_dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
            dQ: T.Tensor(q_shape, accum_dtype), dK: T.Tensor(kv_shape, accum_dtype), dV: T.Tensor(kv_shape, accum_dtype)):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            q_shared = T.alloc_shared([block_N, dim], dtype)
            do_shared = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_shared = T.alloc_shared([block_N], accum_dtype)
            # Additional shared memory buffer for transpose operations
            ds_shared = T.alloc_shared([block_M, block_N], dtype)

            # Workspace buffers
            p_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_acc = T.alloc_fragment([block_M, block_N], accum_dtype)
            dP = T.alloc_fragment([block_M, block_N], accum_dtype)

            # Accumulators
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)

            T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            # Determine loop range
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q_shared)
                T.clear(qkT)

                # Compute QK^T (不在这里应用scaling)
                T.gemm(K_shared, q_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)

                # Compute softmax weights (保持与前向传播一致)
                for i, j in T.Parallel(block_M, block_N):
                    P_acc[i, j] = T.exp(qkT[i, j] * sm_scale - lse_shared[j])

                # Apply causal mask (使用正确的条件逻辑)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        P_acc[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, P_acc[i, j], 0.0)

                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do_shared)
                T.clear(dP)

                # 按照标准实现的确切顺序：先计算dsT = V^T * dO
                T.gemm(V_shared, do_shared, dP, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # 然后计算dV += P^T * dO
                T.copy(P_acc, p_cast)
                T.gemm(p_cast, do_shared, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta_shared)

                # 计算dsT_cast = P * (dsT - Delta) * sm_scale (对应标准实现)
                for i, j in T.Parallel(block_M, block_N):
                    p_cast[i, j] = P_acc[i, j] * (dP[i, j] - delta_shared[j]) * sm_scale

                # Compute dK += dS Q
                T.gemm(p_cast, q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

                # Copy dS to shared memory, then compute dQ += dS^T K
                T.copy(p_cast, ds_shared)
                T.clear(dq)
                T.gemm(ds_shared, K_shared, dq, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # Accumulate to global memory
                for i, j in T.Parallel(block_N, dim):
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            # Write dK and dV using fp32 atomic_add (按照标准实现，不加seq_len检查)
            for i, j in T.Parallel(block_M, dim):
                T.atomic_add(dV[bz, by * block_M + i, bx // groups, j], dv[i, j])
                T.atomic_add(dK[bz, by * block_M + i, bx // groups, j], dk[i, j])
    return flash_bwd_kernel


@tilelang.jit(out_idx=[1])
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    blk = 64
    @T.prim_func
    def flash_bwd_post(
            dQ_in: T.Tensor(shape, accum_dtype), dQ_out: T.Tensor(shape, dtype)):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.copy(
                dQ_in[bz, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )
    return flash_bwd_post


# --- 调试和基准测试函数 (从原始fwd+bwd代码中保留) ---

def debug_tensor_comparison(tensor1, tensor2, name, rtol=1e-3, atol=1e-3):
    """Compare two tensors and output detailed debugging information"""
    print(f"\n=== {name} Comparison ===")
    print(f"Shape: {tensor1.shape} vs {tensor2.shape}")
    print(f"Data type: {tensor1.dtype} vs {tensor2.dtype}")
    print(f"Device: {tensor1.device} vs {tensor2.device}")

    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Difference std: {std_diff:.6f}")

    # Find the position of maximum difference
    if max_diff > atol:
        max_idx = torch.argmax(diff)
        max_idx = np.unravel_index(max_idx.cpu().numpy(), tensor1.shape)
        print(f"Max difference position: {max_idx}")
        print(f"Value1: {tensor1[max_idx].item():.6f}, Value2: {tensor2[max_idx].item():.6f}")

    # Check for NaN and Inf
    nan_count1 = torch.isnan(tensor1).sum().item()
    nan_count2 = torch.isnan(tensor2).sum().item()
    inf_count1 = torch.isinf(tensor1).sum().item()
    inf_count2 = torch.isinf(tensor2).sum().item()

    print(f"NaN count: {nan_count1} vs {nan_count2}")
    print(f"Inf count: {inf_count1} vs {inf_count2}")

    # Calculate relative differences
    relative_diff = diff / (torch.abs(tensor2) + 1e-8)
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()

    print(f"Max relative difference: {max_relative_diff:.6f}")
    print(f"Mean relative difference: {mean_relative_diff:.6f}")

    # Check if within tolerance
    close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    print(f"Within tolerance (rtol={rtol}, atol={atol}): {close}")

    return close, max_diff, mean_diff


def benchmark_function(func, *args, warmup=10, repeat=100):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure time
    times = []
    for _ in range(repeat):
        start = time.time()
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return np.median(times)


# --- 主函数 (已修改为使用 fast_flashattn) ---
def main(batch: int = 1,
         heads: int = 8,
         seq_len: int = 4096,
         dim: int = 128,
         is_causal: bool = False,
         groups: int = 1):

    device = "cuda"
    dtype = torch.float16

    # 设置随机种子以保证可重现性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"Test configuration: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, is_causal={is_causal}, groups={groups}")

    # 计算完整的正向+反向传播的FLOPs
    flops_per_gemm = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 5 * flops_per_gemm  # 完整的正向+反向传播
    # if is_causal:
    #     total_flops *= 0.5  # 因果注意力只需要计算一半的注意力权重

    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFlops")

    # 创建输入张量
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    dO = torch.randn_like(q)

    print("Starting autotuning for Fast FlashAttention-V2 Forward Pass...")
    # 使用新的fast_flashattn kernel
    fwd_kernel = fast_flashattn(batch, heads, seq_len, dim, is_causal, groups)
    if fwd_kernel is None or fwd_kernel.config is None:
        print("Forward pass auto-tuning failed.")
        return
    print(f"Autotuning finished. Best Forward Configuration: {fwd_kernel.config}")

    # 为参考实现创建偏函数
    ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups)

    # 获取 profiler
    # profiler.assert_allclose 期望编译后的kernel返回 (Output, LSE)，这与我们修改后的fast_flashattn一致
    profiler = fwd_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    print("Verifying correctness...")
    # 验证前向传播的正确性
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print("Forward pass is correct.")

    # 运行前向传播以获取反向验证所需的输出
    o_tl, lse_tl = fwd_kernel(q, k, v) # 现在fast_flashattn返回output和lse

    # 准备反向传播
    bwd_prep = flashattn_bwd_preprocess(batch, heads, seq_len, dim)
    delta_tl = bwd_prep(o_tl, dO)

    print("\n开始FlashAttention-V2反向传播自动调优...")
    bwd_kernel = flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups)
    if bwd_kernel is None or bwd_kernel.config is None:
        print("反向传播自动调优失败。")
        return
    print(f"自动调优完成。最佳反向传播配置: {bwd_kernel.config}")

    # 初始化梯度累加器
    dQ_accum = torch.zeros_like(q, dtype=torch.float32)
    dK_tl = torch.zeros_like(k, dtype=torch.float32)
    dV_tl = torch.zeros_like(v, dtype=torch.float32)

    # 运行反向传播
    bwd_kernel(q, k, v, dO, lse_tl, delta_tl, dQ_accum, dK_tl, dV_tl)

    # 后处理 dQ
    post_kernel = flashattn_bwd_postprocess(batch, heads, seq_len, dim)
    dQ_tl = post_kernel(dQ_accum)

    # 用于反向传播验证的参考实现
    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()

    o_ref, _ = ref_program(q_ref, k_ref, v_ref, is_causal, groups)
    o_ref.backward(dO)

    # 验证反向传播的正确性
    print("Verifying backward pass correctness...")
    dq_close, dq_max_diff, dq_mean_diff = debug_tensor_comparison(
        dQ_tl, q_ref.grad, "dQ", rtol=0.05, atol=0.05
    )
    if dq_close:
        print("dQ is correct.")
    else:
        print("dQ mismatch detected.")

    dk_close, dk_max_diff, dk_mean_diff = debug_tensor_comparison(
        dK_tl.to(torch.float16), k_ref.grad, "dK", rtol=0.05, atol=0.05
    )
    if dk_close:
        print("dK is correct.")
    else:
        print("dK mismatch detected.")

    dv_close, dv_max_diff, dv_mean_diff = debug_tensor_comparison(
        dV_tl.to(torch.float16), v_ref.grad, "dV", rtol=0.05, atol=0.05
    )
    if dv_close:
        print("dV is correct.")
    else:
        print("dV mismatch detected.")

    # 性能基准测试
    print("\n=== Performance Benchmarking ===")

    # 测试参考实现性能 (完整的正向+反向)
    def run_reference_fwd_bwd():
        q_ref_bench = q.clone().detach().requires_grad_()
        k_ref_bench = k.clone().detach().requires_grad_()
        v_ref_bench = v.clone().detach().requires_grad_()

        # 前向传播
        o_ref_bench, _ = ref_program(q_ref_bench, k_ref_bench, v_ref_bench, is_causal, groups)

        # 反向传播
        o_ref_bench.backward(dO)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ref_latency = benchmark_function(run_reference_fwd_bwd, warmup=10, repeat=100)
    print(f"Reference PyTorch Forward+Backward: {ref_latency:.2f} ms | {total_flops / ref_latency * 1e-9:.2f} TFlops")

    # 测试Tile-lang完整的前向+反向实现性能
    def run_complete_fwd_bwd():
        # 前向传播
        o_tl_bench, lse_tl_bench = fwd_kernel(q, k, v) # 使用新的fwd_kernel

        # 反向传播预处理
        delta_tl_bench = bwd_prep(o_tl_bench, dO)

        # 完整的反向传播
        dQ_bench = torch.zeros_like(q, dtype=torch.float32)
        dK_bench = torch.zeros_like(k, dtype=torch.float32)
        dV_bench = torch.zeros_like(v, dtype=torch.float32)
        bwd_kernel(q, k, v, dO, lse_tl_bench, delta_tl_bench, dQ_bench, dK_bench, dV_bench)

        # 后处理 dQ
        post_kernel(dQ_bench)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    tile_latency = benchmark_function(run_complete_fwd_bwd, warmup=10, repeat=100)
    print(f"Complete Flash Attention V2 Forward+Backward (Tile-lang): {tile_latency:.2f} ms | {total_flops / tile_latency * 1e-9:.2f} TFlops")

    # 计算加速比
    speedup = ref_latency / tile_latency
    print(f"Speedup: {speedup:.2f}x")

    # 结果总结
    print("\n=== Verification Results Summary ===")
    print(f"Forward output: Passed")
    print(f"dQ: {'Passed' if dq_close else 'Failed'} (Max diff: {dq_max_diff:.6f})")
    print(f"dK: {'Passed' if dk_close else 'Failed'} (Max diff: {dk_max_diff:.6f})")
    print(f"dV: {'Passed' if dv_close else 'Failed'} (Max diff: {dv_max_diff:.6f})")

    if all([dq_close, dk_close, dv_close]):
        print("All checks passed!")
    else:
        print("Some checks failed, may need further debugging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=8, help='heads')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--dim', type=int, default=64, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--groups', type=int, default=1, help='groups')
    args = parser.parse_args()

    # 使用较小的默认值进行测试
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups)