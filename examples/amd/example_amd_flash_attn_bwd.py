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
    lse = torch.logsumexp(scores, dim=-1).float()
    return output, lse


def get_fwd_configs():
    block_M = [64, 128]
    block_N = [32, 64]
    threads = [128, 256]
    num_stages = [0, 1]
    configs = []
    for m, n, stages, t in itertools.product(block_M, block_N, num_stages, threads):
        configs.append({
            "block_M": m, 
            "block_N": n, 
            "num_stages": stages, 
            "threads": t
        })
    return configs


@tilelang.autotune(configs=get_fwd_configs(), cache_input_tensors=True)
@tilelang.jit(out_idx=[3, 4])
def flashattn_fwd(
    batch, heads, seq_len, dim, is_causal, groups,
    block_M: int, block_N: int, num_stages: int, threads: int,
):
    scale = (1.0 / dim)**0.5
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype), K: T.Tensor(kv_shape, dtype), V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.ceildiv(
                    (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
            
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_idx = k * block_N
                T.copy(K[bz, kv_idx:kv_idx + block_N, by // groups, :], K_shared)
                
                # Initialize acc_s, handle causal mask
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        row_idx = bx * block_M + i
                        col_idx = k * block_N + j
                        if row_idx >= col_idx:
                            acc_s[i, j] = 0.0
                        else:
                            acc_s[i, j] = -T.infinity(acc_s.dtype)
                else:
                    T.fill(acc_s, 0.0)
                
                # Compute QK^T
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # Apply scaling factor
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * scale
                
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                
                # Compute scaling factor (using more stable exp calculation)
                for i in T.Parallel(block_M):
                    if scores_max_prev[i] == -T.infinity(accum_dtype):
                        scores_scale[i] = 0.0
                    else:
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                
                # Scale acc_o
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                
                # Compute softmax weights
                for i, j in T.Parallel(block_M, block_N):
                    if acc_s[i, j] == -T.infinity(acc_s.dtype):
                        acc_s[i, j] = 0.0
                    else:
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                # Update logsum
                for i in T.Parallel(block_M):
                    if scores_max_prev[i] == -T.infinity(accum_dtype):
                        # First iteration
                        logsum[i] = scores_sum[i]
                    else:
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # Normalize output and write result
            for i, j in T.Parallel(block_M, dim):
                if logsum[i] > 0:
                    acc_o[i, j] /= logsum[i]
                else:
                    acc_o[i, j] = 0.0
                    
            T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            
            # Compute LSE
            for i in T.Parallel(block_M):
                if bx * block_M + i < seq_len:
                    lse_val = T.if_then_else(
                        logsum[i] > 0,
                        T.log(logsum[i]) + scores_max[i],
                        -T.infinity(dtype)
                    )
                    LSE[bz, by, bx * block_M + i] = lse_val
    return main


def get_bwd_configs():
    """
    使用排列组合生成反向传播的自动调优配置，不进行过滤
    """
    block_M = [16, 32, 64, 128, 256]
    block_N = [16, 32, 64, 128, 256]
    threads = [64, 128, 256, 512]
    num_stages = [0, 1]
    
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
            dQ: T.Tensor(q_shape, accum_dtype), dK: T.Tensor(kv_shape, dtype), dV: T.Tensor(kv_shape, dtype)):
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
                
                # Compute QK^T
                T.gemm(K_shared, q_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # Apply scaling factor
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = qkT[i, j] * sm_scale
                
                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                
                # Compute softmax weights
                for i, j in T.Parallel(block_M, block_N):
                    P_acc[i, j] = T.exp(qkT[i, j] - lse_shared[j])
                
                # Apply causal mask
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        if by * block_M + i < k * block_N + j:
                            P_acc[i, j] = 0.0
                
                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do_shared)
                T.clear(dP)
                
                # Compute dP = V^T dO
                T.gemm(V_shared, do_shared, dP, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # Compute dV += P^T dO
                T.copy(P_acc, p_cast)
                T.gemm(p_cast, do_shared, dv, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta_shared)
                
                # Compute dS = P * (dP - Delta)
                for i, j in T.Parallel(block_M, block_N):
                    p_cast[i, j] = p_cast[i, j] * (dP[i, j] - delta_shared[j]) * sm_scale
                
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

            # Write dK and dV
            T.copy(dv, dV[bz, by * block_M:(by + 1) * block_M, bx // groups, :])
            T.copy(dk, dK[bz, by * block_M:(by + 1) * block_M, bx // groups, :])
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


def main(batch: int = 1,
         heads: int = 8,
         seq_len: int = 4096,
         dim: int = 128,
         is_causal: bool = False,
         groups: int = 1):
    
    device = "cuda"
    dtype = torch.float16
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    print(f"Test configuration: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, is_causal={is_causal}, groups={groups}")
    
    # Calculate FLOPs
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul  # QK^T and PV matrix multiplications
    if is_causal:
        total_flops *= 0.5  # Causal attention only needs to compute half the attention weights
    
    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFlops")
    
    # Create input tensors
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    dO = torch.randn_like(q)

    print("Starting autotuning for FlashAttention-V2 Forward Pass...")
    fwd_kernel = flashattn_fwd(batch, heads, seq_len, dim, is_causal, groups)
    if fwd_kernel is None or fwd_kernel.config is None:
        print("Forward pass auto-tuning failed.")
        return
    print(f"Autotuning finished. Best Forward Configuration: {fwd_kernel.config}")

    # Create partial function for reference implementation
    ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups)

    # Get profiler
    profiler = fwd_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    print("Verifying correctness...")
    # Verify forward pass correctness
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print("Forward pass is correct.")

    # Run forward pass to get outputs for backward verification
    o_tl, lse_tl = fwd_kernel(q, k, v)

    # Prepare backward pass
    bwd_prep = flashattn_bwd_preprocess(batch, heads, seq_len, dim)
    delta_tl = bwd_prep(o_tl, dO)

    print("\n开始FlashAttention-V2反向传播自动调优...")
    bwd_kernel = flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups)
    if bwd_kernel is None or bwd_kernel.config is None:
        print("反向传播自动调优失败。")
        return
    print(f"自动调优完成。最佳反向传播配置: {bwd_kernel.config}")
    
    # Initialize gradient accumulators
    dQ_accum = torch.zeros_like(q, dtype=torch.float32)
    dK_tl = torch.empty_like(k)
    dV_tl = torch.empty_like(v)
    
    # Run backward pass
    bwd_kernel(q, k, v, dO, lse_tl, delta_tl, dQ_accum, dK_tl, dV_tl)

    # Post-process dQ
    post_kernel = flashattn_bwd_postprocess(batch, heads, seq_len, dim)
    dQ_tl = post_kernel(dQ_accum)

    # Reference implementation
    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()
    
    o_ref, lse_ref = ref_program(q_ref, k_ref, v_ref, is_causal, groups)
    o_ref.backward(dO)
    
    # Verify backward pass correctness
    print("Verifying backward pass correctness...")
    dq_close, dq_max_diff, dq_mean_diff = debug_tensor_comparison(
        dQ_tl, q_ref.grad, "dQ", rtol=0.05, atol=0.05
    )
    if dq_close:
        print("dQ is correct.")
    else:
        print("dQ mismatch detected.")
    
    dk_close, dk_max_diff, dk_mean_diff = debug_tensor_comparison(
        dK_tl, k_ref.grad, "dK", rtol=0.05, atol=0.05
    )
    if dk_close:
        print("dK is correct.")
    else:
        print("dK mismatch detected.")
    
    dv_close, dv_max_diff, dv_mean_diff = debug_tensor_comparison(
        dV_tl, v_ref.grad, "dV", rtol=0.05, atol=0.05
    )
    if dv_close:
        print("dV is correct.")
    else:
        print("dV mismatch detected.")
    
    # Performance benchmarking
    print("\n=== Performance Benchmarking ===")
    
    # Test reference implementation performance
    ref_latency = benchmark_function(ref_program_processed, q, k, v, warmup=10, repeat=100)
    print(f"Reference (PyTorch): {ref_latency:.2f} ms | {total_flops / ref_latency * 1e-9:.2f} TFlops")
    
    # Test Tile-lang implementation performance
    def run_fwd_kernel():
        fwd_kernel(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    tile_latency = benchmark_function(run_fwd_kernel, warmup=10, repeat=100)
    print(f"Fast Flash Attention V2 (Tile-lang): {tile_latency:.2f} ms | {total_flops / tile_latency * 1e-9:.2f} TFlops")
    
    # Calculate speedup
    speedup = ref_latency / tile_latency
    print(f"Speedup: {speedup:.2f}x")
    
    # Summary of results
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
    
    # Use smaller default values for testing
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups)