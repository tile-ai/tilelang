# TileLang FlashAttention Tutorial

## 1. Forward Kernel
> **Note**: This is the first half of the complete FlashAttention tutorial, focusing on the forward pass implementation. The backward pass tutorial will be at the second half.

This section explains the FlashAttention forward implementation in TileLang, a parallel programming language for high-performance GPU computing. The kernel achieves **>1.3x speedup** over FlashAttention 2, reaching **630 TFLOPS/s** for 4K sequence lengths on NVIDIA H100 GPUs. Key optimizations include:
- **Tiled computation** with configurable block sizes
- **Pipelined memory operations** to hide latency
- **Numerically stable softmax** using base-2 exponentials
- **Autotuning support** for optimal performance

## Complete Kernel Code
```python
@autotune(configs=get_configs())
@jit(out_idx=[-2, -1])
def mha_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    num_stages=1,
    threads=256,
):
    sm_scale = (1.0 / dim)**0.5 * 1.44269504

    q_shape = [batch, heads, seq_len, dim]
    k_shape = [batch, heads, seq_len_kv, dim]
    v_shape = [batch, heads, seq_len_kv, dim]
    o_shape = [batch, heads, seq_len, dim]
    lse_shape = [batch, heads, seq_len]
    dtype = "bfloat16"
    accum_dtype = "float"
    q_start_id = seq_len_kv - seq_len

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            # Memory allocations
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

            # Load Q tile
            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -2**30)

            # Determine loop range with causal masking
            loop_range = (
                T.min(T.ceildiv(seq_len_kv, block_N),
                T.ceildiv(seq_len_kv - seq_len + (bx + 1) * block_M, block_N)
            ) if is_causal else T.ceildiv(seq_len_kv, block_N)

            # Main attention loop
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Load K tile
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
                
                # Initialize scores with causal masking
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i + q_start_id
                    k_idx = k * block_N + j
                    acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                
                # Compute attention scores
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Softmax preparation
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * sm_scale - scores_max[i] * sm_scale)
                
                # Compute exp scores
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * sm_scale - scores_max[i] * sm_scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                # Update running statistics
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                
                # Accumulate output
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # Final normalization
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])
            
            # Save logsumexp
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * sm_scale
            T.copy(logsum, Lse[bz, by, bx * block_M:(bx + 1) * block_M])

    return main
```

## Key Components Explained

### Kernel Configuration

```python
@autotune(configs=get_configs())
@jit(out_idx=[-2, -1])
```

- Autotuning: Dynamically selects optimal tile sizes (`block_M`, `block_N`) and thread configuration
- Output Indexing: `out_idx=[-2,-1]` specifies tensor contraction dimensions

### Parameters

```python
sm_scale = (1.0 / dim)**0.5 * 1.44269504
```

- Precomputed Scaling: Combines normalization factor (√dk) and log₂(e) constant for softmax

### Memory Management

```python
Q_shared = T.alloc_shared([block_M, dim], dtype)
K_shared = T.alloc_shared([block_N, dim], dtype)
```

- Tiled Loading: Copies blocks of Q/K/V into shared memory
- Efficient Access: Optimized memory layout for high-performance computation

### Causal Masking

```python
acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
```

- Block-Level Masking: Computes valid attention ranges per tile
- Position Tracking: Uses `q_start_id` to align sequences in encoder-decoder setups

### Attention Computation

```python
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

- Matrix Multiplication: Computes attention scores using optimized GEMM operations

### Numerically Stable Softmax

```python
# Compute max per row
T.reduce_max(acc_s, scores_max, dim=1, clear=False)

# Rescale previous statistics
scores_scale[i] = T.exp2(scores_max_prev[i]*sm_scale - scores_max[i]*sm_scale)

# Compute exp scores
acc_s[i, j] = T.exp2(acc_s[i, j]*sm_scale - scores_max[i]*sm_scale)

# Update running sum
logsum[i] = logsum[i]*scores_scale[i] + scores_sum[i]
```

- Base-2 Exponentiation: Leverages hardware-optimized exp2 instructions
- Online Rescaling: Maintains numerical stability across blocks

### Output Accumulation

```python
T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]
```

- Fused GEMM: Multiplies softmax output with value matrix
- Delayed Normalization: Applies softmax scaling after full accumulation

### Pipelining

```python
for k in T.Pipelined(loop_range, num_stages=num_stages)
```

- Overlapped Execution: Hides memory latency by processing multiple blocks concurrently
- Configurable Stages: `num_stages` controls pipeline depth for different hardware

### Log-Sum-Exp Output

```python
logsum[i] = T.log2(logsum[i]) + scores_max[i]*sm_scale
T.copy(logsum, Lse)
```

- Final Logarithm: Converts running sum to log-space
- Gradient Preparation: Stores values needed for backward pass

## Performance Notes

1. **Block Sizing**: Optimal `block_M`/`block_N` vary with sequence length (128 works well for 4K)

2. **Thread Count**: 256 threads balance occupancy and resource usage

3. **Mixed Precision**: `bfloat16` for storage with `float` accumulation maintains precision

4. **Causal Adaptation**: Loop range adjustment minimizes unnecessary computation
