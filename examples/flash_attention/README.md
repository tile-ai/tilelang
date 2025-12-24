# FlashAttention

Using tile-lang, we can define buffers at different memory layers. For instance, `Q_shared`, `K_shared`, and `V_shared` can be defined in shared memory, while `acc_s` and `acc_o` can be placed in registers. This flexibility allows us to represent a complex fusion pattern like FlashAttention in a simple way.

```python
@T.prim_func
def flash_attention(
    Q: T.Tensor(shape, dtype),
    K: T.Tensor(shape, dtype),
    V: T.Tensor(shape, dtype),
    Output: T.Tensor(shape, dtype),
):
    # Launch a specialized T.Kernel with 3D mapping: (bx, by, bz)
    #   bx: block index in sequence dimension
    #   by: block index in "heads" dimension
    #   bz: block index in "batch" dimension
    # threads=thread_num means how many threads per block
    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
        # Allocate shared memory for Q, K, V to reduce global memory accesses
        Q_shared = T.alloc_shared([block_M, dim], dtype)
        K_shared = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_N, dim], dtype)
        # Allocate buffers on register
        # acc_s: buffer to hold intermediate attention scores
        acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
        # acc_s_cast: buffer for storing casted/adjusted scores
        acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
        # acc_o: partial accumulation of output
        acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
        # Buffers to track per-row maximum score and related stats
        scores_max = T.alloc_fragment([block_M], accum_dtype)
        scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
        scores_scale = T.alloc_fragment([block_M], accum_dtype)
        scores_sum = T.alloc_fragment([block_M], accum_dtype)
        logsum = T.alloc_fragment([block_M], accum_dtype)


        # Copy a block of Q from global memory to Q_shared
        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

        # Initialize accumulators
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        loop_range = (
            T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
        )

        # Pipeline the loop to overlap copies/gemm stages
        for k in T.Pipelined(loop_range, num_stages=num_stages):
            # Copy K block into shared memory
            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)

            # Perform the Q*K^T multiplication, Here, transpose_B=True indicates that K_shared is transposed,
            # policy=T.GemmWarpPolicy.FullRow means each warp is responsible for computing an entire row
            # of acc_s, and the resulting acc_s is retained in registers.
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            # Copy V block into shared memory
            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
            for i, j in T.Parallel(block_M, dim):
                acc_s[i, j] *= scale

            # Save old scores_max, then reset scores_max
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Compute the maximum value per row on dimension 1 (block_N)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

            # Compute the factor by which we need to rescale previous partial sums
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])

            # Rescale the partial output accumulation to keep exponents consistent
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

            # Exponentiate (scores - max) for the new block
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

            # Make a cast of acc_s to fp16 for the next GEMM
            T.copy(acc_s, acc_s_cast)

            # Multiply the attention acc_s_cast by V and add to partial output (acc_o)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            # Update the "logsum" tracker with the newly accumulated sum
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        # Final step: divide each partial output by logsum (completing the softmax)
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]

        # Write back the final output block from acc_o to the Output buffer
        T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
```
