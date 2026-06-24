"""Demo kernel for the pass visualizer: fused GEMM + bias + ReLU.

    C = relu(A @ B + bias)

`bias` is a per-output-column vector of shape (N,), broadcast across rows. Both
the bias add and the ReLU are fused into the GEMM epilogue: they run on the
accumulator fragment `C_local` (still in registers) right before writing back to
global memory. No extra kernel launch, no extra global round-trip.

Used as input to ``tilelang.utils.pass_visualizer.viewer``; see the package
README for the exact command.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def gemm_relu(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),  # per-column bias vector, broadcast over rows
        C: T.Tensor((M, N), dtype),
    ):
        # grid: one block per (block_M x block_N) output tile.
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            # Stream the K dimension through shared memory, accumulate in registers.
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # Fused bias + ReLU epilogue: applied on the accumulator before write-back.
            # bias is read straight from global (each element used once per tile).
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j] + bias[bx * block_N + j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
