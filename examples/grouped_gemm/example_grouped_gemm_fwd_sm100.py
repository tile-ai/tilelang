"""Grouped GEMM forward on SM100 (Blackwell), using TMA + tcgen05.mma.

Combines the per-block group-index lookup from
`example_grouped_gemm_fwd.py` with the warp-specialized TMA + tcgen05.mma
pipeline from `examples/gemm_sm100/gemm_tcgen5mma_ws.py`:
  - warp 0 (lane <32) issues TMA loads for the A / B tiles,
  - warp 1 (lane 32-63) issues `tcgen05_gemm` accumulating into TMEM,
  - epilogue copies TMEM -> fragment -> masked element-wise store
    (the grouped kernel has to mask the row tail, so we cannot use the
    unconditional TMA store path that the dense SM100 kernel uses).

Layout (matches the existing grouped GEMM example):
  A: (sum(batch_sizes), K)        in_dtype
  B: (G, K, N)                    in_dtype
  C: (sum(batch_sizes), N)        out_dtype
plus three auxiliary int32 tensors of shape (G,):
  batch_sizes, batch_offsets (unpadded), batch_padded_offsets.
"""

import argparse
import torch

import tilelang
import tilelang.language as T

from example_grouped_gemm_fwd import construct_inputs, torch_gmm


@tilelang.jit(out_idx=[2])
def grouped_gemm_sm100(
    batch_sizes_list,
    K,
    N,
    block_M,
    block_N,
    block_K,
    num_stages=4,
    in_dtype=T.bfloat16,
    out_dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], in_dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], in_dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], out_dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
            loaded = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            tx = T.get_thread_binding()

            T.use_swizzle(8)

            # --- per-block group-index lookup (same as sm80 grouped kernel) ---
            m_start_padded = bx * block_M
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)
            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(
                0,
                T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded),
            )

            k_iters = T.ceildiv(K, block_K)

            if tx < 32:  # warp 0: issue TMA
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.tma_copy(
                        B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                        B_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.mbarrier_arrive(loaded[k % num_stages])
            elif tx < 64:  # warp 1: issue tcgen05.mma
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[k % num_stages, :, :],
                        B_shared[k % num_stages, :, :],
                        C_tmem,
                        mbar=consumed[k % num_stages],
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            # epilogue: TMEM -> fragment -> cast -> masked element-wise store
            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_local_cast)
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local_cast[i, j]

    return kernel


def run(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages, profile=False):
    kernel = grouped_gemm_sm100(
        tuple(batch_sizes_list),
        K,
        N,
        block_M,
        block_N,
        block_K,
        num_stages,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(
        batch_sizes_list,
        K,
        N,
        trans_b=False,
        padding_M=block_M,
        device=device,
        dtype=dtype,
    )

    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref = torch_gmm(A, B, batch_sizes, batch_offsets, trans_b=False)
    if torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
        print("✅ Tilelang(SM100) and Torch match")
    else:
        print("❌ Tilelang(SM100) and Torch mismatch")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=100,
            rep=200,
            input_tensors=[A, B, batch_sizes, batch_offsets, batch_padded_offsets],
        )
        total_M = sum(batch_sizes_list)
        print(f"Latency: {latency:.4f} ms")
        print(f"TFlops:  {2 * total_M * N * K / latency * 1e-9:.2f} TFLOPS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_sizes", type=str, default="1024,1024,1024,1024", help="comma-separated per-group row counts")
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--block_M", type=int, default=128)
    p.add_argument("--block_N", type=int, default=256)
    p.add_argument("--block_K", type=int, default=64)
    p.add_argument("--num_stages", type=int, default=4)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    bs = [int(x) for x in args.batch_sizes.split(",")]
    run(bs, args.K, args.N, args.block_M, args.block_N, args.block_K, args.num_stages, profile=args.profile)
