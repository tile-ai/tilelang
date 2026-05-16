"""Grouped GEMM forward on SM100 (Blackwell), FP8 inputs / BF16 output.

Same warp-specialized TMA + tcgen05.mma pipeline as the bf16 variant
(`example_grouped_gemm_fwd_sm100.py`), but using FP8 for A/B and the
"NT" layout that tcgen05.mma requires for FP8:

  A: (sum(batch_sizes), K)        in_dtype (fp8_e4m3 by default)
  B: (G, N, K)                    in_dtype  -- each expert's B is (N, K), K-contig
  C: (sum(batch_sizes), N)        out_dtype (bf16)

Note B's per-expert layout is (N, K), NOT (K, N) like the bf16 grouped kernel.
This matches the NT layout used by `examples/gemm_fp8/example_tilelang_gemm_fp8_sm100.py`
and the grouped mxfp8 kernels.
"""

import argparse
import torch

import tilelang
import tilelang.language as T

from example_grouped_gemm_fwd import construct_inputs as _construct_inputs_dense  # not used; kept for parity


def construct_inputs_fp8(batch_sizes_list, K, N, padding_M, device, in_torch_dtype, out_torch_dtype):
    """Build A, B (transposed per-expert), C, and the three offset tensors for fp8."""
    import math

    G = len(batch_sizes_list)
    total_M = sum(batch_sizes_list)

    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(G - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(G - 1):
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1] + math.ceil(batch_sizes_list[i] / padding_M) * padding_M
        )

    # randn -> fp16 -> fp8 (avoids torch.randn-into-fp8 limitations)
    A = torch.randn(total_M, K, device=device, dtype=torch.float16).to(in_torch_dtype)
    B = torch.randn(G, N, K, device=device, dtype=torch.float16).to(in_torch_dtype)
    C = torch.empty(total_M, N, device=device, dtype=out_torch_dtype)

    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets


def torch_gmm_fp8_ref(A, B, batch_sizes_list, batch_offsets_list, out_torch_dtype):
    """Per-group reference: A[start:end].to(fp16) @ B[g].T.to(fp16) -> out_dtype."""
    total_M = sum(batch_sizes_list)
    N = B.shape[1]
    out = torch.empty(total_M, N, device=A.device, dtype=out_torch_dtype)
    start = 0
    for g, size in enumerate(batch_sizes_list):
        end = start + size
        a = A[start:end].to(torch.float16)
        b = B[g].to(torch.float16)             # (N, K)
        out[start:end] = (a @ b.T).to(out_torch_dtype)
        start = end
    return out


@tilelang.jit(out_idx=[2])
def grouped_gemm_sm100_fp8(
    batch_sizes_list,
    K,
    N,
    block_M,
    block_N,
    block_K,
    num_stages=4,
    in_dtype=T.float8_e4m3fn,
    out_dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], in_dtype),  # type: ignore
        B: T.Tensor([batch_count, N, K], in_dtype),  # type: ignore  -- (N, K) per expert
        C: T.Tensor([batch_sum, N], out_dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), in_dtype)
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

            # --- per-block group-index lookup (same as bf16 sm100 grouped kernel) ---
            m_start_padded = bx * block_M
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)
            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = (m_start_padded
                       - batch_padded_offsets[cur_batch_idx]
                       + batch_offsets[cur_batch_idx])
            actual_rows = T.max(
                0,
                T.min(block_M,
                      cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded),
            )

            k_iters = T.ceildiv(K, block_K)

            if tx < 32:  # warp 0: issue TMA
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(consumed[k % num_stages],
                                           ((k // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[m_start : m_start + block_M,
                          k * block_K : (k + 1) * block_K],
                        A_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    # B is (G, N, K); load (block_N, block_K) tile for current expert.
                    T.tma_copy(
                        B[cur_batch_idx,
                          by * block_N : (by + 1) * block_N,
                          k * block_K : (k + 1) * block_K],
                        B_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.mbarrier_arrive(loaded[k % num_stages])
            elif tx < 64:  # warp 1: issue tcgen05.mma (NT: transpose_B=True)
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(loaded[k % num_stages],
                                           (k // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[k % num_stages, :, :],
                        B_shared[k % num_stages, :, :],
                        C_tmem,
                        transpose_B=True,
                        mbar=consumed[k % num_stages],
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            # epilogue: TMEM(fp32) -> fragment -> cast to bf16 -> masked element-wise store
            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_local_cast)
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local_cast[i, j]

    return kernel


def run(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages,
        fp8_dtype="e4m3", profile=False):
    in_torch_dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fp8_dtype]
    out_torch_dtype = torch.bfloat16
    in_tvm_dtype = {"e4m3": T.float8_e4m3fn, "e5m2": T.float8_e5m2}[fp8_dtype]

    kernel = grouped_gemm_sm100_fp8(
        tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages,
        in_dtype=in_tvm_dtype,
    )

    device = torch.device("cuda")
    A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs_fp8(
        batch_sizes_list, K, N, padding_M=block_M, device=device,
        in_torch_dtype=in_torch_dtype, out_torch_dtype=out_torch_dtype,
    )

    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref = torch_gmm_fp8_ref(A, B, batch_sizes_list,
                            batch_offsets.tolist(), out_torch_dtype)
    # fp8 has wider tolerance than bf16
    if torch.allclose(out, ref, rtol=2e-2, atol=2e-2):
        print(f"✅ Tilelang(SM100, {fp8_dtype}) and Torch(fp16 ref) match")
    else:
        diff = (out.float() - ref.float()).abs()
        print(f"❌ Tilelang(SM100, {fp8_dtype}) mismatch: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=100, rep=200,
            input_tensors=[A, B, batch_sizes, batch_offsets, batch_padded_offsets],
        )
        total_M = sum(batch_sizes_list)
        print(f"Latency: {latency:.4f} ms")
        print(f"TFlops:  {2 * total_M * N * K / latency * 1e-9:.2f} TFLOPS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_sizes", type=str, default="1024,1024,1024,1024",
                   help="comma-separated per-group row counts")
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--block_M", type=int, default=128)
    p.add_argument("--block_N", type=int, default=256)
    p.add_argument("--block_K", type=int, default=128)  # fp8 favors larger block_K
    p.add_argument("--num_stages", type=int, default=4)
    p.add_argument("--fp8_dtype", type=str, default="e4m3", choices=["e4m3", "e5m2"])
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    bs = [int(x) for x in args.batch_sizes.split(",")]
    run(bs, args.K, args.N, args.block_M, args.block_N, args.block_K,
        args.num_stages, fp8_dtype=args.fp8_dtype, profile=args.profile)
