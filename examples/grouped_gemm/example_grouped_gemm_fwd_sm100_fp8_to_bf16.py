"""Grouped GEMM forward on SM100: bf16 input × fp8 weight (in-block dequant) → bf16.

The mma is bf16 — fp8 weights are loaded into an fp8 staging shared buffer via
TMA, then cast in-block (fp8 → fp32 → bf16) into a bf16 shared buffer that
`tcgen05.mma` consumes. No DSMEM / cluster ops: each block dequants its own
weight tile.

Pipeline (warp-specialized, 128 threads / 4 warps per block):
    W0  (tx <  32):  TMA producer       — TMA bf16 A + TMA fp8 B → SMEM
    W1+W2 (32 - 95): dequant workers    — fp8 SMEM → bf16 SMEM (2 warps, 64 lanes)
    W3  (96 - 127):  tcgen05.mma issuer — bf16 mma into TMEM
                                        (all 128 threads join the epilogue)

Layouts (same NN convention as `example_grouped_gemm_fwd_sm100.py`):
    A: (sum(batch_sizes), K)         bfloat16
    B: (G, K, N)                     fp8_e4m3   -- per-expert (K, N), N-contig
    C: (sum(batch_sizes), N)         bfloat16

Reference for the in-block dequant pattern: examples/grouped_gemm/ref.py.

Assumptions:
    block_N % 256 == 0    -- 32 lanes × vec-of-8 fp8 reads per row.
    block_M % 128 == 0    -- tcgen05 tile size.
"""

import argparse
import math

import torch

import tilelang
import tilelang.language as T


def construct_inputs_bf16_fp8(batch_sizes_list, K, N, padding_M, device):
    """A bf16 (sum_M, K), B fp8 (G, K, N) — randn→fp16→fp8 cast for B."""
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

    A = torch.randn(total_M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(G, K, N, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    C = torch.empty(total_M, N, device=device, dtype=torch.bfloat16)

    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets


def torch_gmm_bf16_fp8_ref(A, B, batch_sizes_list, batch_offsets_list):
    """Per-group: A_bf16[start:end] @ B[g].to(bf16) → bf16."""
    total_M = sum(batch_sizes_list)
    N = B.shape[2]
    out = torch.empty(total_M, N, device=A.device, dtype=torch.bfloat16)
    for g, size in enumerate(batch_sizes_list):
        start = batch_offsets_list[g]
        end = start + size
        out[start:end] = (A[start:end].float() @ B[g].to(torch.bfloat16).float()).to(torch.bfloat16)
    return out


@tilelang.jit(out_idx=[2])
def grouped_gemm_sm100_fp8_to_bf16(
    batch_sizes_list,
    K,
    N,
    block_M=128,
    block_N=256,
    block_K=64,
    num_stages=3,
    in_dtype_a=T.bfloat16,
    in_dtype_b=T.float8_e4m3fn,
    out_dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert block_N % 256 == 0, "block_N must be a multiple of 256 (32 lanes × vec-of-8)"
    assert block_M % 128 == 0, "block_M must be a multiple of 128 (tcgen05 tile)"

    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    threads = 128
    n_chunks = block_N // 256  # how many 256-wide chunks each row splits into

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], in_dtype_a),                  # type: ignore
        B: T.Tensor([batch_count, K, N], in_dtype_b),             # type: ignore
        C: T.Tensor([batch_sum, N], out_dtype),                   # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),            # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),          # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),   # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared      = T.alloc_shared((num_stages, block_M, block_K), in_dtype_a)
            B_fp8_staging = T.alloc_shared((num_stages, block_K, block_N), in_dtype_b)
            B_shared      = T.alloc_shared((num_stages, block_K, block_N), in_dtype_a)  # bf16, post-dequant
            C_tmem        = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local       = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_cast  = T.alloc_fragment((block_M, block_N), out_dtype)

            tma_done  = T.alloc_barrier([32] * num_stages)  # producer → dequant
            loaded    = T.alloc_barrier([64] * num_stages)  # dequant (64 lanes) → mma
            consumed  = T.alloc_barrier([1]  * num_stages)  # mma      → producer
            tmem_full = T.alloc_barrier([1])                # mma      → epilogue

            cur_fp8x8  = T.alloc_local((8,), in_dtype_b)
            cur_fp32x8 = T.alloc_local((8,), T.float32)
            cur_bf16x8 = T.alloc_local((8,), in_dtype_a)

            cur_batch_idx  = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            tx = T.get_thread_binding()

            T.use_swizzle(8)

            # ----- per-block group-index lookup (mirrors example_grouped_gemm_fwd_sm100.py) -----
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
                T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded),
            )

            k_iters = T.ceildiv(K, block_K)

            # ============================ W0: TMA producer ============================
            if tx < 32:
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(consumed[k % num_stages],
                                           ((k // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[m_start : m_start + block_M,
                          k * block_K : (k + 1) * block_K],
                        A_shared[k % num_stages, :, :],
                        barrier=tma_done[k % num_stages],
                    )
                    # B is (G, K, N); per-expert (K, N), so slice [k*bK:(k+1)*bK, by*bN:(by+1)*bN]
                    T.tma_copy(
                        B[cur_batch_idx,
                          k * block_K : (k + 1) * block_K,
                          by * block_N : (by + 1) * block_N],
                        B_fp8_staging[k % num_stages, :, :],
                        barrier=tma_done[k % num_stages],
                    )
                    T.mbarrier_arrive(tma_done[k % num_stages])

            # ============================ W1+W2: dequant workers ======================
            elif tx < 96:
                # 64 lanes total; lane d = (tx - 32) ∈ [0, 64).
                # Map: row_in_pair = d // 32 (0 or 1), col_lane = d % 32 (0..31).
                # Each row-pair iter covers 2 rows × 32 lanes × 8 cols = 512 fp8 = 2 rows of block_N=256.
                deq_lane = tx - 32
                row_in_pair = deq_lane // 32
                col_lane = deq_lane % 32
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(tma_done[k % num_stages],
                                           (k // num_stages) & 1)
                    for ki in T.serial(block_K // 2):
                        row = ki * 2 + row_in_pair
                        for ci in T.serial(n_chunks):
                            col_base = ci * 256 + col_lane * 8
                            for i in T.vectorized(8):
                                cur_fp8x8[i] = B_fp8_staging[k % num_stages, row, col_base + i]
                            for i in T.vectorized(8):
                                cur_fp32x8[i] = T.cast(cur_fp8x8[i], "float")
                            for i in T.vectorized(8):
                                cur_bf16x8[i] = T.cast(cur_fp32x8[i], in_dtype_a)
                            for i in T.vectorized(8):
                                B_shared[k % num_stages, row, col_base + i] = cur_bf16x8[i]
                    T.mbarrier_arrive(loaded[k % num_stages])

            # ============================ W3: tcgen05.mma issuer ======================
            elif tx < 128:
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(loaded[k % num_stages],
                                           (k // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[k % num_stages, :, :],
                        B_shared[k % num_stages, :, :],
                        C_tmem,
                        mbar=consumed[k % num_stages],
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            # ============================ epilogue (all 128 threads) ==================
            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_local_cast)
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local_cast[i, j]

    return kernel


def run(batch_sizes_list, K, N, block_M=128, block_N=256, block_K=64, num_stages=3,
        profile=False):
    kernel = grouped_gemm_sm100_fp8_to_bf16(
        tuple(batch_sizes_list), K, N,
        block_M=block_M, block_N=block_N, block_K=block_K, num_stages=num_stages,
    )

    device = torch.device("cuda")
    A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs_bf16_fp8(
        batch_sizes_list, K, N, padding_M=block_M, device=device,
    )

    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref = torch_gmm_bf16_fp8_ref(A, B, batch_sizes_list, batch_offsets.tolist())
    if torch.allclose(out, ref, rtol=2e-2, atol=2e-2):
        print(f"✅ Tilelang(SM100, bf16×fp8→bf16) and torch ref match")
    else:
        diff = (out.float() - ref.float()).abs()
        print(f"❌ mismatch: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

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
    p.add_argument("--block_K", type=int, default=64)
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    bs = [int(x) for x in args.batch_sizes.split(",")]
    run(bs, args.K, args.N,
        block_M=args.block_M, block_N=args.block_N,
        block_K=args.block_K, num_stages=args.num_stages,
        profile=args.profile)
