"""Grouped GEMM forward (SM90 style): bf16 input × fp8 weight (in-block dequant) → bf16.

Reuses the simple gemm body from `example_grouped_gemm_fwd.py` (single-warp,
`T.Pipelined` loop, `T.copy` + `T.gemm`) and the fp8→bf16 dequant pattern from
`example_grouped_gemm_fwd_sm100_fp8_to_bf16.py`: per-thread vec-of-8 staged
through `cur_fp8x8` → `cur_fp32x8` → `cur_bf16x8` local buffers, written into a
bf16 shared buffer that `T.gemm` consumes.

Layouts (NN, matching `example_grouped_gemm_fwd_sm100_fp8_to_bf16.py`):
    A: (sum(batch_sizes), K)         bfloat16
    B: (G, K, N)                     fp8_e4m3   -- per-expert (K, N), N-contig
    C: (sum(batch_sizes), N)         bfloat16
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
def grouped_gemm_sm90_fp8_to_bf16(
    batch_sizes_list,
    K,
    N,
    block_M=128,
    block_N=128,
    block_K=64,
    num_stages=2,
    threads=128,
    in_dtype_a=T.bfloat16,
    in_dtype_b=T.float8_e4m3fn,
    out_dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    assert block_N % 8 == 0, "block_N must be a multiple of 8 (vec-of-8 fp8 reads)"
    assert (block_K * block_N) % (threads * 8) == 0, \
        "block_K * block_N must be a multiple of threads * 8 (clean per-thread vec-of-8 tiling)"

    vecs_per_thread = (block_K * block_N) // (threads * 8)

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
            A_shared = T.alloc_shared([block_M, block_K], in_dtype_a)
            B_fp8_shared = T.alloc_shared([block_K, block_N], in_dtype_b)
            B_shared = T.alloc_shared([block_K, block_N], in_dtype_a)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

            cur_fp8x8 = T.alloc_local((8,), in_dtype_b)
            cur_fp32x8 = T.alloc_local((8,), T.float32)
            cur_bf16x8 = T.alloc_local((8,), in_dtype_a)

            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            tx = T.get_thread_binding()

            m_start_padded = bx * block_M

            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_fp8_shared)

                # Per-thread vec-of-8 dequant: fp8 → fp32 → bf16, four explicit
                # T.vectorized(8) loops (load / cast / cast / store), no inline
                # T.cast(T.cast(...)).
                for vi in T.serial(vecs_per_thread):
                    flat = (vi * threads + tx) * 8
                    row = flat // block_N
                    col_base = flat % block_N
                    for i in T.vectorized(8):
                        cur_fp8x8[i] = B_fp8_shared[row, col_base + i]
                    for i in T.vectorized(8):
                        cur_fp32x8[i] = T.cast(cur_fp8x8[i], "float")
                    for i in T.vectorized(8):
                        cur_bf16x8[i] = T.cast(cur_fp32x8[i], in_dtype_a)
                    for i in T.vectorized(8):
                        B_shared[row, col_base + i] = cur_bf16x8[i]

                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = T.cast(C_local[i, j], out_dtype)

    return kernel


def run(batch_sizes_list, K, N, block_M=128, block_N=128, block_K=64,
        num_stages=2, threads=128, profile=False):
    kernel = grouped_gemm_sm90_fp8_to_bf16(
        tuple(batch_sizes_list), K, N,
        block_M=block_M, block_N=block_N, block_K=block_K,
        num_stages=num_stages, threads=threads,
    )

    device = torch.device("cuda")
    A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs_bf16_fp8(
        batch_sizes_list, K, N, padding_M=block_M, device=device,
    )

    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref = torch_gmm_bf16_fp8_ref(A, B, batch_sizes_list, batch_offsets.tolist())
    if torch.allclose(out, ref, rtol=2e-2, atol=2e-2):
        print("✅ Tilelang(SM90, bf16×fp8→bf16) and torch ref match")
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
    p.add_argument("--block_N", type=int, default=128)
    p.add_argument("--block_K", type=int, default=64)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    bs = [int(x) for x in args.batch_sizes.split(",")]
    run(bs, args.K, args.N,
        block_M=args.block_M, block_N=args.block_N,
        block_K=args.block_K, num_stages=args.num_stages,
        threads=args.threads, profile=args.profile)
