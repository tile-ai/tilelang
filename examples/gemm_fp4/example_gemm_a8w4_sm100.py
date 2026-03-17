"""A8W4 GEMM on SM100 (B200 / Thor): FP8 activation x FP4 weight.

Uses TCGEN05MMA with TMEM accumulator and mixed-precision operands:
  A: float8_e4m3fn  (activation, 8-bit)
  B: float4_e2m1fn  (weight, 4-bit, packed 2 per byte)
  C: float32        (accumulator / output)

C = A @ B^T   (M x N, float32)

Both operands are K-major: A=(M,K) row-major, B=(N,K) row-major with
trans_B=True.

Requires: SM100 GPU, CUDA 12.8+, PyTorch >= 2.4
"""

import torch
import tilelang
import tilelang.language as T


def gemm_a8w4_sm100(
    M, N, K,
    block_M, block_N, block_K,
    out_dtype, accum_dtype,
    num_stages=2, threads=256,
):
    A_shape = (M, K)
    B_shape = (N, K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, "float8_e4m3fn"),
        B: T.Tensor(B_shape, "float4_e2m1fn"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_shared = T.alloc_shared((block_N, block_K), "float4_e2m1fn")
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages
            ):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    False,
                    True,
                    mbar=mbar,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------
FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def pack_fp4_random(M: int, K: int, device="cuda") -> torch.Tensor:
    """Packed FP4: (M, K//2) int8, two nibbles per byte."""
    lo = torch.randint(0, 16, (M, K // 2), device=device, dtype=torch.uint8)
    hi = torch.randint(0, 16, (M, K // 2), device=device, dtype=torch.uint8)
    return ((hi << 4) | lo).to(torch.int8)


def unpack_fp4_to_float(packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) packed int8 -> (M, K) float32 via e2m1 LUT."""
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    raw = packed.to(torch.uint8).reshape(M, K // 2)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    indices = torch.stack([lo, hi], dim=-1).reshape(M, K).to(torch.int64)
    return lut[indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    M, N, K = 256, 256, 256
    block_M, block_N, block_K = 128, 128, 128
    out_dtype = T.float32
    accum_dtype = T.float32
    num_stages = 2
    threads = 256

    print(f"SM100 A8W4 GEMM: M={M}, N={N}, K={K}")
    print(f"  A: float8_e4m3fn, B: float4_e2m1fn (packed)")

    func = gemm_a8w4_sm100(
        M, N, K, block_M, block_N, block_K,
        out_dtype, accum_dtype, num_stages, threads,
    )

    jit_kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print("Compilation succeeded!")
    print(jit_kernel.get_kernel_source())

    torch.manual_seed(42)

    # A: FP8 e4m3 activation
    a_fp8 = torch.randn(M, K, device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )

    # B: packed FP4 weight (N, K//2)
    b_packed = pack_fp4_random(N, K)

    # --- Test 1: zeros ---
    a_zero = torch.zeros(M, K, device="cuda", dtype=torch.float8_e4m3fn)
    b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8)
    c_zero = jit_kernel(a_zero, b_zero)
    assert c_zero.abs().max().item() == 0.0, (
        f"Zero test failed: max={c_zero.abs().max().item()}"
    )
    print("[PASS] zeros in -> zeros out")

    # --- Test 2: numerical verification ---
    c = jit_kernel(a_fp8, b_packed)

    a_ref = a_fp8.to(torch.float32)
    b_ref = unpack_fp4_to_float(b_packed, N, K)
    ref_c = a_ref @ b_ref.T

    diff = (c.float() - ref_c).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    if rel_err < 0.01:
        print("[PASS] numerical verification (rel_err < 0.01)")
    else:
        print("[WARN] large diff -- investigate descriptor or layout issue")

    # --- Benchmark ---
    profiler = jit_kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Latency: {latency:.4f} ms")
    print(f"TFLOPS:  {2 * M * N * K / (latency / 1e3) / 1e12:.2f}")
