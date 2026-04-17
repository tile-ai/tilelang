"""FP4 (float4_e2m1fn) GEMM on SM100 (B200 / Thor) using TCGEN05MMA.

Uses TCGEN05MMA instructions with TMEM accumulator for SM100 Blackwell
architecture.  Both A and B operands use float4_e2m1fn (e2m1, 4-bit FP)
in K-major layout (A: MxK row-major, B: NxK row-major with trans_B=True).

C = A @ B^T  (M x N, float32)

Requires: SM100 GPU (B200 / Thor), CUDA 12.8+, PyTorch >= 2.4
"""

import torch
import tilelang
import tilelang.language as T
import os


def matmul_fp4_sm100(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    out_dtype,
    accum_dtype,
    num_stages=2,
    threads=256,
):
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, "float4_e2m1fn"),
        B: T.Tensor(B_shape, "float4_e2m1fn"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, "float4_e2m1fn")
            B_shared = T.alloc_shared(B_shared_shape, "float4_e2m1fn")
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
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
# Host-side FP4 helpers
# ---------------------------------------------------------------------------
FP4_E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,  # positive (sign=0)
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,  # negative (sign=1)
]


def pack_fp4_random(M: int, K: int, device="cuda") -> torch.Tensor:
    """Create packed FP4 data: (M, K//2) int8, 2 nibbles per byte."""
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

    print(f"SM100 FP4 GEMM: M={M}, N={N}, K={K}")
    print(f"  block=({block_M},{block_N},{block_K}), stages={num_stages}")

    func = matmul_fp4_sm100(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        out_dtype,
        accum_dtype,
        num_stages,
        threads,
    )

    jit_kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print("Compilation succeeded!")
    with open(os.path.join(os.path.dirname(__file__), "gemm_fp4_sm100.cu"), "w") as f:
        f.write(jit_kernel.get_kernel_source())

    torch.manual_seed(42)

    # Create packed FP4 data
    a_packed = pack_fp4_random(M, K)
    b_packed = pack_fp4_random(N, K)

    # --- Test 1: zero inputs ---
    a_zero = torch.zeros(M, K // 2, device="cuda", dtype=torch.int8)
    b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8)
    c_zero = jit_kernel(a_zero, b_zero)
    assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
    print("[PASS] zeros in -> zeros out")

    # --- Test 2: numerical verification ---
    c = jit_kernel(a_packed, b_packed)

    a_float = unpack_fp4_to_float(a_packed, M, K)
    b_float = unpack_fp4_to_float(b_packed, N, K)
    ref_c = a_float @ b_float.T

    diff = (c.float() - ref_c).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    if max_diff < 1.0:
        print("[PASS] numerical verification (max_abs_diff < 1.0)")
    else:
        print("[WARN] large diff -- investigate layout or descriptor issue")

    # --- Benchmark ---
    profiler = jit_kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Latency: {latency:.4f} ms")
    print(f"TFLOPS:  {2 * M * N * K / (latency / 1e3) / 1e12:.2f}")
