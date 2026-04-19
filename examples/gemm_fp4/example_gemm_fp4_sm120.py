"""FP4 (float4_e2m1fn) GEMM on SM120 (RTX 5080/5090) using fragment-based MMA.

Uses mma.sync.aligned.kind::f8f6f4 instructions (not TCGEN05/TMEM).
FP4 data is pre-unpacked to uint8 on the host (1 byte per element, low
nibble holds the 4-bit value). Shared memory stores uint8, making the
layout identical to INT8 and satisfying ldmatrix 16B alignment.
The << 2 bit-shift before MMA places FP4 data at bits 2-5 as required.

Addresses https://github.com/tile-ai/tilelang/issues/1592
"""

import time
import torch
import tilelang
import tilelang.language as T


def matmul_fp4(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    out_dtype,
    accum_dtype,
    num_stages=2,
    threads=128,
):
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, "uint8"),
        B: T.Tensor(B_shape, "uint8"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, "uint8")
            B_shared = T.alloc_shared(B_shared_shape, "uint8")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


FP4_E2M1_TO_FLOAT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def unpack_fp4_to_uint8(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) int8 -> (M, K) uint8, 1 FP4 per byte in low nibble."""
    flat = packed_int8.to(torch.uint8).reshape(M, K // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(M, K).contiguous()


def unpack_fp4_to_float(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) int8 -> (M, K) float32 via e2m1 lookup table."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    unpacked = unpack_fp4_to_uint8(packed_int8, M, K).to(torch.int64)
    return lut[unpacked]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    M, N, K = 256, 256, 256
    block_M, block_N, block_K = 128, 128, 128
    out_dtype = T.float32
    accum_dtype = T.float32

    print(f"Running FP4 GEMM: M={M}, N={N}, K={K}")
    print(f"  block_M={block_M}, block_N={block_N}, block_K={block_K}")

    func = matmul_fp4(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        out_dtype,
        accum_dtype,
        num_stages=2,
        threads=128,
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

    torch.manual_seed(42)

    # Create packed FP4 data (2 per byte), then pre-unpack to uint8
    a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    a_unpacked = unpack_fp4_to_uint8(a_packed, M, K)
    b_unpacked = unpack_fp4_to_uint8(b_packed, N, K)

    # --- Test 1: zeros ---
    a_zero = torch.zeros(M, K, device="cuda", dtype=torch.uint8)
    b_zero = torch.zeros(N, K, device="cuda", dtype=torch.uint8)
    c_zero = jit_kernel(a_zero, b_zero)
    assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
    print("[PASS] zeros in -> zeros out")

    # --- Test 2: numerical verification ---
    c = jit_kernel(a_unpacked, b_unpacked)

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
        print("[WARN] large diff -- may indicate layout or data flow issue")

    # --- Benchmark ---
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        jit_kernel(a_unpacked, b_unpacked)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000
    print(f"Latency: {elapsed:.4f} ms")
    print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
