"""FP4 (float4_e2m1fn) GEMM on SM100/SM110 (B200/Thor) using TCGEN05 async MMA.

Uses tcgen05.mma.kind::f8f6f4 with TMEM accumulator.
The TMA path keeps global FP4 packed, then writes it into SMEM using the
ALIGN16B unpacksmem layout expected by tcgen05.mma.

Supported: SM100 (B100/B200), SM101/SM110 (DRIVE Thor), SM103 (B300).
"""

import os
import time

import torch
import tilelang
import tilelang.language as T


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


def matmul_fp4_sm100(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages=2,
    threads=256,
    transpose_b=True,
):
    """FP4 GEMM using TCGEN05 async MMA + TMEM."""
    A_shape = (M, K)
    A_shared_shape = (block_M, block_K)
    if transpose_b:
        B_shape = (N, K)
        B_shared_shape = (block_N, block_K)

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype)
                C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
                mbar = T.alloc_barrier(1)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.tcgen05_gemm(
                        A_shared,
                        B_shared,
                        C_tmem,
                        transpose_A=False,
                        transpose_B=True,
                        mbar=mbar,
                        clear_accum=(k == 0),
                    )
                    T.mbarrier_wait_parity(mbar, k % 2)

                T.copy(C_tmem, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    B_shape = (K, N)
    B_shared_shape = (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_A=False,
                    transpose_B=False,
                    mbar=mbar,
                    clear_accum=(k == 0),
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def unpack_fp4_to_float(packed_int8, M, K):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    flat = packed_int8.to(torch.uint8).reshape(M, K // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(M, K).to(torch.int64)
    return lut[unpacked]


M = int(os.environ.get("TL_FP4_M", "256"))
N = int(os.environ.get("TL_FP4_N", "256"))
K = int(os.environ.get("TL_FP4_K", "256"))
block_M = int(os.environ.get("TL_FP4_BLOCK_M", "128"))
block_N = int(os.environ.get("TL_FP4_BLOCK_N", "64"))
block_K = int(os.environ.get("TL_FP4_BLOCK_K", "128"))
in_dtype = T.float4_e2m1fn
out_dtype = T.float32
accum_dtype = T.float32
num_stages = 1
threads = 128

input_mode = os.environ.get("TL_FP4_INPUT_MODE", "random")
transpose_b = os.environ.get("TL_FP4_TRANSPOSE_B", "1") != "0"
print(f"Running FP4 GEMM (SM100/SM110 TCGEN05): M={M}, N={N}, K={K}, input_mode={input_mode}, transpose_b={transpose_b}")

func = matmul_fp4_sm100(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    transpose_b,
)

jit_kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)

print("Compilation succeeded!")

torch.manual_seed(42)


def make_random_fp4(rows, cols, mode):
    if mode == "positive":
        lo = torch.randint(0, 8, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        hi = torch.randint(0, 8, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        return (lo | (hi << 4)).to(torch.int8)
    if mode == "low_nibble":
        lo = torch.randint(0, 16, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        return lo.to(torch.int8)
    if mode == "high_nibble":
        hi = torch.randint(0, 16, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        return (hi << 4).to(torch.int8)
    if mode == "random":
        return torch.randint(0, 256, (rows, cols // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    raise ValueError(f"Unsupported TL_FP4_INPUT_MODE={mode}")


a_packed = make_random_fp4(M, K, input_mode)
b_packed = make_random_fp4(N, K, input_mode) if transpose_b else make_random_fp4(K, N, input_mode)

a_zero = torch.zeros(M, K // 2, device="cuda", dtype=torch.int8)
b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8) if transpose_b else torch.zeros(K, N // 2, device="cuda", dtype=torch.int8)
c_zero = jit_kernel(a_zero, b_zero)
assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
print("[PASS] zeros in -> zeros out")

c = jit_kernel(a_packed, b_packed)
a_float = unpack_fp4_to_float(a_packed, M, K)
b_float = unpack_fp4_to_float(b_packed, N, K) if transpose_b else unpack_fp4_to_float(b_packed, K, N)
ref_c = a_float @ (b_float.T if transpose_b else b_float)

diff = (c.float() - ref_c).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    jit_kernel(a_packed, b_packed)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000
print(f"Latency: {elapsed:.4f} ms")
print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")
