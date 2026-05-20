"""A8W4 GEMM on SM100/SM110 using TCGEN05 f8f6f4 MMA.

A is FP8 e4m3 activation. B is FP4 e2m1 weight stored densely packed on the
PyTorch side and loaded through the SM100 unpacksmem TMA/shared path.
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


def unpack_fp4_to_float(packed_int8, rows, cols):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    flat = packed_int8.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def matmul_a8w4_sm100(M, N, K, block_M, block_N, block_K, out_dtype, accum_dtype, num_stages=1, threads=128):
    A_shape = (M, K)
    B_shape = (N, K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, "float8_e4m3fn"),
        B: T.Tensor(B_shape, T.float4_e2m1fn),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1fn)
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


M = int(os.environ.get("TL_A8W4_M", "256"))
N = int(os.environ.get("TL_A8W4_N", "256"))
K = int(os.environ.get("TL_A8W4_K", "256"))
block_M = int(os.environ.get("TL_A8W4_BLOCK_M", "128"))
block_N = int(os.environ.get("TL_A8W4_BLOCK_N", "64"))
block_K = int(os.environ.get("TL_A8W4_BLOCK_K", "128"))

print(f"Running SM100 A8W4 GEMM: M={M}, N={N}, K={K}, block=({block_M},{block_N},{block_K})")

func = matmul_a8w4_sm100(M, N, K, block_M, block_N, block_K, T.float32, T.float32)
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
a_fp8 = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)

a_zero = torch.zeros(M, K, device="cuda", dtype=torch.float8_e4m3fn)
b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8)
c_zero = jit_kernel(a_zero, b_zero)
assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
print("[PASS] zeros in -> zeros out")

c = jit_kernel(a_fp8, b_packed)
ref = a_fp8.to(torch.float32) @ unpack_fp4_to_float(b_packed, N, K).T
diff = (c.float() - ref).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    jit_kernel(a_fp8, b_packed)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000
print(f"Latency: {elapsed:.4f} ms")
print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")
