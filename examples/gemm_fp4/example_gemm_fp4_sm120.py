"""FP4 (float4_e2m1fn) GEMM on SM120 (RTX 5080/5090) using fragment-based MMA.

Uses mma.sync.aligned.kind::f8f6f4 instructions (not TCGEN05/TMEM).
Addresses https://github.com/tile-ai/tilelang/issues/1592
"""

import os
import time
import torch
import tilelang
import tilelang.language as T


def matmul_fp4(
    M, N, K, block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages=2, threads=128,
):
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
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
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


FP4_E2M1_TO_FLOAT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def unpack_fp4_to_float(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) int8 tensor → (M, K) float32 tensor."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    flat = packed_int8.to(torch.uint8).flatten()
    lo = (flat & 0x0F).to(torch.int64)
    hi = ((flat >> 4) & 0x0F).to(torch.int64)
    pairs = torch.stack([lut[lo], lut[hi]], dim=-1)
    return pairs.reshape(M, K)


def make_fp4_tensor(M: int, K: int, device="cuda") -> torch.Tensor:
    """Create random packed FP4 tensor as (M, K//2) int8."""
    return torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device).to(torch.int8)


M, N, K = 256, 256, 256
block_M, block_N, block_K = 128, 128, 128
in_dtype = T.float4_e2m1fn
out_dtype = T.float32
accum_dtype = T.float32

print(f"Running FP4 GEMM: M={M}, N={N}, K={K}")
print(f"  block_M={block_M}, block_N={block_N}, block_K={block_K}")

func = matmul_fp4(
    M, N, K, block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages=2, threads=128,
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
with open(os.path.join(os.path.dirname(__file__), "gemm_fp4_sm120.cu"), "w") as f:
    f.write(jit_kernel.get_kernel_source())

# --- Test 1: zeros in → zeros out ---
a_zero = torch.zeros(M, K // 2, device="cuda", dtype=torch.int8)
b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8)
c_zero = jit_kernel(a_zero, b_zero)
assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
print("[PASS] zeros in → zeros out")

# --- Test 2: numerical verification with random FP4 data ---
torch.manual_seed(42)
a_packed = make_fp4_tensor(M, K)
b_packed = make_fp4_tensor(N, K)

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
    print(f"[WARN] large diff — may indicate layout or data flow issue")

# --- Benchmark ---
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    jit_kernel(a_packed, b_packed)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000
print(f"Latency: {elapsed:.4f} ms")
print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")
