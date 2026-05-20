"""A8W4 GEMM on SM120: FP8 (e4m3) activation x FP4 (e2m1) weight.

Uses SM120 native mma.sync.kind::f8f6f4 with mixed-type operands:
  A: float8_e4m3fn (activation, 1 byte/element)
  B: float4_e2m1fn stored as uint8 (weight, unpacked, 1 byte/element)
  C: float32 (accumulator/output)

No block scaling. Direct FP8 x FP4 tensor core multiply-accumulate.
"""

import time
import torch
import tilelang
import tilelang.language as T


def gemm_a8w4(
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

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, "float8_e4m3fn"),
        B: T.Tensor(B_shape, "uint8"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_shared = T.alloc_shared((block_N, block_K), "uint8")
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


def fp4_uint8_to_float(tensor_uint8):
    """Convert uint8 tensor (low nibble = FP4 e2m1) to float32."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=tensor_uint8.device)
    return lut[tensor_uint8.to(torch.int64)]


M, N, K = 256, 256, 256
block_M, block_N, block_K = 128, 128, 128
out_dtype = T.float32
accum_dtype = T.float32

print(f"Running A8W4 GEMM: M={M}, N={N}, K={K}")
print("  A: float8_e4m3fn, B: FP4 (unpacked uint8)")

func = gemm_a8w4(
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

# A: FP8 e4m3fn values (create as float then convert)
a_float = torch.randn(M, K, device="cuda", dtype=torch.float16)
a_fp8 = a_float.to(torch.float8_e4m3fn)

# B: FP4 e2m1 values (random nibbles 0-15, stored as uint8)
b_uint8 = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.uint8)

# --- Test 1: zeros ---
a_zero = torch.zeros(M, K, device="cuda", dtype=torch.float8_e4m3fn)
b_zero = torch.zeros(N, K, device="cuda", dtype=torch.uint8)
c_zero = jit_kernel(a_zero, b_zero)
assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
print("[PASS] zeros in -> zeros out")

# --- Test 2: numerical verification ---
c = jit_kernel(a_fp8, b_uint8)

a_ref = a_fp8.to(torch.float32)
b_ref = fp4_uint8_to_float(b_uint8)
ref_c = a_ref @ b_ref.T

diff = (c.float() - ref_c).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
if rel_err < 0.01:
    print("[PASS] numerical verification (rel_err < 0.01)")
else:
    print("[WARN] large diff -- may indicate layout or data flow issue")

# --- Benchmark ---
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    jit_kernel(a_fp8, b_uint8)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000
print(f"Latency: {elapsed:.4f} ms")
print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")
