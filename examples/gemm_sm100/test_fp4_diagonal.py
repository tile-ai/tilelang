"""Diagnostic: FP4 GEMM with diagonal B matrix.

C = A @ B^T where B = identity → C should equal A (in float).
Mismatches reveal exactly which (row, col) the MMA reads incorrectly.
"""

import os
import torch
import tilelang
import tilelang.language as T

M, N, K = 256, 256, 256
# Keep K=128 for the current ALIGN16B gap-aware layout, but reduce N so the
# diagnostic runs within Thor's dynamic shared-memory budget.
#
# We intentionally keep block_M=128 so TCGEN05 stores C through the standard
# atom_m=128 TMEM layout (Layout D) instead of the atom_m=64 WS-specific layout.
block_M, block_N, block_K = 128, 64, 128
in_dtype = T.float4_e2m1fn
out_dtype = T.float32
accum_dtype = T.float32

FP4_E2M1_TO_FLOAT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def unpack_fp4_to_float(packed_int8, rows, cols):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    flat = packed_int8.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def make_fp4_diagonal(N, K):
    """Create packed FP4 identity matrix B[N,K] where B[n,k]=1.0 if n==k."""
    packed = torch.zeros(N, K // 2, dtype=torch.uint8, device="cuda")
    fp4_one = 2  # FP4 encoding for 1.0
    diag = torch.arange(min(N, K), device="cuda", dtype=torch.int64)
    byte_idx = diag // 2
    nibble_shift = (diag % 2).to(torch.uint8) * 4
    packed[diag, byte_idx] = torch.bitwise_left_shift(
        torch.full_like(nibble_shift, fp4_one, dtype=torch.uint8), nibble_shift
    )
    return packed.to(torch.int8)


def make_fp4_row_pattern(M, K):
    """A[m,k] = FP4 value based on (m % 8), so each row has a recognizable pattern."""
    vals = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device="cuda")
    row_vals = vals[torch.arange(M, device="cuda", dtype=torch.int64) % 8]
    packed_row_vals = row_vals | torch.bitwise_left_shift(row_vals, 4)
    packed = packed_row_vals[:, None].expand(M, K // 2).contiguous()
    return packed.to(torch.int8)


def print_generated_kernel_debug_summary(kernel_src_path: str):
    keywords = (
        "arrive_and_expect_tx",
        "initialize_tcgen05_descriptor",
        "fence_proxy_async",
        "tcgen05mma_ss",
        ".wait(",
    )
    with open(kernel_src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    first_mma_line = None
    first_fence_line = None

    print("Generated kernel debug summary:")
    for line_no, line in enumerate(lines, start=1):
        if "tcgen05mma_ss" in line and first_mma_line is None:
            first_mma_line = line_no
        if "fence_proxy_async" in line and first_fence_line is None:
            first_fence_line = line_no
        if any(keyword in line for keyword in keywords):
            print(f"{line_no:4d}: {line.rstrip()}")

    if first_mma_line is None:
        print("No tcgen05mma_ss call found in generated source.")
        return

    if first_fence_line is None:
        print("No fence_proxy_async() found in generated source.")
    elif first_fence_line < first_mma_line:
        print(
            f"First fence_proxy_async() appears before first tcgen05mma_ss "
            f"(fence line {first_fence_line}, mma line {first_mma_line})."
        )
    else:
        print(
            f"First fence_proxy_async() appears after first tcgen05mma_ss "
            f"(fence line {first_fence_line}, mma line {first_mma_line})."
        )


def matmul_fp4_sm100(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.tcgen05_gemm(A_shared, B_shared, C_tmem,
                               transpose_A=False, transpose_B=True,
                               mbar=mbar, clear_accum=(k == 0))
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    return main


print(f"FP4 Diagonal Test: M={M}, N={N}, K={K}")

func = matmul_fp4_sm100(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype)

kernel = tilelang.compile(func, out_idx=[2], target="cuda", pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
print("Compiled OK")

kernel_src_path = os.path.join(os.path.dirname(__file__), "test_fp4_diagonal.generated.cu")
with open(kernel_src_path, "w", encoding="utf-8") as f:
    f.write(kernel.get_kernel_source())
print(f"Kernel source written to: {kernel_src_path}")
print_generated_kernel_debug_summary(kernel_src_path)

if os.environ.get("TL_FP4_COMPILE_ONLY", "0") == "1":
    print("TL_FP4_COMPILE_ONLY=1, skip kernel launch.")
    raise SystemExit(0)

# --- Test: A = row pattern, B = identity ---
print("Building FP4 test inputs...")
a_packed = make_fp4_row_pattern(M, K)
b_packed = make_fp4_diagonal(N, K)
torch.cuda.synchronize()
print("Launching kernel...")

c = kernel(a_packed, b_packed)
torch.cuda.synchronize()
print("Kernel finished.")

a_float = unpack_fp4_to_float(a_packed, M, K)
b_float = unpack_fp4_to_float(b_packed, N, K)
ref_c = a_float @ b_float.T  # should ≈ A (since B is identity)

diff = (c.float() - ref_c).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
print(f"max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")

out = c
print(f"\n{'='*80}")
print(f"Element-by-element comparison (first 8 rows x first 48 cols):")
print(f"{'='*80}")

# Print header
print(f"{'':>6}", end="")
for col in range(48):
    print(f" {col:>5}", end="")
print()

for r in range(8):
    # Expected row
    print(f"E[{r:>2}]", end="")
    for col in range(48):
        print(f" {ref_c[r,col].item():>5.1f}", end="")
    print()
    # Actual row
    print(f"A[{r:>2}]", end="")
    for col in range(48):
        v = out[r, col].item()
        e = ref_c[r, col].item()
        mark = " " if abs(v - e) < 0.01 else "*"
        print(f"{v:>5.1f}{mark}", end="")
    print()
    print()

print(f"\n{'='*80}")
print(f"Columns 120-135 (crossing 128 boundary):")
print(f"{'='*80}")
print(f"{'':>6}", end="")
for col in range(120, 136):
    print(f" {col:>5}", end="")
print()
for r in range(4):
    print(f"E[{r:>2}]", end="")
    for col in range(120, 136):
        print(f" {ref_c[r,col].item():>5.1f}", end="")
    print()
    print(f"A[{r:>2}]", end="")
    for col in range(120, 136):
        v = out[r, col].item()
        e = ref_c[r, col].item()
        mark = " " if abs(v - e) < 0.01 else "*"
        print(f"{v:>5.1f}{mark}", end="")
    print()
    print()
