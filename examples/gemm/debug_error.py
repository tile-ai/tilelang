import os
os.system('find ~/.tilelang/ -name "*.so" -print -delete')
os.environ["TL_DISABLE_G2S_HOIST"] = "1"
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_SCATTERED_WARP_LAYOUT: True,
})
def matmul_nt(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=4, num_xcds=8)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_v2(A_shared, B_shared, C_local, transpose_B=True, k_pack=2)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm

M, N, K = 256, 256, 1024
block_M, block_N, block_K = 256, 256, 64

kernel = matmul_nt(M, N, K, block_M, block_N, block_K)

a = torch.randn(M, K).cuda().bfloat16() *0.0 + 1
b_nt = torch.randn(N, K).cuda().bfloat16() *0.0 + 1
torch.cuda.synchronize()

c = kernel(a, b_nt)
ref_c = a @ b_nt.T
torch.cuda.synchronize()

diff = c - ref_c
abs_diff = diff.abs()

# Overall stats
total = M * N
mismatched = (abs_diff > 0.01).sum().item()
print(f"Total elements: {total}")
print(f"Mismatched elements: {mismatched} ({100*mismatched/total:.1f}%)")
print(f"Max abs diff: {abs_diff.max().item()}")

# Check for zeros in output
zero_rows = (c == 0).all(dim=1)
zero_cols = (c == 0).all(dim=0)
print(f"\nZero rows in output: {zero_rows.sum().item()} / {M}")
print(f"Zero cols in output: {zero_cols.sum().item()} / {N}")

# Which rows are zero?
zero_row_indices = torch.where(zero_rows)[0]
if len(zero_row_indices) > 0:
    print(f"Zero row indices: {zero_row_indices.tolist()}")
    print(f"  Range: [{zero_row_indices[0].item()}, {zero_row_indices[-1].item()}]")

# Which rows have errors?
row_has_error = (abs_diff > 0.01).any(dim=1)
error_row_indices = torch.where(row_has_error)[0]
if len(error_row_indices) > 0:
    print(f"\nRows with errors: {error_row_indices.tolist()}")
    print(f"  Range: [{error_row_indices[0].item()}, {error_row_indices[-1].item()}]")
    print(f"  Count: {len(error_row_indices)}")

# Which cols have errors?
col_has_error = (abs_diff > 0.01).any(dim=0)
error_col_indices = torch.where(col_has_error)[0]
if len(error_col_indices) > 0:
    print(f"\nCols with errors: first={error_col_indices[0].item()}, last={error_col_indices[-1].item()}, count={len(error_col_indices)}")

# Analyze by 16x16 warp tiles (MFMA 16x16x32)
print("\n=== Error distribution by 16x16 tiles ===")
for m_tile in range(0, M, 16):
    for n_tile in range(0, N, 16):
        tile_errors = (abs_diff[m_tile:m_tile+16, n_tile:n_tile+16] > 0.01).sum().item()
        if tile_errors > 0:
            tile_zeros = (c[m_tile:m_tile+16, n_tile:n_tile+16] == 0).sum().item()
            print(f"  Tile M[{m_tile}:{m_tile+16}] x N[{n_tile}:{n_tile+16}]: {tile_errors}/256 errors, {tile_zeros}/256 zeros")

# Analyze by warp (8 warps for 512 threads)
# For 256x256 block with scattered warp layout, figure out warp mapping
# block_M=256, block_N=256, 8 warps
# Typical MFMA tile per warp: 16x16, so warp_M = 256/warp_tile_M, warp_N = 256/warp_tile_N
print("\n=== Error heatmap (32x32 blocks) ===")
for m_tile in range(0, M, 32):
    row = ""
    for n_tile in range(0, N, 32):
        tile_errors = (abs_diff[m_tile:m_tile+32, n_tile:n_tile+32] > 0.01).sum().item()
        tile_max = abs_diff[m_tile:m_tile+32, n_tile:n_tile+32].max().item()
        if tile_errors == 0:
            row += "  .  "
        else:
            pct = 100 * tile_errors / (32*32)
            row += f"{pct:4.0f}%"
    print(f"M[{m_tile:3d}]: {row}")

# Check if it's a partial computation issue (some k iterations missing)
print("\n=== Checking if errors look like missing k iterations ===")
# If first N elements of certain rows are correct, but rest is wrong
for r in [0, 16, 32, 128, 192, 240]:
    if r < M:
        row_errors = abs_diff[r, :]
        first_error_col = -1
        for col in range(N):
            if row_errors[col] > 0.01:
                first_error_col = col
                break
        if first_error_col >= 0:
            print(f"  Row {r}: first error at col {first_error_col}, max_diff={row_errors.max().item():.4f}")
            print(f"    c[{r},0:8] = {c[r, 0:8].tolist()}")
            print(f"    ref[{r},0:8] = {ref_c[r, 0:8].tolist()}")
        else:
            print(f"  Row {r}: no errors")

# Look at the ratio c/ref_c for non-zero ref elements to check if it's a scaling issue
print("\n=== Value ratio analysis (c / ref_c) for mismatched elements ===")
mask = (abs_diff > 0.01) & (ref_c.abs() > 1.0)
if mask.sum() > 0:
    ratios = (c[mask] / ref_c[mask])
    print(f"  Ratio stats: mean={ratios.mean().item():.4f}, std={ratios.std().item():.4f}")
    print(f"  Ratio range: [{ratios.min().item():.4f}, {ratios.max().item():.4f}]")
    # Check if ratio is consistently ~0 (meaning output is zero where it shouldn't be)
    near_zero = (ratios.abs() < 0.01).sum().item()
    print(f"  Ratios near zero: {near_zero}/{mask.sum().item()}")
