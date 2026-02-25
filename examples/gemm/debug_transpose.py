import os
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

M, N, K = 256, 256, 64
kernel = matmul_nt(M, N, K, 256, 256, 64)

a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()

# Check if the result is a per-tile transpose
print("=== Check if c is a per-16x16-tile transpose of ref_c ===")
for m_start in range(0, M, 16):
    for n_start in range(0, N, 16):
        c_tile = c[m_start:m_start+16, n_start:n_start+16]
        ref_tile = ref_c[m_start:m_start+16, n_start:n_start+16]
        ref_tile_T = ref_tile.T  # transpose within tile

        # Check if c_tile matches ref_tile_T
        if torch.allclose(c_tile, ref_tile_T, rtol=1e-2, atol=1e-2):
            pass  # matches transposed
        elif torch.allclose(c_tile, ref_tile, rtol=1e-2, atol=1e-2):
            pass  # matches correct
        else:
            # Neither matches
            diff_normal = (c_tile - ref_tile).abs().max().item()
            diff_transposed = (c_tile - ref_tile_T).abs().max().item()
            print(f"  Tile M[{m_start}:{m_start+16}] x N[{n_start}:{n_start+16}]: "
                  f"normal_diff={diff_normal:.4f}, transposed_diff={diff_transposed:.4f}")

# Count how many tiles match transposed vs normal
normal_ok = 0
transposed_ok = 0
neither = 0
for m_start in range(0, M, 16):
    for n_start in range(0, N, 16):
        c_tile = c[m_start:m_start+16, n_start:n_start+16]
        ref_tile = ref_c[m_start:m_start+16, n_start:n_start+16]
        ref_tile_T = ref_tile.T

        is_normal = torch.allclose(c_tile, ref_tile, rtol=1e-2, atol=1e-2)
        is_transposed = torch.allclose(c_tile, ref_tile_T, rtol=1e-2, atol=1e-2)

        if is_normal and not is_transposed:
            normal_ok += 1
        elif is_transposed and not is_normal:
            transposed_ok += 1
        elif is_normal and is_transposed:
            normal_ok += 1  # both match (e.g., symmetric tile)
        else:
            neither += 1

total_tiles = (M // 16) * (N // 16)
print(f"\nTotal 16x16 tiles: {total_tiles}")
print(f"  Correct (normal): {normal_ok}")
print(f"  Transposed: {transposed_ok}")
print(f"  Neither: {neither}")

# Check a different hypothesis: maybe the C_local to C mapping swaps
# row and col for SOME tiles
# Let's check cross-tile: maybe c[m_start:m_start+16, n_start:n_start+16]
# matches ref_c[n_start:n_start+16, m_start:m_start+16]^T (entire tile from wrong position)
print("\n=== Check if tiles are from wrong position ===")
for m_start in range(0, min(M, 64), 16):
    for n_start in range(0, min(N, 64), 16):
        c_tile = c[m_start:m_start+16, n_start:n_start+16]

        # Check all possible source tiles
        best_match = None
        best_diff = float('inf')
        for m2 in range(0, M, 16):
            for n2 in range(0, N, 16):
                ref_tile = ref_c[m2:m2+16, n2:n2+16]
                diff = (c_tile - ref_tile).abs().max().item()
                if diff < best_diff:
                    best_diff = diff
                    best_match = (m2, n2, False)

                diff_T = (c_tile - ref_tile.T).abs().max().item()
                if diff_T < best_diff:
                    best_diff = diff_T
                    best_match = (m2, n2, True)

        m2, n2, transposed = best_match
        match_str = "TRANSPOSED" if transposed else "normal"
        correct = (m2 == m_start and n2 == n_start and not transposed)
        status = "OK" if correct else "WRONG"
        print(f"  c[{m_start}:{m_start+16}, {n_start}:{n_start+16}] best matches "
              f"ref[{m2}:{m2+16}, {n2}:{n2+16}] ({match_str}, diff={best_diff:.4f}) [{status}]")
