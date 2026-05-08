import tilelang.language as T
from tilelang.tools import plot_layout

# --- Example 1: Simple 2D Transpose (4x4) ---
transpose_layout = T.Layout([4, 4], lambda i, j: (j, i))
print("Transpose 4x4:", transpose_layout)
plot_layout(transpose_layout, name="transpose_4x4")

# --- Example 2: Larger Transpose (8x8) ---
transpose_8x8 = T.Layout([8, 8], lambda i, j: (j, i))
print("Transpose 8x8:", transpose_8x8)
plot_layout(transpose_8x8, name="transpose_8x8")

# --- Example 3: 3D → 2D reshape + transpose ---
# (i, j, k) with shape [2, 4, 8] → (k, i*4+j)
reshape_layout = T.Layout([2, 4, 8], lambda i, j, k: (k, i * 4 + j))
print("Reshape 3D [2,4,8] -> [8,8]:", reshape_layout)
plot_layout(reshape_layout, name="reshape_3d_to_2d")

# --- Example 4: Interleave layout ---
# Even rows from first half, odd rows from second half
interleave = T.Layout([8, 4], lambda i, j: (i % 4 * 2 + i // 4, j))
print("Interleave [8,4]:", interleave)
plot_layout(interleave, name="interleave_8x4")
