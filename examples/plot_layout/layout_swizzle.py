"""Visualize bank-swizzle layouts for shared memory.

The swizzle layout remaps column indices within each 8-row tile using XOR
to avoid shared memory bank conflicts.

Full-bank  (128B): 3-bit XOR — c_swizzle = c ^ s
Half-bank  (64B):  2-bit XOR — c_swizzle = c ^ (s >> 1)
Quarter-bank (32B): 1-bit XOR — c_swizzle = c ^ (s >> 2)

where s = row % 8 and c = column group index.
"""

import tilelang.language as T
from tilelang.tools import plot_layout
from tvm.tir import FloorDiv, FloorMod

element_size = 16  # float16 = 16 bits
vector_size = 128 // element_size  # = 8 elements per 128-bit vector


# ---- 2D view helpers ----
# The raw swizzle Layout outputs 3D (tc, ts, index) which is hard to visualize.
# Instead, we create a 2D Layout: (i, j) -> (i, swizzled_j), which clearly
# shows the column permutation pattern per row.

def make_full_bank_swizzle_2d(stride, continuous, element_size=16):
    """2D view of full-bank (128B) swizzle: 3-bit XOR, c_swizzle = c ^ s."""
    vs = 128 // element_size
    def forward(i, j):
        s = FloorMod(i, 8)
        tc = FloorDiv(FloorDiv(j, vs), 8)
        c = FloorMod(FloorDiv(j, vs), 8)
        vec = FloorMod(j, vs)
        c_swizzle = c ^ s
        return (i, tc * 8 * vs + c_swizzle * vs + vec)
    return T.Layout([stride, continuous], forward)


def make_half_bank_swizzle_2d(stride, continuous, element_size=16):
    """2D view of half-bank (64B) swizzle: 2-bit XOR, c_swizzle = c ^ (s >> 1)."""
    vs = 128 // element_size
    def forward(i, j):
        s = FloorMod(i, 8)
        tc = FloorDiv(FloorDiv(j, vs), 4)
        c = FloorMod(FloorDiv(j, vs), 4)
        vec = FloorMod(j, vs)
        c_swizzle = c ^ FloorDiv(s, 2)
        return (i, tc * 4 * vs + c_swizzle * vs + vec)
    return T.Layout([stride, continuous], forward)


def make_quarter_bank_swizzle_2d(stride, continuous, element_size=16):
    """2D view of quarter-bank (32B) swizzle: 1-bit XOR, c_swizzle = c ^ (s >> 2)."""
    vs = 128 // element_size
    def forward(i, j):
        s = FloorMod(i, 8)
        tc = FloorDiv(FloorDiv(j, vs), 2)
        c = FloorMod(FloorDiv(j, vs), 2)
        vec = FloorMod(j, vs)
        c_swizzle = c ^ FloorDiv(s, 4)
        return (i, tc * 2 * vs + c_swizzle * vs + vec)
    return T.Layout([stride, continuous], forward)


# ---- Plot the swizzle patterns ----

# 1. Quarter-bank (32B) — 1-bit XOR — 8x16
# Rows 0-3: identity; Rows 4-7: two 8-element halves swap
layout = make_quarter_bank_swizzle_2d(8, 16)
print(f"Quarter-bank swizzle (8x16, fp16): {layout}")
plot_layout(layout, name="swizzle_quarter_8x16")

# 2. Half-bank (64B) — 2-bit XOR — 8x32
layout = make_half_bank_swizzle_2d(8, 32)
print(f"Half-bank swizzle (8x32, fp16): {layout}")
plot_layout(layout, name="swizzle_half_8x32")

# 3. Full-bank (128B) — 3-bit XOR — 8x64
layout = make_full_bank_swizzle_2d(8, 64)
print(f"Full-bank swizzle (8x64, fp16): {layout}")
plot_layout(layout, name="swizzle_full_8x64")

# 4. Full-bank (128B) — multi-tile: 32x64
layout = make_full_bank_swizzle_2d(32, 64)
print(f"Full-bank swizzle (32x64, fp16): {layout}")
plot_layout(layout, name="swizzle_full_32x64")
