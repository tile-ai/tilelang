"""
Re-check the lane-to-K mapping with the ACTUAL generated S2R addresses from the new kernel.

The generated S2R code for A (line 50 of K=64 kernel):
A_shared[
    (((tid & 255) >> 6) * 4096) +     // warp_group
    (sub_m * 2048) +
    (i_3 * 1024) +
    ((tid & 15) * 64) +
    (((((tid & 7) >> 2) + local_id) & 1) * 32) +         // bit2 (weight 32)
    (((((tid & 63) >> 5) + ((tid & 3) >> 1)) & 1) * 16) + // bit1 (weight 16)
    (((((tid & 31) >> 4) + (tid & 1)) & 1) * 8)           // bit0 (weight 8)
]

Note: local_id here is 0..1 (vectorized), and the actual local_id values from the layout
function that got vectorized are:
  loop local_id=0 → layout local_id values 0..7 (vectorized read of 8 elements)
  loop local_id=1 → layout local_id values 8..15 (vectorized read of 8 elements)

The uint4 read loads 8 consecutive bf16 starting at the computed address.
So the address uses the FIRST element's coordinates (local_id_base = loop_local_id * 8).
"""

def g2s_swizzle(tid):
    """G2S XOR swizzle for K offset"""
    bit0 = ((((tid & 15) >> 3) + (tid & 1)) & 1) * 8
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    return bit0 + bit1 + bit2

def new_s2r_a_addr(tid, sub_m, i_5, local_id_loop):
    """NEW S2R address from the generated kernel after fix.
    local_id_loop is 0 or 1 (the vectorized loop variable)."""
    wg = (tid & 255) >> 6
    lane = tid & 15
    # From the generated code (line 50):
    bit2 = ((((tid & 7) >> 2) + local_id_loop) & 1) * 32
    bit1 = ((((tid & 63) >> 5) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = ((((tid & 31) >> 4) + (tid & 1)) & 1) * 8
    return wg * 4096 + sub_m * 2048 + i_5 * 1024 + lane * 64 + bit0 + bit1 + bit2

# Build G2S map
g2s_map = {}
for tid_w in range(512):
    lds_base = tid_w * 8
    row = tid_w >> 3
    k_start = g2s_swizzle(tid_w)
    for j in range(8):
        g2s_map[lds_base + j] = (row, k_start + j)

# Check lane-to-K mapping for wavefront 0
print("=== NEW S2R Lane-to-K mapping (wavefront 0, wg=0) ===")
print()

# Check sub_m=0, i_5=0, local_id_loop=0 (maps to A_local[0:8], used by kp=0)
print("sub_m=0, i_5=0, local_id_loop=0 (kp=0):")
mismatches = 0
for lane in range(64):
    tid = lane
    addr = new_s2r_a_addr(tid, sub_m=0, i_5=0, local_id_loop=0)
    g2s_row, g2s_k = g2s_map[addr]
    mfma_k_start = (lane // 16) * 8
    expected_m_row = tid & 15
    row_ok = (g2s_row == expected_m_row)
    k_ok = (g2s_k == mfma_k_start)
    if not row_ok or not k_ok:
        mismatches += 1
        print(f"  lane={lane:2d}: addr={addr:4d} → G2S(row={g2s_row:2d}, K={g2s_k:2d}) "
              f"vs MFMA expects(row={expected_m_row:2d}, K={mfma_k_start:2d})")
if mismatches == 0:
    print("  All lanes match! ✓")
else:
    print(f"  {mismatches}/64 MISMATCHES")

# Check local_id_loop=1 (maps to A_local[8:16], used by kp=1)
print()
print("sub_m=0, i_5=0, local_id_loop=1 (kp=1):")
mismatches = 0
for lane in range(64):
    tid = lane
    addr = new_s2r_a_addr(tid, sub_m=0, i_5=0, local_id_loop=1)
    g2s_row, g2s_k = g2s_map[addr]
    mfma_k_start = 32 + (lane // 16) * 8
    expected_m_row = tid & 15
    row_ok = (g2s_row == expected_m_row)
    k_ok = (g2s_k == mfma_k_start)
    if not row_ok or not k_ok:
        mismatches += 1
        print(f"  lane={lane:2d}: addr={addr:4d} → G2S(row={g2s_row:2d}, K={g2s_k:2d}) "
              f"vs MFMA expects(row={expected_m_row:2d}, K={mfma_k_start:2d})")
if mismatches == 0:
    print("  All lanes match! ✓")
else:
    print(f"  {mismatches}/64 MISMATCHES")

# Also show what the OLD swizzle was for comparison
print()
print("=== OLD S2R swizzle (for reference) ===")
def old_s2r_a_addr(tid, sub_m, i_5, local_id_loop):
    wg = (tid & 255) >> 6
    lane = tid & 15
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = (((local_id_loop + (tid & 1)) & 1) * 8)
    return wg * 4096 + sub_m * 2048 + i_5 * 1024 + lane * 64 + bit0 + bit1 + bit2

print("OLD sub_m=0, i_5=0, local_id_loop=0 (kp=0):")
for lane in [0, 16, 32, 48]:
    addr_old = old_s2r_a_addr(lane, 0, 0, 0)
    addr_new = new_s2r_a_addr(lane, 0, 0, 0)
    g2s_row_old, g2s_k_old = g2s_map[addr_old]
    g2s_row_new, g2s_k_new = g2s_map[addr_new]
    print(f"  lane={lane:2d}: OLD addr={addr_old:4d}→K={g2s_k_old:2d}, NEW addr={addr_new:4d}→K={g2s_k_new:2d}, "
          f"MFMA wants K={((lane//16)*8):2d}")
