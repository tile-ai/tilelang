"""
For v_mfma_f32_16x16x32_bf16:
  Lane l provides: src[l%16, (l//16)*8..(l//16)*8+7]
  - lanes 0-15:  K[0:8]
  - lanes 16-31: K[8:16]
  - lanes 32-47: K[16:24]
  - lanes 48-63: K[24:32]

For k_pack=2, kp=0 processes K[0:32], kp=1 processes K[32:64].
A_local[i_6*2+kp] (as bfloat16x8_vec) maps to:
  i_6=0, kp=0: A_local[0:8]   (S2R: i_5=0, local_id=0)
  i_6=0, kp=1: A_local[8:16]  (S2R: i_5=0, local_id=1)
  i_6=1, kp=0: A_local[16:24] (S2R: i_5=1, local_id=0)
  i_6=1, kp=1: A_local[24:32] (S2R: i_5=1, local_id=1)

So MFMA with kp=0 uses data from local_id=0,
   MFMA with kp=1 uses data from local_id=1.

For kp=0, lane l should provide K[(l//16)*8..(l//16)*8+7]
This data should come from A_local[i_6*2+0] = S2R(i_5=i_6, local_id=0).

Question: Does S2R(i_5, local_id=0) at lane l provide K[(l//16)*8..(l//16)*8+7]?
"""

def g2s_swizzle(tid):
    """G2S XOR swizzle for K offset"""
    bit0 = ((((tid & 15) >> 3) + (tid & 1)) & 1) * 8
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    return bit0 + bit1 + bit2

def s2r_addr(tid, sub_m, i_5, local_id):
    """S2R address within A_shared (no double buffer offset)"""
    wg = (tid & 255) >> 6
    lane = tid & 15
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = (((local_id + (tid & 1)) & 1) * 8)
    return wg * 4096 + sub_m * 2048 + i_5 * 1024 + lane * 64 + bit0 + bit1 + bit2

# Build G2S map: lds_addr → (row, K_col)
g2s_map = {}
for tid_w in range(512):
    lds_base = tid_w * 8  # within i_1 block
    row = tid_w >> 3
    k_start = g2s_swizzle(tid_w)
    for j in range(8):
        # addr within the 4096-element block
        g2s_map[lds_base + j] = (row, k_start + j)

# Now check: for wavefront 0 (threads 0-63), what K columns does each lane get?
print("=== Lane-to-K mapping verification (wavefront 0, wg=0) ===")
print("For MFMA 16x16x32: lane l expects K[(l//16)*8..(l//16)*8+7]")
print()

# Check sub_m=0, i_5=0 (first 16 M rows), local_id=0 (kp=0, first 32 K)
print("sub_m=0, i_5=0, local_id=0 (kp=0):")
mismatches = 0
for lane in range(64):
    tid = lane  # wavefront 0
    addr = s2r_addr(tid, sub_m=0, i_5=0, local_id=0)
    # addr is within [0, 4096) for wg=0

    # What does G2S have at this address?
    g2s_row, g2s_k = g2s_map[addr]

    # What K-column does MFMA expect from this lane?
    mfma_k_start = (lane // 16) * 8

    expected_m_row = tid & 15  # MFMA row = lane % 16

    # Check
    row_ok = (g2s_row == expected_m_row)
    k_ok = (g2s_k == mfma_k_start)

    if not row_ok or not k_ok:
        mismatches += 1
        print(f"  lane={lane:2d} (tid={tid:3d}): addr={addr:4d} → G2S(row={g2s_row:2d}, K={g2s_k:2d}) "
              f"vs MFMA expects(row={expected_m_row:2d}, K={mfma_k_start:2d}) "
              f"{'ROW_WRONG' if not row_ok else ''} {'K_WRONG' if not k_ok else ''}")

if mismatches == 0:
    print("  All lanes match! ✓")
else:
    print(f"  {mismatches}/64 lanes have mismatches!")

# Also check local_id=1 (kp=1, second 32 K elements)
print()
print("sub_m=0, i_5=0, local_id=1 (kp=1):")
mismatches = 0
for lane in range(64):
    tid = lane
    addr = s2r_addr(tid, sub_m=0, i_5=0, local_id=1)
    g2s_row, g2s_k = g2s_map[addr]

    # For kp=1, K offset is 32 + (lane//16)*8
    mfma_k_start = 32 + (lane // 16) * 8
    expected_m_row = tid & 15

    row_ok = (g2s_row == expected_m_row)
    k_ok = (g2s_k == mfma_k_start)

    if not row_ok or not k_ok:
        mismatches += 1
        print(f"  lane={lane:2d} (tid={tid:3d}): addr={addr:4d} → G2S(row={g2s_row:2d}, K={g2s_k:2d}) "
              f"vs MFMA expects(row={expected_m_row:2d}, K={mfma_k_start:2d}) "
              f"{'ROW_WRONG' if not row_ok else ''} {'K_WRONG' if not k_ok else ''}")

if mismatches == 0:
    print("  All lanes match! ✓")
else:
    print(f"  {mismatches}/64 lanes have mismatches!")

# Check for the full pattern
print()
print("=== Complete check: all wavefronts, all S2R parameters ===")
total_errors = 0
for wf in range(8):  # 8 wavefronts
    wf_errors = 0
    for sub_m in range(2):
        for i_5 in range(2):
            for local_id in range(2):
                for lane in range(64):
                    tid = wf * 64 + lane
                    wg = (tid & 255) >> 6
                    addr = s2r_addr(tid, sub_m, i_5, local_id)

                    # Get the i_1 block this address falls into
                    block = addr // 4096
                    offset = addr % 4096

                    # Look up what G2S wrote
                    g2s_row, g2s_k = g2s_map[offset]  # row within the block
                    actual_row = block * 64 + g2s_row

                    # Expected M row
                    expected_m = wg * 64 + sub_m * 32 + i_5 * 16 + (tid & 15)

                    # Expected K column
                    if local_id == 0:
                        expected_k = (lane // 16) * 8  # kp=0: K[0:32]
                    else:
                        expected_k = 32 + (lane // 16) * 8  # kp=1: K[32:64]

                    if actual_row != expected_m or g2s_k != expected_k:
                        wf_errors += 1
                        if wf_errors <= 3:
                            print(f"  wf={wf} tid={tid:3d} lane={lane:2d} wg={wg} sub_m={sub_m} i_5={i_5} lid={local_id}: "
                                  f"row {actual_row} vs {expected_m}, K {g2s_k} vs {expected_k}")

    status = "OK" if wf_errors == 0 else f"{wf_errors} ERRORS"
    total_errors += wf_errors
    print(f"Wavefront {wf} (threads {wf*64}-{wf*64+63}): {status}")

print(f"\nTotal errors: {total_errors}")
