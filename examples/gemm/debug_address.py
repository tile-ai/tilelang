"""
Verify G2S write layout matches S2R read layout in shared memory.

G2S writes A[M,K] global → A_shared[...] LDS
S2R reads A_shared[...] LDS → A_local[...] registers

For the kernel to be correct, the data at each LDS location must be
consistent between write and read.

We'll trace through all threads and check what (M_row, K_col) each thread
writes to each LDS address, and what (M_row, K_col) each thread expects
to read from that same LDS address.
"""

# Parameters
block_M = 256
block_K = 64
num_threads = 512
K_stride = 64  # K=64 case (single iteration)

# === G2S write: what does each thread write to which LDS address? ===
# LDS address: A_shared[i_1*4096 + tid*8 + j] for j=0..7
# Global address: A[i_1*4096 + (tid>>3)*K_stride + swizzle(tid)]
# The data written is A[row, col..col+7]

def swizzle_offset(tid):
    """Compute the swizzle K-offset for a given thread."""
    bit0 = ((((tid & 15) >> 3) + (tid & 1)) & 1) * 8
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    return bit0 + bit1 + bit2

# Build G2S write map: lds_addr -> (global_row, global_col_start)
g2s_map = {}  # lds_flat_addr -> (M_row, K_col_start)

for i_1 in range(4):
    for tid in range(num_threads):
        lds_base = i_1 * 4096 + tid * 8

        # Global: row = i_1 * (4096 // K_stride) + (tid >> 3)
        # But global addr = i_1*4096 + (tid>>3)*K_stride + swizzle
        # Since K_stride=64 and global has flat layout:
        # flat_global = i_1*4096 + (tid>>3)*64 + swizzle(tid)
        # row = flat_global // K_stride = i_1*64 + (tid>>3)
        # (only if swizzle < 64, which it always is since max swizzle = 32+16+8=56 and we add up to 7)

        global_row = i_1 * 64 + (tid >> 3)
        global_k_col = swizzle_offset(tid)

        for j in range(8):
            lds_addr = lds_base + j
            g2s_map[lds_addr] = (global_row, global_k_col + j)


# === S2R read: what does each thread read from which LDS address? ===
# A_shared[
#   ((tid & 255) >> 6) * 4096 +     // warp_group (0-3)
#   sub_m * 2048 +                   // 0 or 1
#   i_5 * 1024 +                     // 0 or 1
#   (tid & 15) * 64 +                // lane row position (0..15) * 64
#   swizzle_bits +                   // same swizzle pattern
#   local_id * 8                     // 0 or 1 (selects 8-element half)
# ]
# Each read is 8 bf16 (uint4 = 16 bytes)

def s2r_swizzle_offset(tid, local_id):
    """Swizzle for S2R read."""
    bit0 = (((((tid & 15) >> 3) + (tid & 1)) & 1) * 32)  # Wait, let me re-read from the code...
    # From line 50 of K=64 kernel:
    # (((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32)
    # (((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16)
    # (((local_id + (tid & 1)) & 1) * 8)
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = (((local_id + (tid & 1)) & 1) * 8)
    return bit0 + bit1 + bit2

# Build S2R read map for EACH MFMA tile
# The S2R read tells us: thread tid, at (sub_m, i_5, local_id) reads from LDS to get
# a portion of the A matrix data for its MFMA computation.
#
# For an MFMA 16x16x32 instruction:
#   - 16 M-rows: determined by (tid & 15)
#   - 32 K-elements packed into bf16x8 (with k_pack=2, there are 2 kp iterations)
#   - The warp_group selects which 64 rows of M this warp processes
#   - sub_m selects upper/lower 32 rows within the 64-row group
#   - i_5 selects upper/lower 16 rows within the 32-row group

# Let's check: for warp_group 0 (tid & 255 >> 6 == 0), sub_m=0, i_5=0, local_id=0
# the LDS address is: 0*4096 + 0*2048 + 0*1024 + (tid&15)*64 + swizzle + 0*8
# This should read from the first 16 rows of A, K columns determined by swizzle.
#
# The G2S wrote these 16 rows at: LDS addresses tid*8 for tid=0..127 (rows 0..15, each row spans 8*8=64 elements -> but 64/8=8 threads per row)
# Wait: G2S has 512 threads. For i_1=0: LDS[tid*8..tid*8+7] for tid=0..511
# That's 512*8 = 4096 elements. Row = tid//8 (0..63), col within row = (tid%8)*8 .. (tid%8)*8+7 mapped via swizzle

# Actually the G2S layout in LDS is simply:
#   LDS[row * 64 + col] = A[global_row, global_col]
# where row = tid//8 (local row 0..63 for i_1=0), and col is thread-dependent via swizzle

# Let me verify: for tid=0, i_1=0:
#   LDS addr = 0*4096 + 0*8 = 0..7
#   Global = A[row=0, col=swizzle(0)..swizzle(0)+7]
#   swizzle(0) = (((0>>3)+(0&1))&1)*8 + (((0>>4)+(0>>1)&1)&1)*16 + (((0>>5)+(0>>2)&1)&1)*32
#              = 0 + 0 + 0 = 0
#   So LDS[0..7] = A[0, 0..7] ✓

# For tid=1, i_1=0:
#   LDS addr = 0*4096 + 1*8 = 8..15
#   Global = A[row=0, col=swizzle(1)..swizzle(1)+7]
#   swizzle(1) = (((0)+(1))&1)*8 + (((0)+(0))&1)*16 + (((0)+(0))&1)*32
#              = 1*8 + 0 + 0 = 8
#   So LDS[8..15] = A[0, 8..15] ✓

# For tid=8, i_1=0:
#   LDS addr = 0*4096 + 8*8 = 64..71
#   Global = A[row=1, col=swizzle(8)..swizzle(8)+7]
#   swizzle(8) = (((1)+(0))&1)*8 + (((0)+(0))&1)*16 + (((0)+(2))&1)*32
#              = 1*8 + 0 + 0 = 8
#   So LDS[64..71] = A[1, 8..15]
#   But plain row-major would be LDS[64..71] = A[1, 0..7]
#   The swizzle makes row 1 start at col 8 instead of col 0. This is the XOR swizzle pattern.

# Now for S2R read: warp_group=0, sub_m=0, i_5=0, tid with (tid&15)=0, local_id=0:
#   LDS addr = 0*4096 + 0*2048 + 0*1024 + 0*64 + swizzle(tid,0) + 0*8
#   = swizzle(tid, 0)
#   s2r_swizzle(tid, 0) = bit0 + bit1 + bit2
#   For tid=0: bit2 = (((0>>5)+(0>>2))&1)*32 = 0; bit1 = (((0>>4)+(0>>1))&1)*16 = 0; bit0 = ((0+(0&1))&1)*8 = 0
#   So reads from LDS[0..7] → expects A[0, 0..7] ← matches G2S ✓

# Let's check a potentially problematic case.
# For S2R with tid=0, sub_m=0, i_5=0, local_id=1:
#   LDS addr = 0 + 0 + 0 + 0 + s2r_swizzle(0, 1)
#   bit0 = ((1 + 0) & 1) * 8 = 8
#   So reads from LDS[8..15]
#   G2S wrote LDS[8..15] = A[0, 8..15] (from tid=1)
#   S2R expects this to be A[0, 8..15] → the second 8-element block of K ✓

# The key question: does the S2R read layout MATCH the G2S write layout?
# Let's systematically check for warp_group 0.

print("=== Checking G2S write vs S2R read consistency ===")
print("Checking warp_group 0 (first 64 rows)")

errors = []

# Check for every S2R read access
for sub_m in range(2):
    for i_5 in range(2):
        for lane in range(16):  # (tid & 15)
            for local_id in range(2):
                # Construct a representative tid for this lane in warp_group 0
                # warp_group 0: (tid & 255) >> 6 == 0 → tid & 255 in [0, 63]
                # We need (tid & 15) == lane, so tid = warp_offset + lane
                # where warp_offset is a multiple of 64 that keeps us in warp_group 0
                # For warp_group 0: warp_offset = 0 (or 256 for the N>128 half)

                for tid_base in [0, 256]:  # both N-halves
                    # Need (tid & 255) >> 6 == 0 → (tid & 255) in [0,63]
                    # And (tid & 15) == lane
                    # So tid_base must be 0 or 256, and we need the warp bits (bits 4-5) to be 0
                    # Actually (tid & 255) >> 6 gives bits 6,7 of (tid&255)
                    # For tid_base=0: tid in [0..63] → bits 6,7 = 0 ✓
                    # For tid_base=256: (256 & 255) = 0, so >> 6 = 0 ✓

                    # We also need the swizzle bits to be determined by the full tid
                    # Let's check tid = tid_base + lane (simplest case, bits 4-5 = 0)
                    tid = tid_base + lane

                    # S2R LDS address
                    warp_group = (tid & 255) >> 6  # should be 0
                    s2r_addr = (warp_group * 4096 + sub_m * 2048 + i_5 * 1024 +
                               (tid & 15) * 64 + s2r_swizzle_offset(tid, local_id))

                    # What does S2R expect at this address?
                    # M_row = warp_group*64 + sub_m*32 + i_5*16 + lane
                    expected_row = warp_group * 64 + sub_m * 32 + i_5 * 16 + lane
                    # K_col = s2r_swizzle determines which 8-element block
                    # But we need to know what K_col the MFMA expects

                    # What did G2S actually write at this address?
                    for j in range(8):
                        addr = s2r_addr + j
                        if addr in g2s_map:
                            g2s_row, g2s_col = g2s_map[addr]
                            # Check if g2s_row matches expected_row
                            if g2s_row != expected_row:
                                errors.append({
                                    'tid': tid,
                                    'sub_m': sub_m,
                                    'i_5': i_5,
                                    'lane': lane,
                                    'local_id': local_id,
                                    'j': j,
                                    'lds_addr': addr,
                                    'expected_row': expected_row,
                                    'actual_row': g2s_row,
                                    'actual_col': g2s_col,
                                })
                        else:
                            errors.append({
                                'tid': tid,
                                'lds_addr': addr,
                                'error': 'address not in g2s_map'
                            })

if errors:
    print(f"FOUND {len(errors)} mismatches!")
    # Show first few
    for e in errors[:20]:
        print(f"  {e}")

    # Summarize which rows are affected
    affected_rows = set()
    for e in errors:
        if 'expected_row' in e:
            affected_rows.add(e['expected_row'])
    print(f"\nAffected M rows: {sorted(affected_rows)}")
else:
    print("All warp_group 0 reads match G2S writes! Address mapping is consistent.")

# Now check ALL warp groups
print("\n=== Checking ALL warp_groups ===")
all_errors = {0: 0, 1: 0, 2: 0, 3: 0}

for sub_m in range(2):
    for i_5 in range(2):
        for lane in range(16):
            for local_id in range(2):
                for wg in range(4):
                    for n_half in range(2):
                        # Construct tid: need (tid&255)>>6 == wg, (tid&15)==lane, tid>>8 == n_half
                        # bits: [8] = n_half, [7:6] = wg, [5:4] = XX, [3:0] = lane
                        # For simplicity, set bits 4-5 to 0
                        tid = (n_half << 8) | (wg << 6) | lane

                        warp_group_check = (tid & 255) >> 6
                        assert warp_group_check == wg

                        s2r_addr = (wg * 4096 + sub_m * 2048 + i_5 * 1024 +
                                   lane * 64 + s2r_swizzle_offset(tid, local_id))

                        expected_row = wg * 64 + sub_m * 32 + i_5 * 16 + lane

                        for j in range(8):
                            addr = s2r_addr + j
                            if addr in g2s_map:
                                g2s_row, g2s_col = g2s_map[addr]
                                if g2s_row != expected_row:
                                    all_errors[wg] += 1

for wg in range(4):
    print(f"  Warp group {wg}: {all_errors[wg]} mismatches")

# Also check with different bits 4-5 values
print("\n=== Checking with ALL thread IDs (full 512) ===")
wg_errors = {0: 0, 1: 0, 2: 0, 3: 0}

for tid in range(512):
    wg = (tid & 255) >> 6
    for sub_m in range(2):
        for i_5 in range(2):
            for local_id in range(2):
                s2r_addr = (wg * 4096 + sub_m * 2048 + i_5 * 1024 +
                           (tid & 15) * 64 + s2r_swizzle_offset(tid, local_id))

                expected_row = wg * 64 + sub_m * 32 + i_5 * 16 + (tid & 15)

                for j in range(8):
                    addr = s2r_addr + j
                    if addr in g2s_map:
                        g2s_row, g2s_col = g2s_map[addr]
                        if g2s_row != expected_row:
                            wg_errors[wg] += 1
                            if wg_errors[wg] <= 3:
                                print(f"  tid={tid} wg={wg} sub_m={sub_m} i_5={i_5} local_id={local_id} j={j}: expected row {expected_row}, got row {g2s_row} (col {g2s_col}) at LDS[{addr}]")

for wg in range(4):
    status = "ERROR" if wg_errors[wg] > 0 else "OK"
    print(f"  Warp group {wg}: {wg_errors[wg]} mismatches [{status}]")
