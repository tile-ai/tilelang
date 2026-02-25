"""
Check if any S2R (shared→register) read accesses addresses outside the valid range.

A_shared has 32768 elements (double buffered: 2 × 16384).
For K=64, single iteration, no double buffering needed, so valid range is [0, 16384).

B_shared has 32768 elements, valid range [0, 16384) for single iteration.

Let's check ALL threads' S2R addresses.
"""

block_M = 256
block_N = 256
block_K = 64
num_threads = 512

# A_shared: 2 * 16384 = 32768 elements total
# For K=64 (1 iteration), only buffer 0 is used (no k&1 indexing needed)
# Buffer 0 occupies [0, 16384), buffer 1 occupies [16384, 32768)

# The prologue loads into buffer 0 (no (k+1)&1 offset since there's no k loop)
# The epilogue GEMM reads from... let me check the K=64 kernel

# From the K=64 kernel code:
# S2R for A (line 50):
# A_shared[
#   ((tid & 255) >> 6) * 4096 +    // warp_group * 4096
#   sub_m * 2048 +
#   i_3 * 1024 +
#   (tid & 15) * 64 +
#   swizzle_bits +
#   local_id * 8
# ]
# No double buffer offset (k&1 = 0) since K=64 has only 1 iteration

# Max address for A S2R:
# wg=3: 3*4096=12288, sub_m=1: +2048=14336, i_3=1: +1024=15360
# (tid&15)=15: +960=16320, swizzle max=32+16=48: +48=16368, local_id=1: +8=16376
# With uint4 read (8 elements), reads 16376..16383 → within [0, 16384) ✓

# But wait, what about the scattered warp layout affecting the addresses?
# Let me compute all actual S2R addresses for warp_group 0

def s2r_a_addr(tid, sub_m, i_5, local_id):
    """S2R address for A_local load (K=64 kernel, no double buffer offset)"""
    wg = (tid & 255) >> 6
    lane = tid & 15
    # swizzle bits
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = (((local_id + (tid & 1)) & 1) * 8)
    swizzle = bit0 + bit1 + bit2

    addr = (wg * 4096 + sub_m * 2048 + i_5 * 1024 + lane * 64 + swizzle)
    return addr

def s2r_b_addr(tid, sub_n, j, local_id):
    """S2R address for B_local load (K=64 kernel, no double buffer offset)"""
    n_half = tid >> 8
    lane = tid & 15
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit0 = (((local_id + (tid & 1)) & 1) * 8)
    swizzle = bit0 + bit1 + bit2

    addr = (n_half * 8192 + sub_n * 4096 + j * 1024 + lane * 64 + swizzle)
    return addr

# Check all S2R addresses for A
print("=== A S2R address bounds check ===")
a_max = 0
a_min = float('inf')
a_oob = []
for tid in range(num_threads):
    for sub_m in range(2):
        for i_5 in range(2):
            for local_id in range(2):
                addr = s2r_a_addr(tid, sub_m, i_5, local_id)
                end_addr = addr + 7  # reads 8 elements (uint4)
                a_max = max(a_max, end_addr)
                a_min = min(a_min, addr)
                if end_addr >= 16384:
                    a_oob.append((tid, sub_m, i_5, local_id, addr, end_addr))

print(f"A S2R address range: [{a_min}, {a_max}]")
print(f"Valid range: [0, 16383]")
print(f"Out-of-bounds accesses: {len(a_oob)}")
if a_oob:
    for tid, sub_m, i_5, local_id, addr, end in a_oob[:10]:
        wg = (tid & 255) >> 6
        lane = tid & 15
        print(f"  tid={tid} wg={wg} sub_m={sub_m} i_5={i_5} local_id={local_id} lane={lane}: "
              f"addr={addr}..{end} (OOB by {end - 16383})")

# Check all S2R addresses for B
print("\n=== B S2R address bounds check ===")
b_max = 0
b_min = float('inf')
b_oob = []
for tid in range(num_threads):
    for sub_n in range(2):
        for j in range(4):
            for local_id in range(2):
                addr = s2r_b_addr(tid, sub_n, j, local_id)
                end_addr = addr + 7
                b_max = max(b_max, end_addr)
                b_min = min(b_min, addr)
                if end_addr >= 16384:
                    b_oob.append((tid, sub_n, j, local_id, addr, end_addr))

print(f"B S2R address range: [{b_min}, {b_max}]")
print(f"Valid range: [0, 16383]")
print(f"Out-of-bounds accesses: {len(b_oob)}")
if b_oob:
    for tid, sub_n, j, local_id, addr, end in b_oob[:10]:
        n_half = tid >> 8
        lane = tid & 15
        print(f"  tid={tid} n_half={n_half} sub_n={sub_n} j={j} local_id={local_id} lane={lane}: "
              f"addr={addr}..{end} (OOB by {end - 16383})")

# NOW: Check what data each S2R access gets vs what G2S wrote
# For warp_group 0, the S2R reads from A_shared[0:4096]
# G2S writes A_shared[tid*8 .. tid*8+7] for i_1=0 (first 4096 elements)
# The G2S layout: A_shared[row*64 + swizzled_col]
# where row = tid//8 (within the 64-row block), and K columns are swizzled

# Let me build the G2S write map more carefully
print("\n=== Detailed G2S → S2R mapping for warp_group 0 ===")

def g2s_swizzle(tid):
    """G2S swizzle offset for K dimension"""
    bit0 = ((((tid & 15) >> 3) + (tid & 1)) & 1) * 8
    bit1 = ((((tid & 31) >> 4) + ((tid & 3) >> 1)) & 1) * 16
    bit2 = ((((tid & 63) >> 5) + ((tid & 7) >> 2)) & 1) * 32
    return bit0 + bit1 + bit2

# Build LDS content map for A_shared[0:4096] (i_1=0, first 64 rows)
# LDS addr → (M_row, K_col)
lds_content = {}
for tid in range(512):
    lds_base = tid * 8  # within the i_1=0 block
    row = tid >> 3       # 0..63
    k_start = g2s_swizzle(tid)
    for j in range(8):
        lds_content[lds_base + j] = (row, k_start + j)

# Now check what warp_group 0 S2R reads
# Focus on the problem: rows 8-15 get wrong data, rows 50-63 get garbage
print("\nChecking S2R reads for warp_group 0 threads that MAP to error rows:")
print("(Rows 8-15 are wg=0, sub_m=0, i_5=0, lane=8..15)")
print("(Rows 48-63 are wg=0, sub_m=1, i_5=1, lane=0..15)")

# For the error rows, what threads compute them?
# C store: M = wg*64 + (i_9>>3)*16 + (tid&15)
# For M=8: wg=0, (i_9>>3)=0, tid&15=8 → tid with (tid&255)>>6=0 and tid&15=8
# Thread 8 (in warp 0) or thread 256+8=264 (N-half 1)

for target_row in [0, 8, 16, 32, 48, 49, 50, 56, 63]:
    wg = 0
    sub_m = (target_row % 64) // 32
    i_5 = ((target_row % 64) % 32) // 16
    lane = target_row % 16

    # Find a representative tid
    tid = (wg << 6) | lane  # simplest: bits 4-5 = 0

    print(f"\nRow {target_row}: tid={tid} (wg={wg}, sub_m={sub_m}, i_5={i_5}, lane={lane})")

    for local_id in range(2):
        addr = s2r_a_addr(tid, sub_m, i_5, local_id)
        if addr in lds_content:
            row_read, k_col = lds_content[addr]
            expected_row = target_row
            status = "OK" if row_read == expected_row else f"WRONG (reads row {row_read})"
            print(f"  local_id={local_id}: reads A_shared[{addr}] → (row={row_read}, K={k_col}) [{status}]")
        else:
            # Address might be in a different i_1 block
            block_idx = addr // 4096
            offset = addr % 4096
            print(f"  local_id={local_id}: reads A_shared[{addr}] (block {block_idx}, offset {offset}) - outside block 0")
            # Check if it's in a valid block
            if addr < 16384:
                # Look up in full LDS map
                i_1_block = addr // 4096
                offset_in_block = addr % 4096
                # For i_1=block: G2S writes at i_1*4096 + tid*8
                # So the content at addr was written by the thread where i_1*4096+tid*8 <= addr < i_1*4096+tid*8+8
                # tid_writer = (addr - i_1_block*4096) // 8
                tid_writer = offset_in_block // 8
                elem_idx = offset_in_block % 8
                row_written = i_1_block * 64 + (tid_writer >> 3)
                k_col_written = g2s_swizzle(tid_writer) + elem_idx
                expected_row_actual = wg * 64 + sub_m * 32 + i_5 * 16 + lane
                status = "OK" if row_written == expected_row_actual else f"WRONG (reads row {row_written}, expected {expected_row_actual})"
                print(f"    → i_1={i_1_block}, tid_writer={tid_writer}: (row={row_written}, K={k_col_written}) [{status}]")
