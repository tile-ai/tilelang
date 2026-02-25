"""
Check the MFMA operand mapping and C store address.

In the generated kernel:
  mfma(B_local[j*2+kp], A_local[i*2+kp], C_local[sub_m*16+i*8+sub_n*4+j])

__builtin_amdgcn_mfma_f32_16x16x32_bf16(src_a, src_b, src_c)
- src_a = first operand → contributes to M (rows) of result
- src_b = second operand → contributes to N (cols) of result
- Result: C[M,N] += A[M,K] * B[N,K]^T

In the kernel:
- src_a = B_local → so B contributes to M (rows)
- src_b = A_local → so A contributes to N (cols)

This means the MFMA computes: C[m,n] += B_data[m,k] * A_data[n,k]

But the GEMM should be: C[m,n] = A[m,:] @ B[n,:]^T (NT layout)
                        = sum_k A[m,k] * B[n,k]

So the MFMA operand order SWAPS M and N:
- What MFMA thinks is "M" is actually indexed by B (N tiles)
- What MFMA thinks is "N" is actually indexed by A (M tiles)

This could be OK IF the C_local indexing and C store address account for this swap.

Let me trace through the full flow:
"""

# Kernel structure:
# 512 threads, block_M=256, block_N=256, block_K=64
#
# S2R for A:
#   warp_group = (tid & 255) >> 6   → 0..3 (selects M block of 64 rows)
#   sub_m = 0..1                     → selects 32-row half
#   i_5 = 0..1                       → selects 16-row MFMA tile within half
#   lane = tid & 15                  → row within 16-row tile
#   → A_local has data for M rows: [wg*64 + sub_m*32 + i_5*16 + lane, K_cols]
#   → A_local[i_5*16 + local_id*8 .. +7] (32 elements total for 2 i_5 * 2 local_id * 8)
#
# S2R for B:
#   n_half = tid >> 8               → 0..1 (selects N block of 128 cols)
#   sub_n = 0..1                     → selects 64-col half
#   j = 0..3                         → selects 16-col MFMA tile within half
#   lane = tid & 15                  → col within 16-col tile
#   → B_local has data for N cols: [n_half*128 + sub_n*64 + j*16 + lane_n_pos, K_cols]
#   → B_local[j*16 + local_id*8 .. +7] (64 elements total for 4 j * 2 local_id * 8)
#
# MFMA call:
#   C_local[sub_m*16 + i_6*8 + sub_n*4 + j_1] = mfma(
#       B_local[j_1*2+kp],    // src_a → M dimension in MFMA
#       A_local[i_6*2+kp],    // src_b → N dimension in MFMA
#       C_local[sub_m*16 + i_6*8 + sub_n*4 + j_1])
#
# So C_local index = sub_m*16 + i_6*8 + sub_n*4 + j_1
#   - sub_m ranges 0..1 (stride 16) — related to A's M tiles
#   - i_6 ranges 0..1 (stride 8) — related to A_local indexing
#   - sub_n ranges 0..1 (stride 4) — related to B's N tiles
#   - j_1 ranges 0..3 (stride 1) — related to B_local indexing
#
# But the MFMA's M dimension comes from B (j_1), and N from A (i_6).
# The C_local layout has:
#   - Outer: sub_m (A's M) → stride 16
#   - Next: i_6 (A's sub-tile) → stride 8
#   - Next: sub_n (B's N) → stride 4
#   - Inner: j_1 (B's sub-tile) → stride 1
#
# C store (line 80 in K=64 kernel, line 130 in K=1024 kernel):
# C[
#   ((tid & 255) >> 6) * 16384 +      // warp_group * (64*256) → M block
#   (i_9 >> 3) * 4096 +               // (i_9 >> 3) * (16*256) → sub-M
#   (tid & 15) * 256 +                // lane_M * N_stride
#   (tid >> 8) * 128 +                // n_half * 128 → N block
#   (i_9 & 7) * 16 +                  // N tile offset
#   ((tid & 63) >> 4) * 4             // lane_N offset
# ]
#
# i_9 ranges 0..31, so:
#   i_9 >> 3 = 0..3 → sub_M index (maps to C_local[i*8] part? or sub_m*16+i*8?)
#   i_9 & 7 = 0..7 → N tile index (maps to sub_n*4+j part)
#
# C_local[i_9 * 4] → each i_9 has 4 float32 elements (one mfma result)
# C_local index = i_9 * 4
# We need i_9 to map to the same C_local index as (sub_m*16 + i_6*8 + sub_n*4 + j_1)
# With i_9 = sub_m*8 + i_6*4 + sub_n*2 + j_1... no wait:
# sub_m*16 + i_6*8 + sub_n*4 + j_1 ranges 0..31
# And C_local has 32 float4 entries (128 floats).
# The i_9 loop iterates i_9 from 0..31.
# So C_local[i_9 * 4] corresponds to the tile at index i_9.
#
# Decomposing i_9:
#   i_9 >> 3 = i_9 / 8 → 0..3 → this encodes sub_m*2 + i_6 (since sub_m has stride 16/4=4? no...)
#   Wait: sub_m*16 + i_6*8 + sub_n*4 + j_1
#   = (sub_m*2 + i_6) * 8 + (sub_n*4 + j_1)
#   So index / 8 = sub_m*2 + i_6 → 0..3
#   And index % 8 = sub_n*4 + j_1 → 0..7
#
# C store uses:
#   i_9 >> 3 → sub_m*2 + i_6 → M sub-tile index (0..3) → offset by 4096 = 16 rows * 256 cols
#   i_9 & 7 → sub_n*4 + j_1 → N sub-tile index (0..7) → offset by 16 cols
#
# C store M position:
#   M = (warp_group * 64) + (i_9>>3)*16 + (tid & 15)
#   = wg*64 + (sub_m*2+i_6)*16 + lane
#   = wg*64 + sub_m*32 + i_6*16 + lane
#
# C store N position:
#   N = (tid>>8)*128 + (i_9&7)*16 + ((tid&63)>>4)*4
#   = n_half*128 + (sub_n*4+j_1)*16 + lane_N*4
#   Hmm, (sub_n*4+j_1)*16: sub_n*64 + j_1*16
#   And lane_N = (tid & 63) >> 4 → bits 4,5 of tid → 0..3

# Now the key question: in the MFMA call, what does each C_local element represent?
#
# MFMA src_a = B_local[j_1*2+kp] → this is B's N-tile j_1 data
#   B data is from: n_half*128 + sub_n*64 + j_1*16 + lane_N_position
#   So MFMA's "M" dimension (from src_a/first operand) = B's N position
#
# MFMA src_b = A_local[i_6*2+kp] → this is A's M-tile i_6 data
#   A data is from: wg*64 + sub_m*32 + i_6*16 + lane_M_position
#   So MFMA's "N" dimension (from src_b/second operand) = A's M position
#
# MFMA result C_mfma[m, n] = sum_k B[m_idx, k] * A[n_idx, k]
# where m_idx is from B's tile and n_idx is from A's tile.
#
# This means: C_local[sub_m*16+i_6*8+sub_n*4+j_1] stores the MFMA result where:
#   - MFMA rows correspond to B's N dimension
#   - MFMA cols correspond to A's M dimension
#
# So C_local[...][mfma_m, mfma_n] = C_correct[A_M, B_N]
#   only if mfma_m → B_N and mfma_n → A_M
#
# In MFMA 16x16x32_bf16 on gfx9:
#   - 64 lanes (one warp = 64 threads)
#   - Result: 4 float32 per lane → 4x16x16/64 = 4 elements/lane
#   - Lane mapping: lane_id → (row, col) in 16x16 result
#   - row = lane_id % 16, col = (lane_id // 16) (groups of 16 lanes map to 4 columns)
#   - Each lane holds result[row, col*4 .. col*4+3]? Or result[row, col]?
#
# For v_mfma_f32_16x16x32_bf16:
#   4 output VGPRs per lane
#   Lane mapping: lane = tid % 64 within the warp
#   Output mapping: dst[i] for i=0..3
#     row = lane % 16
#     col = (lane // 16) * 4 + i
#   So dst[i] = result[lane%16, (lane//16)*4 + i]
#   The (lane//16) gives 0..3 (since lane is 0..63), so col = group*4+i spans 0..15

# In the generated C store:
#   M_store = wg*64 + (sub_m*2+i_6)*16 + (tid & 15)
#   N_store = n_half*128 + (sub_n*4+j_1)*16 + ((tid&63)>>4)*4  ← but then + which of the 4?
#
# Wait, the store writes uint2 (4 bf16 = 2 uint32 = 8 bytes) at once:
#   C_local_cast[0..3] ← converted from C_local[i_9*4 .. i_9*4+3]
#   Store at: C[offset] where offset includes ((tid&63)>>4)*4
#   And the 4 bf16 values are written consecutively → N positions: base, base+1, base+2, base+3
#
# So N_store for the 4 values = n_half*128 + (sub_n*4+j_1)*16 + ((tid&63)>>4)*4 + {0,1,2,3}
#
# ((tid&63)>>4) = lane // 16 within the 64-thread warp → 0..3
# So N = n_half*128 + sub_n*64 + j_1*16 + (lane//16)*4 + {0..3}
# This covers all 16 N positions within the MFMA tile: (lane//16)*4+i for lane//16=0..3, i=0..3

# Similarly, M_store:
#   wg*64 + sub_m*32 + i_6*16 + (tid&15)
#   tid&15 = lane % 16 → 0..15
#   This gives 16 M positions within the MFMA tile

# Now, what does the MFMA actually compute?
#
# The MFMA result at lane tid, dst[i]:
#   row = tid % 16
#   col = (tid%64 // 16) * 4 + i
#   result[row, col] = sum_k src_a[row, k] * src_b[col, k]
#
# Where:
#   src_a = B_local[j_1*2+kp] → B data → conceptually B[B_N_pos, K]
#   src_b = A_local[i_6*2+kp] → A data → conceptually A[A_M_pos, K]
#
# So: result[row, col] = sum_k B[row_in_B_tile, k] * A[col_in_A_tile, k]
#
# And we store this at:
#   M_store = wg*64 + sub_m*32 + i_6*16 + row    (row maps to A_M via i_6 and lane)
#   N_store = n_half*128 + sub_n*64 + j_1*16 + col (col maps to B_N via j_1 and lane group)
#
# But the MFMA computed:
#   result[row, col] = sum_k B[row_of_B_tile, k] * A[col_of_A_tile, k]
#
# The B tile (src_a) corresponds to B[sub_n*64+j_1*16 + lane_pos, K]
#   → row_of_B_tile maps to which N position? The 16 rows of the MFMA src_a
#   → row (=tid%16) within the MFMA tile, which reads from B's K-column at the
#     position determined by the S2R layout
#
# This is where it gets tricky. The MFMA's "row" dimension of src_a uses
# B data loaded from shared memory. The B data in B_local comes from
# B_shared indexed by lane (tid&15). So MFMA row index = tid%16 = lane row,
# and this maps to B's N-tile.
#
# Meanwhile M_store = wg*64 + sub_m*32 + i_6*16 + (tid&15)
# This stores the result at M position = A's M position
# But the MFMA's row dimension came from B (N dimension)!
#
# MFMA row (from src_a/B) → stored at M position (from A's tile structure)
# MFMA col (from src_b/A) → stored at N position (from B's tile structure)
#
# This is a SWAP! The data from B (which should go to N) ends up stored at M.

# Let me verify more concretely.
# If we have A[m,k] and B[n,k], the correct result is C[m,n] = sum_k A[m,k]*B[n,k]
#
# What the kernel computes:
#   MFMA result[row, col] = sum_k B_tile[row_k_data] * A_tile[col_k_data]
# where row indexes into B's data and col indexes into A's data.
# So result[row, col] = C_correct[col_A_pos, row_B_pos] (transposed!)
#
# Store maps: row → M_store (A's M position), col → N_store (B's N position)
# But result[row,col] = C_correct[A_col_pos, B_row_pos]
#
# If row→M and col→N, then we store C_correct[A_col_pos, B_row_pos] at (M, N)
# This is only correct if A_col_pos = M and B_row_pos = N
# i.e., the M store position must match where A's data came from,
# and the N store position must match where B's data came from.

# The lane-level mapping:
# MFMA row = tid & 15, MFMA col = ((tid&63)>>4)*4 + i
#
# For src_a (B_local): the data was loaded from shared at lane position = tid & 15
#   This came from B[n_pos, k_pos] where n_pos depends on the B shared layout
#   n_pos within the 16-element MFMA tile = tid & 15 (the lane maps to N directly)
#
# For src_b (A_local): the data was loaded at lane position = tid & 15
#   This came from A[m_pos, k_pos] where m_pos depends on the A shared layout
#   m_pos within the 16-element MFMA tile = tid & 15
#
# MFMA result[row, col]:
#   row = tid & 15 → this is BOTH B's N-lane and A's M-lane (same lane!)
#   col = (tid // 16) % 4 * 4 + i → this is the group ID within the warp
#
# Wait, this doesn't make sense. In MFMA, src_a and src_b use different lane mappings.
# Let me look at the actual MFMA lane mapping more carefully.

# For v_mfma_f32_16x16x32_bf16 (gfx950):
# Input A (src_a): 8 bf16 packed in each lane's register
#   The 64 lanes of a warp collectively provide a 16x32 matrix A
#   Lane l provides A[l%16, (l//16)*8 .. (l//16)*8+7]
#   So lane l provides row l%16, columns (l//16)*8..(l//16)*8+7
#   (l//16 ranges 0..3, giving 4*8=32 K columns)
#
# Input B (src_b): 8 bf16 packed in each lane's register
#   Same layout: lane l provides B[l%16, (l//16)*8..(l//16)*8+7]
#
# Output D = C + A * B^T (where B^T means the K dimension is contracted)
#   D is 16x16
#   Lane l gets D[l%16, (l//16)*4..(l//16)*4+3] → 4 float32 values
#
# So the computation is: D[m, n] = C[m, n] + sum_k A[m, k] * B[n, k]
#   where m = row index (0..15), n = col index (0..15)
#
# In our kernel:
#   src_a = B_local data → provides MFMA's "A matrix" → row dimension = m
#   src_b = A_local data → provides MFMA's "B matrix" → col dimension = n
#
# So: D[m, n] = sum_k B_local_data[m, k] * A_local_data[n, k]
#
# Lane l holds: D[l%16, (l//16)*4+i] for i=0..3
#
# For src_a (B_local): lane l provides data at row = l%16
#   B_local data at lane l was loaded from: B_shared[... + (tid&15)*64 + swizzle]
#   where tid = actual threadIdx.x and l = tid within the conceptual warp
#   Wait, but tid & 15 IS l%16 (since the 64-lane warp is mapped from 64 threads)
#
# Hmm, actually for scattered warp layout, the warp structure might be different.
# Let me just check: are all 64 threads in a wavefront contiguous?
# In AMD, a wavefront is always 64 contiguous threads.
# So warp 0 = threads 0-63, warp 1 = threads 64-127, etc.
# Lane within warp = tid % 64
#
# For MFMA: lane = tid % 64
#   src_a provides A_mfma[lane%16, (lane//16)*8..(lane//16)*8+7]
#   src_b provides B_mfma[lane%16, (lane//16)*8..(lane//16)*8+7]
#   result[lane%16, (lane//16)*4+i] for i=0..3

# In the kernel, src_a = B_local. What does lane l (=tid%64) provide from B_local?
# B_local is loaded from B_shared. But the B S2R load uses:
#   B_shared[... + (tid & 15) * 64 + ...]
# Here (tid & 15) = lane % 16 → this is the row position in shared memory
# The swizzle + local_id determines the K column
# So B_local at lane l represents B[N_pos_determined_by_tile + l%16, K_columns]
#
# Similarly, A_local at lane l represents A[M_pos_determined_by_tile + l%16, K_columns]
#
# MFMA: src_a provides row l%16 → which has B's data at N-row = l%16 offset within tile
#        src_b provides row l%16 → which has A's data at M-row = l%16 offset within tile
#
# MFMA computes: D[m, n] = sum_k src_a[m, k] * src_b[n, k]
# With src_a = B data and src_b = A data:
#   D[m, n] = sum_k B[m_within_tile, k] * A[n_within_tile, k]
#
# But we want: C[M, N] = sum_k A[M, k] * B[N, k]
#
# So D[m, n] = C_correct[n_within_A_tile, m_within_B_tile]
# The M and N are TRANSPOSED in the MFMA result!
#
# Now, where is D[m, n] stored?
# Lane l gets D[l%16, (l//16)*4+i]
# Store address:
#   M = wg*64 + (sub_m*2+i_6)*16 + (tid & 15)       // (tid&15) = l%16 = MFMA m index
#   N = n_half*128 + (sub_n*4+j_1)*16 + ((tid&63)>>4)*4 + i  // (tid&63)>>4 = l//16 = MFMA col group
#
# So we store D[l%16, (l//16)*4+i] at C[M, N] where:
#   M uses l%16 (= MFMA m = B's N-lane position)
#   N uses l//16 (= MFMA n group)
#
# The M position = wg*64 + sub_m*32 + i_6*16 + l%16
#   This should correspond to A's M position, but MFMA m comes from B's data!
#   i_6 indexes A_local, so the tile offset (sub_m*32+i_6*16) is correct for A's M
#   But l%16 within the MFMA tile is B's N-lane (not A's M-lane)
#
# The N position = n_half*128 + sub_n*64 + j_1*16 + (l//16)*4 + i
#   j_1 indexes B_local, so the tile offset is correct for B's N
#   But (l//16)*4+i within the MFMA tile is A's column group (not B's N-lane group)
#
# CONCLUSION: The M/N mapping within each 16x16 MFMA tile IS CORRECT!
# Because: both A and B use (tid&15) as their row index in shared memory.
# Lane l has: A data at M-offset l%16, B data at N-offset l%16
# MFMA result D[m,n] has m = l%16 = both A's and B's lane row
# But D[m,n] = src_a[m,k]*src_b[n,k] where m maps to row in src_a (B) and n maps to row in src_b (A)
#
# Wait, I keep going in circles. Let me be very precise.

print("=== Precise MFMA lane analysis ===")
print()
print("For v_mfma_f32_16x16x32_bf16:")
print("  src_a (first operand): 8 bf16 per lane")
print("  src_b (second operand): 8 bf16 per lane")
print("  D = C + src_a_matrix * src_b_matrix^T")
print("  D is 16x16")
print()
print("Lane l provides:")
print("  src_a_matrix[l%16, (l//16)*8..(l//16)*8+7]")
print("  src_b_matrix[l%16, (l//16)*8..(l//16)*8+7]")
print("  Gets D[l%16, (l//16)*4..(l//16)*4+3]")
print()
print("In our kernel:")
print("  src_a = B_local → B data at tile_N_offset + l%16")
print("  src_b = A_local → A data at tile_M_offset + l%16")
print()
print("D[m, n] = sum_k B[tile_N + m, k] * A[tile_M + n, k]")
print("        = C_correct[tile_M + n, tile_N + m]  (transposed!)")
print()
print("Store: D[l%16, (l//16)*4+i] → C[M_store, N_store]")
print("  M_store = wg*64 + (sub_m*2+i_6)*16 + l%16")
print("  N_store = n_half*128 + (sub_n*4+j_1)*16 + (l//16)*4+i")
print()
print("But D[m=l%16, n=(l//16)*4+i] = C_correct[tile_M + n, tile_N + m]")
print("  = C_correct[tile_M + (l//16)*4+i, tile_N + l%16]")
print()
print("We store this at C[M_store, N_store] where:")
print("  M_store = wg*64 + (sub_m*2+i_6)*16 + l%16")
print("  N_store = n_half*128 + (sub_n*4+j_1)*16 + (l//16)*4+i")
print()
print("For correctness we need:")
print("  M_store = tile_M + (l//16)*4+i  → the A (M) index from MFMA col")
print("  N_store = tile_N + l%16          → the B (N) index from MFMA row")
print()
print("But M_store uses l%16 (MFMA row = B's N index)")
print("And N_store uses (l//16)*4+i (MFMA col = A's M index)")
print()
print("MISMATCH! The intra-tile positions are swapped!")
print()

# Let me verify with a concrete example
print("=== Concrete example ===")
# Thread 0, warp 0: lane = 0
# sub_m=0, i_6=0, sub_n=0, j_1=0, kp=0
# MFMA result D[0, 0] = sum_k B[tile_N+0, k] * A[tile_M+0, k]
# = C_correct[tile_M+0, tile_N+0]  ← element (M=0, N=0)
# Stored at: M = 0*64 + 0*16 + 0 = 0, N = 0*128 + 0*16 + 0 = 0
# C[0, 0] = C_correct[0, 0] ✓ (matches because both are 0)

# Thread 1, warp 0: lane = 1
# D[1, 0] = sum_k B[tile_N+1, k] * A[tile_M+0, k]
# = C_correct[tile_M+0, tile_N+1]  ← element (M=0, N=1)
# Stored at: M = 0*64 + 0*16 + 1 = 1, N = 0*128 + 0*16 + 0 = 0
# C[1, 0] = C_correct[0, 1]  ← WRONG! Should be at C[0, 1] but stored at C[1, 0]

print("Thread 1 (lane 1), sub_m=0, i_6=0, sub_n=0, j_1=0:")
print("  D[1, 0] = C_correct[M=0, N=1]")
print("  Stored at C[M=1, N=0]")
print("  => C[1,0] gets value C_correct[0,1] — TRANSPOSED!")
print()

# Thread 16 (lane 16):
# D[0, 4] = sum_k B[tile_N+0, k] * A[tile_M+4, k]
# = C_correct[tile_M+4, tile_N+0]
# Stored at: M = 0*64+0*16+0 = 0, N = 0+0+0+4 = 4
# C[0, 4] = C_correct[4, 0] ← WRONG! Should be C_correct[0, 4]

print("Thread 16 (lane 16), sub_m=0, i_6=0, sub_n=0, j_1=0:")
print("  D[0, 4] = C_correct[M=4, N=0]")
print("  Stored at C[M=0, N=4]")
print("  => C[0,4] gets value C_correct[4,0] — TRANSPOSED!")
print()

# So within each 16x16 MFMA tile, the result is TRANSPOSED!
# C[M_store, N_store] = C_correct[N_store_within_tile + tile_M, M_store_within_tile + tile_N]

# However... if the matrix is square and the tile offsets happen to map M→M and N→N
# at the tile level, then only the INTRA-tile positions would be swapped.
#
# The tile-level mapping:
#   tile_M = wg*64 + sub_m*32 + i_6*16  (from A_local indexing)
#   tile_N = n_half*128 + sub_n*64 + j_1*16  (from B_local indexing)
#   These are CORRECT at the tile level.
#   But within the tile, m_offset = l%16 should map to N, and n_offset should map to M.

# So the result is: C is correct at the tile level but transposed within each 16x16 tile!
# This would cause ~24% mismatch for random data where C[m,n] ≠ C[n,m].

print("CONCLUSION: The output is transposed within each 16x16 MFMA tile!")
print("The MFMA operands A_local and B_local are passed in the wrong order.")
print("Fix: swap the first two arguments of __builtin_amdgcn_mfma_f32_16x16x32_bf16")
print("  Current: mfma(B_local, A_local, C_local)")
print("  Should be: mfma(A_local, B_local, C_local)")
