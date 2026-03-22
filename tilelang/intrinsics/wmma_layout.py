"""Layout functions for AMD RDNA WMMA instructions (gfx11/gfx12).

EMPIRICALLY VERIFIED hardware layouts for wmma_f32_16x16x16_f16_w32_gfx12:

  A[M=16][K=16]:
    thread t, elem e -> A[M=t%16][K=(t//16)*8+e]
    Forward: (M, K) -> (thread=(K//8)*16+M, local=K%8)
    Reverse: (thread, local) -> (M=thread%16, K=(thread//16)*8+local)
    Memory load: A[M=t%16][K=(t//16)*8..+7] -> CONTIGUOUS in K (vectorized)

  B[K=16][N=16] (non-transposed, K x N storage):
    thread t, elem e -> B[K=(t//16)*8+e][N=t%16]
    Forward: (K, N) -> (thread=(K//8)*16+N, local=K%8)
    Reverse: (thread, local) -> (K=(thread//16)*8+local, N=thread%16)

  B_T[N=16][K=16] (transposed storage of B):
    B_T[N=t%16][K=(t//16)*8+e] -> CONTIGUOUS in K (vectorized)

  D[M=16][N=16]:
    thread t, elem l -> D[M=(t//16)*8+l][N=t%16]
    Forward: (M, N) -> (thread=(M//8)*16+N, local=M%8)
    Reverse: (thread, local) -> (M=(thread//16)*8+local, N=thread%16)
    Store: D[M=(t//16)*8+l][N=t%16] = d_vec[l]

NOTE: A and D have DIFFERENT layouts (A uses t%16 for M, D uses (t//16)*8+l for M).
This means they cannot be used interchangeably without a layout change.

local_size = 8 per thread
"""

from tvm.runtime import convert


# ──────────────────────────────────────────────────────────────────────────────
# A matrix: shared[M=16][K=16]
# A[M=t%16][K=(t//16)*8+l] -> vectorized load from row M=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_A(i, j):
    """Forward: A[i=M, j=K] -> (thread=(j//8)*16+i, local=j%8)."""
    thread_id = (j // 8) * 16 + i  # (K//8)*16 + M
    local_id = j % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_A(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=thread%16, j=K=(thread//16)*8+local)."""
    return thread_id % 16, (thread_id // 16) * 8 + local_id


# ──────────────────────────────────────────────────────────────────────────────
# B matrix (non-transposed, K x N): shared[K=16][N=16]
# B[K=(t//16)*8+l][N=t%16]
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_B(i, j):
    """Forward: B[i=K, j=N] -> (thread=(i//8)*16+j, local=i%8)."""
    thread_id = (i // 8) * 16 + j  # (K//8)*16 + N
    local_id = i % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_B(thread_id, local_id):
    """Reverse: (thread, local) -> (i=K=(thread//16)*8+local, j=N=thread%16)."""
    return (thread_id // 16) * 8 + local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# B_T matrix (transposed storage, N x K): shared[N=16][K=16]
# B_T[N=t%16][K=(t//16)*8+l] -> vectorized load from row N=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_B_colmajor(i, j):
    """Forward: B_T[i=N, j=K] -> (thread=(j//8)*16+i, local=j%8)."""
    thread_id = (j // 8) * 16 + i  # (K//8)*16 + N
    local_id = j % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_B_colmajor(thread_id, local_id):
    """Reverse: (thread, local) -> (i=N=thread%16, j=K=(thread//16)*8+local)."""
    return thread_id % 16, (thread_id // 16) * 8 + local_id


# ──────────────────────────────────────────────────────────────────────────────
# D/C output matrix: shared[M=16][N=16] fp32
# D[M=(t//16)*8+l][N=t%16] -- hardware native
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_C(i, j):
    """Forward: D[i=M, j=N] -> (thread=(i//8)*16+j, local=i%8)."""
    thread_id = (i // 8) * 16 + j  # (M//8)*16 + N
    local_id = i % 8  # M%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_C(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=(thread//16)*8+local, j=N=thread%16)."""
    return (thread_id // 16) * 8 + local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# Store index map: (thread, local) -> (M, N) in D  (hardware D layout)
# D[M=(t//16)*8+local][N=t%16] -- affine, invertible
# ──────────────────────────────────────────────────────────────────────────────


def wmma_store_index_map(thread_id, local_id):
    """(thread, local) -> (M, N) in D.  Hardware D layout."""
    i = (thread_id // 16) * 8 + local_id  # M
    j = thread_id % 16  # N
    return convert([i, j])
