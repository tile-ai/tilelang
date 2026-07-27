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

EMPIRICALLY VERIFIED hardware layouts for wmma_f32_16x16x16_f16_w32 (gfx11):

  A[M=16][K=16]:
    thread t, elem e -> A[M=t%16][K=e]
    Forward: (M, K) -> (thread=M, local=K%16) [Mapping to tid=0~15]
    Reverse: (thread, local) -> (M=thread%16, K=local)
    Memory load: A[M=t%16][K=0..+15] -> CONTIGUOUS in K (vectorized)

  B[K=16][N=16] (non-transposed, K x N storage):
    thread t, elem e -> B[K=e][N=t%16]
    Forward: (K, N) -> (thread=N, local=K%16) [Mapping to tid=0~15]
    Reverse: (thread, local) -> (K=local, N=thread%16)

  B_T[N=16][K=16] (transposed storage of B):
    B_T[N=t%16][K=e] -> CONTIGUOUS in K (vectorized)

  D[M=16][N=16]:
    thread t, elem l -> D[M=(t//16)+l*2][N=t%16]
    Forward: (M, N) -> (thread=(M%2)*16+N, local=M//2)
    Reverse: (thread, local) -> (M=(thread//16)+local*2, N=thread%16)
    Store: D[M=(t//16)+l*2][N=t%16] = d_vec[l]

NOTE:
1.  A and D have DIFFERENT layouts (e.g. For gfx12, A uses t%16 for M,
    D uses (t//16)*8+l for M). This means they cannot be used interchangeably
    without a layout change.
2.  For gfx11, lane 16~31 share the same A/B data as lane 0~15.

local_size = 8 (gfx12) | 16 (gfx11)
"""

from tvm.runtime import convert


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 A matrix: shared[M=16][K=16]
# A[M=t%16][K=(t//16)*8+l] -> vectorized load from row M=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_A_gfx12(i, j):
    """Forward: A[i=M, j=K] -> (thread=(j//8)*16+i, local=j%8)."""
    thread_id = (j // 8) * 16 + i  # (K//8)*16 + M
    local_id = j % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_A_gfx12(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=thread%16, j=K=(thread//16)*8+local)."""
    return thread_id % 16, (thread_id // 16) * 8 + local_id


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 A_T matrix (transposed storage, K x M): shared[K=16][M=16]
# A_T[K=(t//16)*8+l][M=t%16]
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_A_colmajor_gfx12(i, j):
    """Forward: A_T[i=K, j=M] -> (thread=(i//8)*16+j, local=i%8)."""
    thread_id = (i // 8) * 16 + j  # (K//8)*16 + M
    local_id = i % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_A_colmajor_gfx12(thread_id, local_id):
    """Reverse: (thread, local) -> (i=K=(thread//16)*8+local, j=M=thread%16)."""
    return (thread_id // 16) * 8 + local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 B matrix (non-transposed, K x N): shared[K=16][N=16]
# B[K=(t//16)*8+l][N=t%16]
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_B_gfx12(i, j):
    """Forward: B[i=K, j=N] -> (thread=(i//8)*16+j, local=i%8)."""
    thread_id = (i // 8) * 16 + j  # (K//8)*16 + N
    local_id = i % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_B_gfx12(thread_id, local_id):
    """Reverse: (thread, local) -> (i=K=(thread//16)*8+local, j=N=thread%16)."""
    return (thread_id // 16) * 8 + local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 B_T matrix (transposed storage, N x K): shared[N=16][K=16]
# B_T[N=t%16][K=(t//16)*8+l] -> vectorized load from row N=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_B_colmajor_gfx12(i, j):
    """Forward: B_T[i=N, j=K] -> (thread=(j//8)*16+i, local=j%8)."""
    thread_id = (j // 8) * 16 + i  # (K//8)*16 + N
    local_id = j % 8  # K%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_B_colmajor_gfx12(thread_id, local_id):
    """Reverse: (thread, local) -> (i=N=thread%16, j=K=(thread//16)*8+local)."""
    return thread_id % 16, (thread_id // 16) * 8 + local_id


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 D/C output matrix: shared[M=16][N=16] fp32
# D[M=(t//16)*8+l][N=t%16] -- hardware native
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_C_gfx12(i, j):
    """Forward: D[i=M, j=N] -> (thread=(i//8)*16+j, local=i%8)."""
    thread_id = (i // 8) * 16 + j  # (M//8)*16 + N
    local_id = i % 8  # M%8
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_C_gfx12(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=(thread//16)*8+local, j=N=thread%16)."""
    return (thread_id // 16) * 8 + local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx12 store index map: (thread, local) -> (M, N) in D (hardware D layout)
# D[M=(t//16)*8+local][N=t%16] -- affine, invertible
# ──────────────────────────────────────────────────────────────────────────────


def wmma_store_index_map_gfx12(thread_id, local_id):
    """(thread, local) -> (M, N) in D.  Hardware D layout."""
    i = (thread_id // 16) * 8 + local_id  # M
    j = thread_id % 16  # N
    return convert([i, j])


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 A matrix: shared[M=16][K=16]
# A[M=t%16][K=l] -> vectorized load from row M=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x16_layout_A_gfx11(i, j):
    """
    Forward: A[i=M, j=K] -> (thread=i, local=j%16).
    ATTN: Here we only reflect (i, j) to the lower-half-lane of threads in
    a warp.
    """
    thread_id = i
    local_id = j % 16
    return thread_id, local_id


def thread_id_shared_access_32x16_to_16x16_layout_A_gfx11(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=thread%16, j=K=local)"""
    return thread_id % 16, local_id


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 A_T matrix (transposed storage, K x M): shared[K=16][M=16]
# A_T[K=l][M=t%16]
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x16_layout_A_colmajor_gfx11(i, j):
    """
    Forward: A_T[i=K, j=M] -> (thread=M, local=K%16).
    ATTN: Here we only reflect (i, j) to the lower-half-lane of threads in
    a warp.
    """
    thread_id = j
    local_id = i % 16
    return thread_id, local_id


def thread_id_shared_access_32x16_to_16x16_layout_A_colmajor_gfx11(thread_id, local_id):
    """Reverse: (thread, local) -> (i=K=local, j=M=thread%16)"""
    return local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 B matrix (non-transposed, K x N): shared[K=16][N=16]
# B[K=l][N=t%16]
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x16_layout_B_gfx11(i, j):
    """
    Forward: B[i=K, j=N] -> (thread=N, local=K%16).
    ATTN: Here we only reflect (i, j) to the lower-half-lane of threads in
    a warp.
    """
    thread_id = j
    local_id = i % 16
    return thread_id, local_id


def thread_id_shared_access_32x16_to_16x16_layout_B_gfx11(thread_id, local_id):
    """Reverse: (thread, local) -> (i=K=local, j=N=thread%16)"""
    return local_id, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 B_T matrix (transposed storage, N x K): shared[N=16][K=16]
# B_T[N=t%16][K=l] -> vectorized load from row N=t%16, consecutive K
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x16_layout_B_colmajor_gfx11(i, j):
    """
    Forward: B_T[i=N, j=K] -> (thread=i, local=j%16).
    ATTN: Here we only reflect (i, j) to the lower-half-lane of threads in
    a warp.
    """
    thread_id = i
    local_id = j % 16
    return thread_id, local_id


def thread_id_shared_access_32x16_to_16x16_layout_B_colmajor_gfx11(thread_id, local_id):
    """Reverse: (thread, local) -> (j=K=local, i=N=thread%16)"""
    return thread_id % 16, local_id


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 D/C output matrix: shared[M=16][N=16] fp32
# D[M=(t//16)+l*2][N=t%16] -- hardware native
# ──────────────────────────────────────────────────────────────────────────────


def shared_16x16_to_local_32x8_layout_C_gfx11(i, j):
    """Forward: D[i=M, j=N] -> (thread=(i%2)*16+j, local=i//2)."""
    thread_id = (i % 2) * 16 + j
    local_id = i // 2
    return thread_id, local_id


def thread_id_shared_access_32x8_to_16x16_layout_C_gfx11(thread_id, local_id):
    """Reverse: (thread, local) -> (i=M=(thread//16)+local*2, j=N=thread%16)"""
    return (thread_id // 16) + local_id * 2, thread_id % 16


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 store index map: (thread, local) -> (M, N) in D (hardware D layout)
# D[M=(t//16)+local*2][N=t%16] -- affine, invertible
# ──────────────────────────────────────────────────────────────────────────────


def wmma_store_index_map_gfx11(thread_id, local_id):
    """(thread, local) -> (M, N) in D.  Hardware D layout."""
    i = (thread_id // 16) + local_id * 2
    j = thread_id % 16
    return convert([i, j])


# ──────────────────────────────────────────────────────────────────────────────
# gfx11 fragment-forward helpers for duplicated half-wave ownership
# ──────────────────────────────────────────────────────────────────────────────


def fragment_forward_A_gfx11(i, j, rep):
    """Replicated fragment forward map for gfx11 A.

    The canonical owner lives in the lower half-wave and `rep` selects whether
    the logical element is materialized in the lower or upper half-wave copy.
    """
    thread_id, local_id = shared_16x16_to_local_32x16_layout_A_gfx11(i, j)
    return thread_id + 16 * rep, local_id


def fragment_forward_A_colmajor_gfx11(i, j, rep):
    """Replicated fragment forward map for gfx11 transposed A."""
    thread_id, local_id = shared_16x16_to_local_32x16_layout_A_colmajor_gfx11(i, j)
    return thread_id + 16 * rep, local_id


def fragment_forward_B_gfx11(i, j, rep):
    """Replicated fragment forward map for gfx11 B."""
    thread_id, local_id = shared_16x16_to_local_32x16_layout_B_gfx11(i, j)
    return thread_id + 16 * rep, local_id


def fragment_forward_B_colmajor_gfx11(i, j, rep):
    """Replicated fragment forward map for gfx11 transposed B."""
    thread_id, local_id = shared_16x16_to_local_32x16_layout_B_colmajor_gfx11(i, j)
    return thread_id + 16 * rep, local_id


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────


def _unsupported_rdna_generation(rdna_gen: int):
    raise ValueError(f"Unsupported RDNA generation for WMMA layout: {rdna_gen}")


def get_wmma_a_layout_funcs(rdna_gen: int, transposed: bool):
    """Return (forward_map, reverse_map) for A layout."""
    if rdna_gen == 11:
        if transposed:
            return (
                shared_16x16_to_local_32x16_layout_A_colmajor_gfx11,
                thread_id_shared_access_32x16_to_16x16_layout_A_colmajor_gfx11,
            )
        return (
            shared_16x16_to_local_32x16_layout_A_gfx11,
            thread_id_shared_access_32x16_to_16x16_layout_A_gfx11,
        )
    if rdna_gen == 12:
        if transposed:
            return (
                shared_16x16_to_local_32x8_layout_A_colmajor_gfx12,
                thread_id_shared_access_32x8_to_16x16_layout_A_colmajor_gfx12,
            )
        return (
            shared_16x16_to_local_32x8_layout_A_gfx12,
            thread_id_shared_access_32x8_to_16x16_layout_A_gfx12,
        )
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_b_layout_funcs(rdna_gen: int, transposed: bool):
    """Return (forward_map, reverse_map) for B layout."""
    if rdna_gen == 11:
        if transposed:
            return (
                shared_16x16_to_local_32x16_layout_B_colmajor_gfx11,
                thread_id_shared_access_32x16_to_16x16_layout_B_colmajor_gfx11,
            )
        return (
            shared_16x16_to_local_32x16_layout_B_gfx11,
            thread_id_shared_access_32x16_to_16x16_layout_B_gfx11,
        )
    if rdna_gen == 12:
        if transposed:
            return (
                shared_16x16_to_local_32x8_layout_B_colmajor_gfx12,
                thread_id_shared_access_32x8_to_16x16_layout_B_colmajor_gfx12,
            )
        return (
            shared_16x16_to_local_32x8_layout_B_gfx12,
            thread_id_shared_access_32x8_to_16x16_layout_B_gfx12,
        )
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_c_layout_funcs(rdna_gen: int):
    """Return (forward_map, reverse_map) for C/D layout."""
    if rdna_gen == 11:
        return (
            shared_16x16_to_local_32x8_layout_C_gfx11,
            thread_id_shared_access_32x8_to_16x16_layout_C_gfx11,
        )
    if rdna_gen == 12:
        return (
            shared_16x16_to_local_32x8_layout_C_gfx12,
            thread_id_shared_access_32x8_to_16x16_layout_C_gfx12,
        )
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_store_index_map_func(rdna_gen: int):
    """Return the (thread_id, local_id) -> (row, col) store map."""
    if rdna_gen == 11:
        return wmma_store_index_map_gfx11
    if rdna_gen == 12:
        return wmma_store_index_map_gfx12
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_a_fragment_forward_func(rdna_gen: int, transposed: bool):
    """Return the fragment forward function for A layout."""
    if rdna_gen == 11:
        return fragment_forward_A_colmajor_gfx11 if transposed else fragment_forward_A_gfx11
    if rdna_gen == 12:
        return None
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_b_fragment_forward_func(rdna_gen: int, transposed: bool):
    """Return the fragment forward function for B layout."""
    if rdna_gen == 11:
        return fragment_forward_B_colmajor_gfx11 if transposed else fragment_forward_B_gfx11
    if rdna_gen == 12:
        return None
    _unsupported_rdna_generation(rdna_gen)


def get_wmma_fragment_replicate_count(rdna_gen: int):
    """Return the fragment replicate count used for logical one-to-many owners."""
    if rdna_gen == 11:
        return 2
    if rdna_gen == 12:
        return 1
    _unsupported_rdna_generation(rdna_gen)
