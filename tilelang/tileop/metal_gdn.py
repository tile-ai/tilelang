from __future__ import annotations

from tilelang import language as T
from tilelang.tileop import metal_simdgroup as metal_sg


@T.macro
def kkt_score_tile(
    row_k_data,
    col_k_data,
    scores_data,
    *,
    block: int = 8,
    key_dim: int = 8,
) -> None:
    """Compute one 8x8 GDN KKT score tile from staged fp32 key tiles."""
    row_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    col_rt = metal_sg.alloc_rt(T.float32, 1, 1, layout=metal_sg.TileLayout.TRANSPOSED)
    score_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    metal_sg.fill_rt(score_rt, T.float32(0.0))
    metal_sg.load_threadgroup_to_rt(row_rt, T.float32, row_k_data, 0, block * key_dim, key_dim)
    metal_sg.load_threadgroup_to_rt(
        col_rt,
        T.float32,
        col_k_data,
        0,
        block * key_dim,
        key_dim,
        transpose=True,
    )
    metal_sg.mma_abt(score_rt, row_rt, col_rt)
    metal_sg.materialize_rt_to_shared(score_rt, T.float32, scores_data, 0, block * block, block)


@T.macro
def kkt_score_tile_accum(
    row_k_data,
    col_k_data,
    scores_data,
    *,
    block: int = 8,
    key_dim: int = 16,
    key_offset: int = 0,
    clear: bool = True,
) -> None:
    """Accumulate one 8-column slice into a staged KKT 8x8 score tile."""
    row_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    col_rt = metal_sg.alloc_rt(T.float32, 1, 1, layout=metal_sg.TileLayout.TRANSPOSED)
    score_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    if clear:
        metal_sg.fill_rt(score_rt, T.float32(0.0))
    else:
        metal_sg.load_threadgroup_to_rt(score_rt, T.float32, scores_data, 0, block * block, block)
    metal_sg.load_threadgroup_to_rt(row_rt, T.float32, row_k_data, key_offset, block * key_dim, key_dim)
    metal_sg.load_threadgroup_to_rt(
        col_rt,
        T.float32,
        col_k_data,
        key_offset,
        block * key_dim,
        key_dim,
        transpose=True,
    )
    metal_sg.mma_abt(score_rt, row_rt, col_rt)
    metal_sg.materialize_rt_to_shared(score_rt, T.float32, scores_data, 0, block * block, block)


@T.macro
def apply_kkt_gate_triangular_tile(
    scores,
    g_row,
    g_col,
    a_pre,
    head,
    row_block,
    col_block,
    lane,
    *,
    block: int = 8,
    chunk_size: int,
    threads: int = 32,
) -> None:
    """Apply GDN KKT gate decay and causal triangular mask to one score tile."""
    for linear in T.serial(lane, block * block, step=threads):
        local_row = linear // block
        local_col = linear - local_row * block
        c = row_block * block + local_row
        d = col_block * block + local_col
        if c < chunk_size and d < chunk_size:
            if d < c:
                a_pre[c, head, d] = scores[local_row, local_col] * T.exp(g_row[local_row] - g_col[local_col])
            else:
                a_pre[c, head, d] = 0.0


@T.macro
def wu_linear_element(
    k,
    v,
    beta,
    g_cum,
    a,
    w,
    u,
    head,
    linear,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
) -> None:
    """Compute one scalar W or U output element from solved GDN A."""
    c = linear // (key_dim + value_dim)
    rem = linear - c * (key_dim + value_dim)
    acc = T.alloc_var(T.float32)
    acc = 0.0
    if rem < key_dim:
        kk = rem
        for d in T.serial(chunk_size):
            acc += a[c, head, d] * T.cast(k[d, head, kk], T.float32) * beta[d, head] * T.exp(g_cum[d, head])
        w[c, head, kk] = acc
    else:
        vv = rem - key_dim
        for d in T.serial(chunk_size):
            acc += a[c, head, d] * T.cast(v[d, head, vv], T.float32) * beta[d, head]
        u[c, head, vv] = acc


@T.macro
def wu_score_tiles_strided(
    a_data,
    k_scaled_data,
    v_scaled_data,
    w_acc,
    u_acc,
    *,
    a_offset: int = 0,
    k_offset: int = 0,
    v_offset: int = 0,
    a_stride: int = 16,
    kv_stride: int = 16,
    block: int = 8,
) -> None:
    """Accumulate one strided 8x8 A/K/V tile slice into W and U outputs."""
    a_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    k_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    v_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    metal_sg.load_threadgroup_to_rt(a_rt, T.float32, a_data, a_offset, block * a_stride, a_stride)
    metal_sg.load_threadgroup_to_rt(k_rt, T.float32, k_scaled_data, k_offset, block * kv_stride, kv_stride)
    metal_sg.load_threadgroup_to_rt(v_rt, T.float32, v_scaled_data, v_offset, block * kv_stride, kv_stride)
    metal_sg.mma_ab(w_acc, a_rt, k_rt)
    metal_sg.mma_ab(u_acc, a_rt, v_rt)


@T.macro
def wu_score_tiles(
    a_data,
    k_scaled_data,
    v_scaled_data,
    w_acc,
    u_acc,
    *,
    block: int = 8,
) -> None:
    """Accumulate one staged 8x8 A tile into W and U RegisterTile outputs."""
    a_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    k_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    v_rt = metal_sg.alloc_rt(T.float32, 1, 1)
    metal_sg.load_threadgroup_to_rt(a_rt, T.float32, a_data, 0, block * block, block)
    metal_sg.load_threadgroup_to_rt(k_rt, T.float32, k_scaled_data, 0, block * block, block)
    metal_sg.load_threadgroup_to_rt(v_rt, T.float32, v_scaled_data, 0, block * block, block)
    metal_sg.mma_ab(w_acc, a_rt, k_rt)
    metal_sg.mma_ab(u_acc, a_rt, v_rt)
