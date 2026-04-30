from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilelang import language as T


class TileLayout(str, Enum):
    """Internal Metal register-tile layout metadata.

    These values describe how higher-level tile code intends to interpret a
    tile.  The current MSL lowering still uses explicit simdgroup load/store
    transpose flags; this metadata is deliberately internal until the layout
    contract has survived GEMM, attention, and MoE retargeting.
    """

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    TRANSPOSED = "transposed"


@dataclass(frozen=True)
class RegisterTile:
    """Opaque array of Metal 8x8 simdgroup register fragments.

    The ``fragment`` object is still the only object passed to TileLang/Metal
    intrinsics.  This metadata gives compiler-owned lowerings a reusable way to
    address arrays of fragments without exposing scalar fragment indexing.
    """

    fragment: object
    fragments_m: int
    fragments_n: int
    rows: int = 8
    cols: int = 8
    layout: TileLayout = TileLayout.ROW_MAJOR

    @property
    def data(self):
        return self.fragment.data

    def index(self, tile_m: int, tile_n: int = 0) -> int:
        return tile_m * self.fragments_n + tile_n


@dataclass(frozen=True)
class MMATile(RegisterTile):
    """Backward-compatible name for existing internal simdgroup users."""


@dataclass(frozen=True)
class RowVector:
    """Internal row vector backed by explicit scalar storage.

    Row vectors intentionally do not index ``metal.simdgroup`` fragments.  They
    operate on materialized buffers until there is a native register-vector
    lowering for row reductions and normalization.
    """

    values: object
    length: int
    dtype: object = T.float32

    @property
    def data(self):
        return self.values.data


def _require_layout(tile: RegisterTile, expected: TileLayout, role: str, op_name: str) -> None:
    if tile.layout != expected:
        raise ValueError(f"{op_name} requires {role} layout {expected.value}, got {tile.layout.value}")


def _require_load_layout(tile: RegisterTile, transpose: bool, op_name: str) -> None:
    expected = TileLayout.TRANSPOSED if transpose else TileLayout.ROW_MAJOR
    if tile.layout != expected:
        mode = "transposed" if transpose else "row-major"
        raise ValueError(f"{op_name} {mode} load requires tile layout {expected.value}, got {tile.layout.value}")


def _require_store_layout(tile: RegisterTile, transpose: bool, op_name: str) -> None:
    expected = TileLayout.TRANSPOSED if transpose else TileLayout.ROW_MAJOR
    if tile.layout != expected:
        mode = "transposed" if transpose else "row-major"
        raise ValueError(f"{op_name} {mode} store requires tile layout {expected.value}, got {tile.layout.value}")


@T.macro
def alloc_rt(
    dtype,
    fragments_m: int,
    fragments_n: int = 1,
    *,
    rows: int = 8,
    cols: int = 8,
    layout: TileLayout = TileLayout.ROW_MAJOR,
) -> RegisterTile:
    """Allocate an internal Metal register tile backed by 8x8 fragments."""
    if rows != 8 or cols != 8:
        raise ValueError(f"Metal register tiles are 8x8 fragments, got {rows}x{cols}")
    rt_fragment = T.alloc_fragment((fragments_m * fragments_n, rows, cols), dtype, scope="metal.simdgroup")
    return RegisterTile(rt_fragment, fragments_m, fragments_n, rows, cols, layout)


@T.macro
def fill(fragment, matrix_index, value, rows: int = 8, cols: int = 8) -> None:
    """Fill one opaque Metal simdgroup matrix fragment."""
    T.make_filled_simdgroup_matrix(fragment.data, matrix_index, value, rows, cols)


@T.macro
def access_ptr(dtype, data, offset, extent, rw_mask: int):
    return T.tvm_access_ptr(T.type_annotation(dtype), data, offset, extent, rw_mask)


@T.macro
def load(
    fragment,
    matrix_index,
    dtype,
    data,
    offset,
    extent,
    stride,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    T.simdgroup_load(
        fragment.data,
        matrix_index,
        access_ptr(dtype, data, offset, extent, 1),
        stride,
        rows,
        cols,
        T.bool(transpose),
    )


@T.macro
def store(
    fragment,
    matrix_index,
    dtype,
    data,
    offset,
    extent,
    stride,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    T.simdgroup_store(
        fragment.data,
        matrix_index,
        access_ptr(dtype, data, offset, extent, 2),
        stride,
        rows,
        cols,
        T.bool(transpose),
    )


@T.macro
def mma(acc, a, b, acc_index=0, a_index=0, b_index=0, out_index=None) -> None:
    """Accumulate ``a @ b`` into an opaque Metal simdgroup accumulator."""
    if out_index is None:
        out_index = acc_index
    T.simdgroup_multiply_accumulate(
        acc.data,
        out_index,
        a.data,
        a_index,
        b.data,
        b_index,
        acc.data,
        acc_index,
    )


@T.macro
def fill_tile(tile: MMATile, value) -> None:
    for tile_m in T.unroll(tile.fragments_m, explicit=True):
        for tile_n in T.unroll(tile.fragments_n, explicit=True):
            fill(tile.fragment, tile.index(tile_m, tile_n), value, tile.rows, tile.cols)


@T.macro
def fill_rt(tile: RegisterTile, value) -> None:
    """Fill every 8x8 fragment in a register tile."""
    for tile_m in T.unroll(tile.fragments_m, explicit=True):
        for tile_n in T.unroll(tile.fragments_n, explicit=True):
            fill(tile.fragment, tile.index(tile_m, tile_n), value, tile.rows, tile.cols)


@T.macro
def load_tile(
    tile: MMATile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    load(
        tile.fragment,
        tile.index(0, 0),
        dtype,
        data,
        offset,
        extent,
        stride,
        rows,
        cols,
        transpose,
    )


@T.macro
def load_global_to_rt(
    tile: RegisterTile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    tile_m: int = 0,
    tile_n: int = 0,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    """Load one 8x8 global-memory tile into a register-tile fragment."""
    _require_load_layout(tile, transpose, "load_global_to_rt")
    load(
        tile.fragment,
        tile.index(tile_m, tile_n),
        dtype,
        data,
        offset,
        extent,
        stride,
        rows,
        cols,
        transpose,
    )


@T.macro
def load_threadgroup_to_rt(
    tile: RegisterTile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    tile_m: int = 0,
    tile_n: int = 0,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    """Load one 8x8 threadgroup-memory tile into a register-tile fragment."""
    _require_load_layout(tile, transpose, "load_threadgroup_to_rt")
    load(
        tile.fragment,
        tile.index(tile_m, tile_n),
        dtype,
        data,
        offset,
        extent,
        stride,
        rows,
        cols,
        transpose,
    )


@T.macro
def store_tile(
    tile: MMATile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    store(
        tile.fragment,
        tile.index(0, 0),
        dtype,
        data,
        offset,
        extent,
        stride,
        rows,
        cols,
        transpose,
    )


@T.macro
def store_rt(
    tile: RegisterTile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    tile_m: int = 0,
    tile_n: int = 0,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    """Store one 8x8 register-tile fragment through explicit materialization."""
    _require_store_layout(tile, transpose, "store_rt")
    store(
        tile.fragment,
        tile.index(tile_m, tile_n),
        dtype,
        data,
        offset,
        extent,
        stride,
        rows,
        cols,
        transpose,
    )


@T.macro
def materialize_rt_to_shared(
    tile: RegisterTile,
    dtype,
    data,
    offset,
    extent,
    stride,
    *,
    tile_m: int = 0,
    tile_n: int = 0,
    rows: int = 8,
    cols: int = 8,
    transpose: bool = False,
) -> None:
    """Materialize one register-tile fragment into explicit shared storage."""
    store_rt(
        tile,
        dtype,
        data,
        offset,
        extent,
        stride,
        tile_m=tile_m,
        tile_n=tile_n,
        rows=rows,
        cols=cols,
        transpose=transpose,
    )


@T.macro
def mma_tile(acc: MMATile, a: MMATile, b: MMATile) -> None:
    for tile_m in T.unroll(acc.fragments_m, explicit=True):
        for tile_n in T.unroll(acc.fragments_n, explicit=True):
            mma(
                acc.fragment,
                a.fragment,
                b.fragment,
                acc.index(tile_m, tile_n),
                a.index(tile_m, 0),
                b.index(0, tile_n),
            )


@T.macro
def mma_ab(
    acc: RegisterTile,
    a: RegisterTile,
    b: RegisterTile,
    *,
    acc_m: int = 0,
    acc_n: int = 0,
    a_m: int = 0,
    a_n: int = 0,
    b_m: int = 0,
    b_n: int = 0,
) -> None:
    """Accumulate ``A @ B`` into one accumulator tile fragment."""
    _require_layout(acc, TileLayout.ROW_MAJOR, "accumulator", "mma_ab")
    _require_layout(a, TileLayout.ROW_MAJOR, "A", "mma_ab")
    _require_layout(b, TileLayout.ROW_MAJOR, "B", "mma_ab")
    mma(
        acc.fragment,
        a.fragment,
        b.fragment,
        acc.index(acc_m, acc_n),
        a.index(a_m, a_n),
        b.index(b_m, b_n),
    )


@T.macro
def mma_abt(
    acc: RegisterTile,
    a: RegisterTile,
    bt: RegisterTile,
    *,
    acc_m: int = 0,
    acc_n: int = 0,
    a_m: int = 0,
    a_n: int = 0,
    b_m: int = 0,
    b_n: int = 0,
) -> None:
    """Accumulate ``A @ B.T`` after ``B`` has been loaded transposed."""
    _require_layout(acc, TileLayout.ROW_MAJOR, "accumulator", "mma_abt")
    _require_layout(a, TileLayout.ROW_MAJOR, "A", "mma_abt")
    _require_layout(bt, TileLayout.TRANSPOSED, "B", "mma_abt")
    mma(
        acc.fragment,
        a.fragment,
        bt.fragment,
        acc.index(acc_m, acc_n),
        a.index(a_m, a_n),
        bt.index(b_m, b_n),
    )


@T.macro
def prefix_block_vector(
    src,
    head,
    block_index,
    dst,
    *,
    block: int,
    length: int,
    writeback=None,
    writeback_guard=True,
) -> None:
    """Compute an inclusive block-local prefix vector from a 2D source."""
    block_start = block_index * block
    acc = T.alloc_var(T.float32)
    acc = 0.0
    for idx in T.serial(block_start):
        acc += src[idx, head]
    for local_idx in T.serial(block):
        token = block_start + local_idx
        value = T.alloc_var(T.float32)
        value = 0.0
        if token < length:
            acc += src[token, head]
            value = acc
            if writeback is not None:
                if writeback_guard:
                    writeback[token, head] = value
        dst[local_idx] = value


@T.macro
def row_max(src, dst: RowVector, *, rows: int, cols: int, clear: bool = True) -> None:
    """Compute per-row maxima over a materialized scalar tile."""
    for row in T.Parallel(rows):
        acc = T.alloc_var(dst.dtype)
        if clear:
            acc = T.cast(-3.4028234663852886e38, dst.dtype)
        else:
            acc = dst.values[row]
        for col in T.serial(cols):
            acc = T.max(acc, src[row, col])
        dst.values[row] = acc


@T.macro
def row_sum(src, dst: RowVector, *, rows: int, cols: int, clear: bool = True) -> None:
    """Compute per-row sums over a materialized scalar tile."""
    for row in T.Parallel(rows):
        acc = T.alloc_var(dst.dtype)
        if clear:
            acc = T.cast(0, dst.dtype)
        else:
            acc = dst.values[row]
        for col in T.serial(cols):
            acc += src[row, col]
        dst.values[row] = acc


@T.macro
def mul_row(src, vec: RowVector, *, rows: int, cols: int) -> None:
    """Scale each materialized scalar-tile row by a row-vector value."""
    for row, col in T.Parallel(rows, cols):
        src[row, col] *= vec.values[row]


@T.macro
def div_row(src, vec: RowVector, *, rows: int, cols: int) -> None:
    """Divide each materialized scalar-tile row by a row-vector value."""
    for row, col in T.Parallel(rows, cols):
        src[row, col] /= vec.values[row]
