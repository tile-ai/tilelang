"""Ascend fractal/NZ layout wrappers (moved out of swizzle.py)."""

# pylint: disable=invalid-name
from __future__ import annotations

import tvm_ffi

from tilelang import _ffi_api
from tilelang._typing import BufferLikeType

from .layout import Layout
from .swizzle import _get_buffer_info


@tvm_ffi.register_object("tl.AscendFractalLayout")
class _AscendFractalLayout(Layout):
    """Internal tagged representation of a canonical Ascend layout."""


def make_ascend_nz_layout(buffer: BufferLikeType):
    """
    Create an Ascend NZ (zN fractal) layout for a Cube operand buffer.

    A logical ``[.., rows, cols]`` tile maps to physical
    ``[.., rows / 16, cols / C0, 16, C0]`` -- a grid of 16xC0 fractals -- where
    ``C0 = 32 bytes / element_size`` is the Cube unit's fractal width and the
    forward map is ``(.., r, c) -> [.., r / 16, c / C0, r % 16, c % C0]``.

    Args:
        buffer: BufferLikeType (rank >= 2, constant last-2 dims)

    Returns:
        Layout: The NZ fractal layout for the buffer.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_nz_layout(buf)


def make_ascend_compact_nz_layout(buffer: BufferLikeType):
    """Create a compact staged-NZ layout for a pre-packed UB buffer.

    Unlike :func:`make_ascend_nz_layout`, this layout does not round the row
    extent of every leading stage up to 16.  A logical
    ``[..., rows, cols]`` buffer is stored as consecutive
    ``[C0-group, rows, C0]`` blocks, with all leading dimensions flattened
    into the C0-group axis.  Consequently, a padded ``rows = tile_rows + 1``
    stage has exactly the source pitch expected by
    ``ascend_nd2nz_post_copy``.

    This layout is intended for UB data that is produced directly in NZ order
    (for example with ``vsstb`` or a layout-driven ``T.copy``) and then copied
    to L1. It is not an L1 Cube operand layout. For an ND-to-NZ ``T.copy``, the
    allocation's row extent must include one trailing padding row while the
    copied destination region excludes it.
    """

    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_compact_nz_layout(buf)


def make_ascend_major_k_layout(buffer: BufferLikeType, k_align: int = 1):
    """
    Create an Ascend fractal layout for logical ``[.., MN, K]`` tiles.

    The contiguous reduce-K axis is grouped by C0 and is the outer physical
    major: ``(.., mn, k) -> [.., k / C0, mn / 16, mn % 16, k % C0]``.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_major_k_layout(buf, k_align)


def make_ascend_major_mn_layout(buffer: BufferLikeType, k_align: int = 1):
    """
    Create an Ascend fractal layout for logical ``[.., K, MN]`` tiles.

    The contiguous output M/N axis is grouped by C0 and is the outer physical
    major: ``(.., k, mn) -> [.., mn / C0, k / 16, k % 16, mn % C0]``.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_major_mn_layout(buf, k_align)


def make_ascend_l0c_layout(buffer: BufferLikeType):
    """
    Create the fixed Ascend L0C accumulator fractal layout for logical
    ``[.., M, N]`` tiles.

    The physical shape is ``[.., N / 16, M / 16, 16, 16]`` and the forward map
    is ``(.., m, n) -> [.., n / 16, m / 16, m % 16, n % 16]``.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_l0c_layout(buf)


def make_ascend_sf_layout(buffer: BufferLikeType):
    """
    Create an Ascend SF layout for MX scale-factor L1 buffers.

    A logical ``[..., MN, K]`` scale tile maps to physical
    ``[..., MN/16, K / (16 / dtype.bits), 16, 16 / dtype.bits]``.
    For uint8/int8 SF this is ``[..., MN/16, K/2, 16, 2]``.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_ascend_sf_layout(buf)


def make_strided_slice(phys_buf, layout, logical_region):
    """
    Compute a 0-origin strided buffer view for a logical sub-region on a
    layout-annotated physical buffer.

    The result is a plain strided Buffer sharing ``phys_buf``'s data pointer,
    with explicit shape (extents) and strides pointing at the sub-region's
    base element. No layout knowledge is needed by the consumer.

    Args:
        phys_buf: The physical (remapped) buffer, e.g. with NZ shape
            ``[M/16, K/C0, 16, C0]``.
        layout: The Layout whose forward map describes the physical layout.
        logical_region: One ``Range`` per layout input dim, ``[min, min+extent)``.

    Returns:
        Buffer: A strided buffer view into ``phys_buf``.
    """
    return _ffi_api.make_strided_slice(phys_buf, layout, logical_region)


class FractalLayoutInfo:
    """Semantic view of a canonical Ascend fractal layout.

    This is a Python wrapper around the C++ ``TryExtractAscendFractalLayout``
    utility. The layout kind is recorded by the canonical constructor; the
    remaining fields are derived from that tag and the buffer.

    Attributes:
        kind: 0 = MajorK, 1 = L0C (fixed 16×16), 2 = SF, 3 = MajorMN
        c0: C0 value (or 16 for L0C, pack for SF)
        row_frac: always 16
        rows: logical rows (dim -2)
        cols: logical cols (dim -1)
        outer0: full physical extent of C0-axis outer for matrix data or row16-axis outer for SF
        outer1: full physical extent of row16-axis outer for matrix data or C0-axis outer for SF
        c0_axis: 0 = row, 1 = col (pack axis for SF)
        row16_axis: 0 = row, 1 = col
    """

    __slots__ = (
        "kind",
        "c0",
        "row_frac",
        "rows",
        "cols",
        "outer0",
        "outer1",
        "c0_axis",
        "row16_axis",
    )

    def __init__(self, kind, c0, row_frac, rows, cols, outer0, outer1, c0_axis, row16_axis):
        self.kind = kind
        self.c0 = c0
        self.row_frac = row_frac
        self.rows = rows
        self.cols = cols
        self.outer0 = outer0
        self.outer1 = outer1
        self.c0_axis = c0_axis
        self.row16_axis = row16_axis

    def __repr__(self):
        axis_name = {0: "row", 1: "col"}
        kind_name = {0: "MajorK", 1: "L0C", 2: "SF", 3: "MajorMN"}
        return (
            f"FractalLayoutInfo(kind={kind_name.get(self.kind, '?')}, "
            f"c0={self.c0}, row_frac={self.row_frac}, "
            f"c0_axis={axis_name.get(self.c0_axis, '?')}, "
            f"row16_axis={axis_name.get(self.row16_axis, '?')}, "
            f"outer0={self.outer0}, outer1={self.outer1})"
        )


def try_extract_fractal_layout(layout, buffer):
    """Get semantic information from a canonical Ascend fractal layout.

    Returns:
        FractalLayoutInfo, or None when the input is undefined/invalid.

    Raises:
        InternalError: If ``layout`` was not made by a canonical Ascend layout
            constructor or a transform dropped its semantic tag.
    """
    buf, _, _ = _get_buffer_info(buffer)
    result = _ffi_api.try_extract_fractal_layout(layout, buf)
    if result is None:
        return None
    return FractalLayoutInfo(
        kind=int(result["kind"]),
        c0=result["c0"],
        row_frac=result["row_frac"],
        rows=result["rows"],
        cols=result["cols"],
        outer0=result["outer0"],
        outer1=result["outer1"],
        c0_axis=int(result["c0_axis"]),
        row16_axis=int(result["row16_axis"]),
    )
