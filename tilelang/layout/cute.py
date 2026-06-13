"""CuTe layout IR objects and layout-algebra Python API.

This module is the single Python home for everything CuTe-layout related:

* the FFI-registered IR objects (:class:`Swizzle`, :class:`IntTuple` and its leaf
  kinds :class:`IntTupleConst` / :class:`IntTuplePrimExpr` /
  :class:`IntTupleScaledBasis` and branch :class:`IntTupleTuple`,
  :class:`CuteLayout`, :class:`ComposedLayout`);
* the recovery entry points :meth:`ComposedLayout.from_tilelang` /
  :meth:`CuteLayout.from_tilelang`;
* the layout-algebra ops mirroring CuTeDSL (``cutlass.cute``):
  :func:`make_layout` (keyword ``stride=``), :func:`make_identity_layout`,
  :func:`ScaledBasis` / :func:`E`, :func:`coalesce`, :func:`composition`,
  :func:`right_inverse`, :func:`flatten`, :func:`rank`, :func:`size`,
  :func:`get` (and ``layout[idx]``), and the TMA box derivation
  :func:`derive_tma_tile`.

All of these are registered C++-side under the ``tl.cute.*`` FFI namespace and
surfaced through :mod:`tilelang.layout._cute_ffi_api`. Import as
``from tilelang.layout import cute`` and call ``cute.coalesce(...)`` etc.
"""

from __future__ import annotations

import tvm
import tvm_ffi
from tvm.ir.base import Node
from tvm.runtime import Scriptable

from tilelang._typing import BufferLikeType
from . import _cute_ffi_api


# ---------------------------------------------------------------------------
# IR objects (FFI-registered; keys must match the C++ TVM_FFI type keys).
# ---------------------------------------------------------------------------
@tvm_ffi.register_object("tl.cute.Swizzle")
class Swizzle(Node, Scriptable):
    """A CuTe-style XOR swizzle functor <b_bits, m_base, s_shift>."""

    b_bits: int
    m_base: int
    s_shift: int

    @property
    def is_swizzled(self) -> bool:
        """Whether the layout applies an XOR swizzle (b_bits > 0)."""
        return self.b_bits > 0

    def recast(self, old_bits: int, new_bits: int) -> Swizzle:
        """Reinterpret the swizzle when the buffer is viewed as a different
        element type. Only ``m_base`` shifts (by ``log2(old_bits/new_bits)``);
        ``b_bits`` and ``s_shift`` are preserved. Sizes are element widths in
        bits and must be byte-aligned powers of two."""
        return _cute_ffi_api.swizzle_recast(self, int(old_bits), int(new_bits))


@tvm_ffi.register_object("tl.cute.IntTuple")
class IntTuple(Node, Scriptable):
    """Base of a CuTe hierarchical IntTuple: either a scalar leaf
    (:class:`IntTupleConst`, :class:`IntTuplePrimExpr`,
    :class:`IntTupleScaledBasis`) or a tuple branch (:class:`IntTupleTuple`).
    Dispatch by ``isinstance`` on the concrete kind."""


@tvm_ffi.register_object("tl.cute.IntTupleConst")
class IntTupleConst(IntTuple):
    """A compile-time integer leaf."""

    value: int


@tvm_ffi.register_object("tl.cute.IntTuplePrimExpr")
class IntTuplePrimExpr(IntTuple):
    """A dynamic (runtime-valued) integer leaf, carrying a PrimExpr ``value``."""

    value: object


@tvm_ffi.register_object("tl.cute.IntTupleScaledBasis")
class IntTupleScaledBasis(IntTuple):
    """A CuTe ScaledBasis leaf ``value * E<mode...>``: ``value`` is the scalar
    scale leaf (a const or primexpr) and the basis ``mode`` (CuTe's ScaledBasis
    mode-path) identifies the unit basis vector. Mirrors CuTeDSL
    ``cute.ScaledBasis`` (``.value`` / ``.mode``)."""

    value: IntTuple
    basis: list

    @property
    def mode(self) -> list:
        """The basis mode-path (CuTeDSL ``ScaledBasis.mode``)."""
        return list(self.basis)


@tvm_ffi.register_object("tl.cute.IntTupleTuple")
class IntTupleTuple(IntTuple):
    """A branch of an IntTuple: a tuple of :class:`IntTuple` children."""

    fields: list


@tvm_ffi.register_object("tl.cute.Layout")
class CuteLayout(Node, Scriptable):
    """A CuTe layout described by hierarchical ``shape`` and ``stride``
    :class:`IntTuple` trees. ``shape`` / ``stride`` expose the flattened per-leaf
    integer lists (the common view)."""

    @property
    def shape(self) -> list:
        """The flattened per-leaf extents as a list of ints."""
        return list(_cute_ffi_api.layout_flat_shape(self))

    @property
    def stride(self) -> list:
        """The flattened per-leaf strides as a list of ints. Errors if any leaf
        is dynamic or a ScaledBasis; use :attr:`stride_leaves` for those."""
        return list(_cute_ffi_api.layout_flat_stride(self))

    @property
    def stride_leaves(self) -> list:
        """The flattened stride leaves as :class:`IntTuple` leaf objects
        (preserves dynamic :class:`IntTuplePrimExpr` and
        :class:`IntTupleScaledBasis` leaves)."""
        return list(_cute_ffi_api.layout_stride_leaves(self))

    def __getitem__(self, idx: int) -> CuteLayout:
        """The ``idx``-th sublayout (CuTe ``get<idx>``, tuple-like syntax)."""
        return _cute_ffi_api.layout_get(self, int(idx))

    @staticmethod
    def from_tilelang(layout):
        """The affine special case of :meth:`ComposedLayout.from_tilelang`:
        recover a TileLang layout as a flat :class:`CuteLayout` when it has no
        swizzle and zero offset, else ``None``."""
        return _cute_ffi_api.layout_from_tilelang(layout)


@tvm_ffi.register_object("tl.cute.ComposedLayout")
class ComposedLayout(Node, Scriptable):
    """A CuTe ComposedLayout ``Swizzle o offset o Layout``. Evaluating at a
    coordinate ``x`` gives ``swizzle.apply(offset + layout(x))``."""

    swizzle: Swizzle
    offset: int
    layout: CuteLayout

    def recast(self, old_bits: int, new_bits: int) -> ComposedLayout:
        """Recast the whole composed layout into a different element width: the
        swizzle's ``m_base`` shifts and the plain layout's strides/offset scale
        by ``log2(old_bits/new_bits)``. Sizes must be byte-aligned powers of
        two."""
        return _cute_ffi_api.composed_layout_recast(self, int(old_bits), int(new_bits))

    @staticmethod
    def from_tilelang(layout, buffer: BufferLikeType = None):
        """Recover an arbitrary CUTLASS/CuTe XOR swizzle over an affine layout
        from a TileLang layout, as a CuTe :class:`ComposedLayout`
        (``Swizzle o offset o Layout``).

        Returns ``.swizzle`` (with ``b_bits``, ``m_base``, ``s_shift``,
        ``is_swizzled``) and ``.layout`` (the recovered unswizzled affine
        layout). A non-swizzled (linear) layout is reported with
        ``swizzle.b_bits == 0``. ``None`` is returned when the layout cannot be
        analyzed.

        Detection is dtype-agnostic and reports addresses in element-offset
        positions. When ``buffer`` is given, the result is recast into
        byte-address space using its element size (the CuTe/CUTLASS convention
        where swizzles act on byte addresses)."""
        mode = _cute_ffi_api.composed_layout_from_tilelang(layout)
        if buffer is None or mode is None:
            return mode
        from tilelang.layout.swizzle import _get_buffer_info

        _, _, dtype = _get_buffer_info(buffer)
        return mode.recast(int(tvm.DataType(dtype).bits), 8)


def _as_leaf(s) -> IntTuple:
    """Coerce a scalar entry to an :class:`IntTuple` leaf: leaves pass through, a
    Python int becomes :class:`IntTupleConst`, a :class:`ScaledBasis` lowers to
    :class:`IntTupleScaledBasis`, and a PrimExpr (or other) becomes
    :class:`IntTuplePrimExpr`."""
    if isinstance(s, IntTuple):
        return s
    if isinstance(s, ScaledBasis):
        return s.to_leaf()
    if isinstance(s, int):
        return _cute_ffi_api.make_int_const(int(s))
    return _cute_ffi_api.make_int_expr(s)


class ScaledBasis:
    """A CuTe scaled-basis stride element ``value * E<mode...>``, mirroring
    CuTeDSL's :class:`cutlass.cute.ScaledBasis`. ``value`` is the scale (an int,
    PrimExpr, or scalar :class:`IntTuple` leaf) and ``mode`` is the basis
    mode-path (an int or list of ints). Pass instances as ``make_layout`` strides
    (e.g. ``make_layout((4, 4), stride=(E(0), E(1)))``)."""

    def __init__(self, value, mode):
        self._value = value
        self._mode = [mode] if isinstance(mode, int) else [int(m) for m in mode]

    @property
    def value(self):
        return self._value

    @property
    def mode(self) -> list:
        return list(self._mode)

    def to_leaf(self) -> IntTupleScaledBasis:
        """Lower to the FFI :class:`IntTupleScaledBasis` leaf object."""
        v = self._value
        v = v if isinstance(v, IntTuple) else _cute_ffi_api.make_int_const(int(v)) if isinstance(v, int) else _cute_ffi_api.make_int_expr(v)
        return _cute_ffi_api.make_scaled_basis(v, self._mode)

    def __repr__(self):
        return f"ScaledBasis({self._value}, {self._mode})"


def E(mode) -> ScaledBasis:
    """A unit CuTe basis element ``1 * E<mode...>`` (CuTeDSL ``cute.E``). ``mode``
    is an int or list of ints identifying the basis axis."""
    return ScaledBasis(1, mode)


def make_layout(shape, stride=None):
    """Build a flat :class:`CuteLayout` from per-mode ``shape`` and ``stride``
    (fastest mode first), mirroring CuTeDSL ``cute.make_layout(shape,
    stride=...)``. ``stride`` is keyword-only and, if omitted, defaults to the
    compact column-major stride of ``shape``. ``stride`` entries may be Python
    ints, PrimExprs, :class:`ScaledBasis`/:class:`E`, or :class:`IntTuple`
    leaves."""
    shape = [int(s) for s in shape]
    if stride is None:
        stride, acc = [], 1
        for s in shape:
            stride.append(acc)
            acc *= s
    if all(isinstance(s, int) for s in stride):
        return _cute_ffi_api.make_layout(shape, [int(s) for s in stride])
    return _cute_ffi_api.make_layout_leaves(shape, [_as_leaf(s) for s in stride])


def make_identity_layout(shape):
    """CuTe ``make_identity_layout`` / ``make_basis_like``: a flat layout over
    ``shape`` whose per-mode strides are the unit ScaledBases ``E<k>``."""
    return _cute_ffi_api.make_identity_layout([int(s) for s in shape])


def rank(layout) -> int:
    """CuTe ``rank``: the number of top-level modes of ``layout``."""
    return int(_cute_ffi_api.layout_rank(layout))


def size(layout) -> int:
    """CuTe ``size``: the product of the layout's shape (its domain size)."""
    return int(_cute_ffi_api.layout_size(layout))


def get(layout, idx) -> CuteLayout:
    """CuTe ``get<idx>``: the ``idx``-th sublayout (same as ``layout[idx]``)."""
    return _cute_ffi_api.layout_get(layout, int(idx))


def flatten(layout) -> CuteLayout:
    """CuTe ``flatten``: a depth-1 layout with the same leaves in order."""
    return _cute_ffi_api.layout_flatten(layout)


def coalesce(layout):
    """CuTe ``coalesce``: merge contiguous adjacent modes, drop size-1 modes. The
    result is flat (depth <= 1)."""
    return _cute_ffi_api.coalesce(layout)


def right_inverse(layout):
    """CuTe ``right_inverse``: the layout ``r`` with ``layout(r(i)) == i``."""
    return _cute_ffi_api.right_inverse(layout)


def composition(lhs, rhs):
    """CuTe ``composition`` lhs o rhs: ``result(c) == lhs(rhs(c))``."""
    return _cute_ffi_api.composition(lhs, rhs)


def derive_tma_tile_layouts(gmem, smem_plain, tile_shape):
    """Derive the faithful-CuTe TMA decomposition (CuTe ``construct_tma_gbasis``).
    Returns the three :class:`CuteLayout`s ``(box, rest_gmem, rest_smem)`` or
    ``None`` when not TMA-expressible. ``gmem`` is the global tensor as a
    :class:`CuteLayout` (extents as shape, element strides as stride; may be
    dynamic). See :func:`derive_tma_tile` for the decoded tuple form."""
    out = _cute_ffi_api.derive_tma_tile(gmem, smem_plain, [int(s) for s in tile_shape])
    if out is None:
        return None
    return tuple(out)


def derive_tma_tile(gmem, smem_plain, tile_shape):
    """Decode :func:`derive_tma_tile_layouts` into ``(box, rest)`` or ``None``:

    * ``box`` -- list of ``(extent, axis)`` per descriptor box mode (fastest
      first); the single TMA descriptor covers these contiguous (unit gmem step)
      modes.
    * ``rest`` -- list of ``(extent, scale, axis, smem_stride)`` iteration modes
      replayed as separate TMA instructions: for digit ``d`` in ``range(extent)``
      the gmem coord of ``axis`` shifts by ``d*scale`` and the SMEM pointer by
      ``d*smem_stride``. CuTe truncates the box at the first non-contiguous global
      mode; those (and any >256 box overflow) land here.
    """
    out = derive_tma_tile_layouts(gmem, smem_plain, tile_shape)
    if out is None:
        return None
    box_l, rest_gmem, rest_smem = out
    box = [(int(e), int(s.basis[0])) for e, s in zip(box_l.shape, box_l.stride_leaves)]
    rest = []
    for e, sg, ss in zip(rest_gmem.shape, rest_gmem.stride_leaves, rest_smem.stride_leaves):
        e = int(e)
        if e == 1:  # scalar (1):(0) -> no rest
            continue
        rest.append((e, int(sg.value.value), int(sg.basis[0]), int(ss.value)))
    return box, rest
