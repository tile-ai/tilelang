"""Ascend on-chip buffer allocation: L1 (CBuf) and the L0A/L0B/L0C Cube scopes."""

from __future__ import annotations

from tilelang._typing import DType, ShapeType
from tilelang.language.allocate import _with_span
from tvm import DataType
from tvm.script import tirx as T
from tvm.tirx.buffer import Buffer
from tvm.tirx.script.builder.ir import sblock_attr


def alloc_shared(shape: ShapeType, dtype: DType, scope="shared.dyn") -> Buffer:
    """Allocate a UB buffer.

    Same surface as the common ``T.alloc_shared`` minus its CUDA-only bool
    workaround (bool buffers there downgrade to the static "shared" scope
    because the smem-merge pass cannot handle bool): Ascend UB handles bool
    buffers in the requested scope directly.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l1(shape: ShapeType, dtype: DType, scope="shared.l1") -> Buffer:
    """Allocate an L1 buffer (__cbuf__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l1"
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0a(shape: ShapeType, dtype: DType, scope="shared.l0a") -> Buffer:
    """Allocate an L0A buffer (__ca__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0b(shape: ShapeType, dtype: DType, scope="shared.l0b") -> Buffer:
    """Allocate an L0B buffer (__cb__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0c(shape: ShapeType, dtype: DType, scope="shared.l0c", layout: bool = True) -> Buffer:
    """Allocate an L0C buffer (__cc__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l0c"
        layout: Whether to annotate the fixed Ascend L0C accumulator layout.
    """
    buf = _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))
    if layout:
        from tilelang.layout import make_ascend_l0c_layout

        if len(buf.shape) >= 2:
            sblock_attr({"layout_map": {buf.data: make_ascend_l0c_layout(buf)}})
    return buf


# One E8M0 scale factor covers 32 K elements; wider SF dtypes pack several
# scales per element (uint16 = one pair per 64 K elements).
_MX_SF_K_PER_SCALE_BYTE = 32


def _alloc_l0_sf(buf: Buffer, expected_scope: str, sf_dtype: DType, sf_shape: ShapeType | None) -> Buffer:
    assert buf.scope() == expected_scope, f"expected a {expected_scope} data tile, got scope {buf.scope()} for {buf.name}"
    if sf_shape is None:
        assert len(buf.shape) == 2, (
            f"MX SF handle default shape needs a plain 2-D data tile, got shape {buf.shape} for {buf.name}; "
            "a manually multi-buffered tile carries leading version dims — pass sf_shape explicitly with the same leading dims"
        )
        k_per_sf = _MX_SF_K_PER_SCALE_BYTE * (DataType(sf_dtype).bits // 8)
        rows, k = buf.shape[-2], buf.shape[-1]
        if isinstance(k, (int,)) or hasattr(k, "value"):
            k_value = int(k)
            assert k_value % k_per_sf == 0, (
                f"data tile K extent {k_value} of {buf.name} is not divisible by {k_per_sf} (K elements per {sf_dtype} scale element); pass sf_shape explicitly"
            )
        sf_shape = (rows, k // k_per_sf)
    return _with_span(T.sblock_alloc_buffer(sf_shape, sf_dtype, scope=expected_scope + ".sf"))


def alloc_l0a_sf(buf: Buffer, sf_dtype: DType = "uint16", sf_shape: ShapeType | None = None) -> Buffer:
    """Return the MX scale-factor handle of an L0A data tile.

    The handle (scope ``shared.l0a.sf``) materializes no storage: the
    hardware keys a tile's scale slots to the tile's own address. Load
    scales with ``T.copy(sf_l1, handle)`` and pass the handle to
    ``T.gemm_blockscaled`` as ``SFA``; that gemm binds the handle to its A
    tile — ``buf`` only supplies the default shape and a scope check. Scale
    slots are sticky, so one scale load may serve several data loads (see
    testing/ascend/language/test_tilelang_ascend_mx_sf_slots.py).

    The default shape assumes an untransposed ``(rows, K)`` data tile with
    one scale per 32 K elements packed into ``sf_dtype`` (``uint16`` = one
    pair per 64 K elements); pass ``sf_shape`` explicitly for transposed
    tiles or other packings.
    """
    return _alloc_l0_sf(buf, "shared.l0a", sf_dtype, sf_shape)


def alloc_l0b_sf(buf: Buffer, sf_dtype: DType = "uint16", sf_shape: ShapeType | None = None) -> Buffer:
    """Return the MX scale-factor handle of an L0B data tile.

    See :func:`alloc_l0a_sf`; identical semantics for the L0B slot shadow.
    """
    return _alloc_l0_sf(buf, "shared.l0b", sf_dtype, sf_shape)


__all__ = [
    "alloc_shared",
    "alloc_l1",
    "alloc_l0a",
    "alloc_l0a_sf",
    "alloc_l0b",
    "alloc_l0b_sf",
    "alloc_l0c",
]
