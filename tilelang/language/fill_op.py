"""Fill operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang.language import _intrinsics
from tilelang.language import _tensor_spec
from tvm_tirx import ir as _ir
from tilelang._typing import BufferLikeType

from tilelang.language.common import has_let_value, get_let_value
from tilelang.language.utils import _normalize_annotations
from tilelang.utils.language import get_buffer_region_from_load, to_buffer_region


def fill(buffer: BufferLikeType, value: _ir.Expr, annotations: dict | None = None) -> _ir.Expr:
    """Fill a buffer or buffer region with a specified value.

    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with
        annotations: Optional annotations to attach to the fill call

    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    # Resolve scalar/slice aliases, preserving a native tensor definition.
    if isinstance(buffer, _ir.Var) and not _tensor_spec.is_buffer_like(buffer) and has_let_value(buffer):
        buffer = get_let_value(buffer)

    # Build tl.region as argument
    if _tensor_spec.is_buffer_like(buffer):
        extents = list(buffer.shape)
    elif isinstance(buffer, _ir.TensorRegion):
        extents = [r.extent for r in buffer.region]
    elif isinstance(buffer, _ir.TensorLoad):
        region = get_buffer_region_from_load(buffer)
        if region is not None:
            extents = [r.extent for r in region.region]
        else:
            extents = [_ir.IntImm("int32", 1) for _ in buffer.indices]
    else:
        extents = []
    return _intrinsics.call_intrin(
        "handle",
        _ir.Op.get("tl.tileop.fill"),
        to_buffer_region(buffer, access_type="w", extents=extents),
        value,
        annotations=_normalize_annotations(annotations),
    )


def clear(buffer: BufferLikeType, annotations: dict | None = None) -> _ir.Expr:
    """Clear a buffer by filling it with zeros.

    Args:
        buffer: Either a TVM buffer or a variable that contains a buffer region
        annotations: Optional annotations forwarded to the underlying fill

    Returns:
        A fill operation that sets the buffer contents to zero

    Raises:
        ValueError: If the buffer variable contains an invalid buffer region
    """
    if isinstance(buffer, _ir.Var) and not _tensor_spec.is_buffer_like(buffer) and has_let_value(buffer):
        buffer_region = get_let_value(buffer)  # Get the actual buffer region from variable
        if isinstance(buffer_region, _ir.TensorRegion):
            return fill(buffer_region, 0, annotations=annotations)
        elif isinstance(buffer_region, _ir.TensorLoad):
            region = get_buffer_region_from_load(buffer_region)
            if region is None:
                raise ValueError(f"Invalid buffer region: {buffer_region}, type: {type(buffer_region)}")
            return fill(region, 0, annotations=annotations)
        else:
            raise ValueError(f"Invalid buffer region: {buffer_region}, type: {type(buffer_region)}")
    return fill(buffer, 0, annotations=annotations)
