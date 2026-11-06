"""Logical operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang.language import _construction
from tilelang.language import _tensor_spec
from tvm_tirx import ir as _ir

from tilelang import language as T
from tvm_tirx.ir import TensorRegion as BufferRegion

from tilelang.utils.language import get_buffer_elems
from tilelang._typing import BufferLikeType


def any_of(buffer: BufferLikeType) -> _ir.Expr:
    """Check if any element in the buffer is true.

    Args:
        buffer: Either a TVM buffer or buffer region to be checked

    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if _tensor_spec.is_buffer_like(buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, _ir.Op.get("tl.any_of"), T.access_ptr(buffer, "r"), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.source, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r.begin)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(r.begin)
        buffer_load = _ir.TensorLoad(buffer, _construction.as_indices(new_region), _construction.current_location())
        return T.call_intrin(
            return_type,
            _ir.Op.get("tl.any_of"),
            T.access_ptr(buffer_load, "r", extent=extent),
            extent,
        )
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")


def all_of(buffer: BufferLikeType) -> _ir.Expr:
    """Check if all elements in the buffer are true.

    Args:
        buffer: Either a TVM buffer or buffer region to be checked

    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if _tensor_spec.is_buffer_like(buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, _ir.Op.get("tl.all_of"), T.access_ptr(buffer, "r"), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.source, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r.begin)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(r.begin)
        buffer_load = _ir.TensorLoad(buffer, _construction.as_indices(new_region), _construction.current_location())
        return T.call_intrin(
            return_type,
            _ir.Op.get("tl.all_of"),
            T.access_ptr(buffer_load, "r", extent=extent),
            extent,
        )
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")
