# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang import language as T
import tvm
from tvm.tir import Buffer, BufferRegion
from tvm.ir import Range
from tvm.ir import register_op_attr, register_intrin_lowering
from tvm import tir
from typing import Union
from tilelang.utils.language import get_buffer_elems


# TODO: move this part into src to reduce runtime overhead
def any_of_op(op):
    args = op.args
    assert len(args) == 2
    buffer_address, elems = args
    return T.call_extern("bool", "tl::Any", buffer_address, elems)


register_op_attr("tl.any_of", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
register_op_attr("tl.any_of", "TScriptPrinterName", "any_of")
register_intrin_lowering("tl.any_of", target="cuda", f=any_of_op)


def any_of(buffer: Union[T.Tensor, BufferRegion]):
    """Check if any element in the buffer is true.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be checked
    
    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if isinstance(buffer, Buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, tir.op.Op.get("tl.any_of"), T.address_of(buffer), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.buffer, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(Range(r.min, 1))
        buffer = BufferRegion(buffer, new_region)
        return T.call_intrin(return_type, tir.op.Op.get("tl.any_of"), T.address_of(buffer), extent)
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")


def all_of_op(op):
    args = op.args
    assert len(args) == 2
    buffer_address, elems = args
    return T.call_extern("bool", "tl::All", buffer_address, elems)


register_op_attr("tl.all_of", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
register_op_attr("tl.all_of", "TScriptPrinterName", "all_of")
register_intrin_lowering("tl.all_of", target="cuda", f=all_of_op)


def all_of(buffer: Union[T.Tensor, BufferRegion]):
    """Check if all elements in the buffer are true.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be checked
    
    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if isinstance(buffer, Buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, tir.op.Op.get("tl.all_of"), T.address_of(buffer), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.buffer, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(Range(r.min, 1))
        buffer = BufferRegion(buffer, new_region)
        return T.call_intrin(return_type, tir.op.Op.get("tl.all_of"), T.address_of(buffer), extent)
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")
