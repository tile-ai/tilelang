"""TileOp schema + concrete op catalog (split by category).

Re-exports the field-marker machinery and ``TileOp`` base from ``_base`` and
every concrete op from its category submodule, preserving the historical
``from tilelang.tileir.ir.ops import X`` public API.
"""

from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
    nested_block,
    operand,
)
from tilelang.tileir.ir.ops.control_flow import Break, Continue, GridSync, IfElse, Loop
from tilelang.tileir.ir.ops.data_movement import (
    Copy,
    CopyGather,
    CopyScatter,
    Fill,
    GatherLoad,
    Load,
    PartitionView,
    Store,
    TmaCopy,
    TransposeCopy,
)
from tilelang.tileir.ir.ops.compute import (
    Cumsum,
    Dp4a,
    Gemm,
    GemmScaled,
    Reduce,
    Tcgen05Gemm,
    ThreadAllreduce,
)
from tilelang.tileir.ir.ops.elementwise import (
    Broadcast,
    Cast,
    Constant,
    Elementwise,
    Iota,
    Permute,
    RepeatInterleave,
    Select,
)
from tilelang.tileir.ir.ops.atomics import AtomicCAS, AtomicLoad, AtomicRMW, AtomicStore
from tilelang.tileir.ir.ops.misc import (
    Barrier,
    DebugPrint,
    DecodeFp4Twiddling,
    DecodeI2,
    DecodeI4,
    DeviceAssert,
)

__all__ = [
    "Effect",
    "operand",
    "buffer_operand",
    "attribute",
    "nested_block",
    "TileOp",
    # Control flow
    "Loop",
    "IfElse",
    "Break",
    "Continue",
    "GridSync",
    # Data movement
    "Copy",
    "CopyGather",
    "CopyScatter",
    "TmaCopy",
    "TransposeCopy",
    "GatherLoad",
    "Load",
    "Store",
    "Fill",
    "PartitionView",
    # Compute
    "Gemm",
    "GemmScaled",
    "Tcgen05Gemm",
    "Reduce",
    "Cumsum",
    "ThreadAllreduce",
    "Elementwise",
    "Cast",
    "Select",
    "Iota",
    "Broadcast",
    "Permute",
    "RepeatInterleave",
    # Atomic
    "AtomicRMW",
    "AtomicLoad",
    "AtomicStore",
    "AtomicCAS",
    # Misc
    "Constant",
    "Barrier",
    "DeviceAssert",
    "DebugPrint",
    "DecodeI4",
    "DecodeI2",
    "DecodeFp4Twiddling",
    "Dp4a",
]
