"""TileIR typed intermediate representation — ir package."""

from .types import DType, TileType, MemSpace, Layout, dtype, dtype_from_mlir, DTYPES
from .value import Value, Block, Region, fresh_value
from .ops import (
    Effect,
    TileOp,
    operand,
    buffer_operand,
    attribute,
    nested_block,
    # Control flow
    Loop,
    IfElse,
    Break,
    Continue,
    GridSync,
    # Data movement
    Copy,
    TmaCopy,
    Load,
    Store,
    Fill,
    PartitionView,
    # Compute
    Gemm,
    Tcgen05Gemm,
    Reduce,
    Cumsum,
    ThreadAllreduce,
    Elementwise,
    Cast,
    Select,
    RepeatInterleave,
    # Atomic
    AtomicRMW,
    AtomicCAS,
    # Misc
    Barrier,
    DeviceAssert,
    DebugPrint,
    DecodeI4,
    DecodeI2,
    Dp4a,
)
from .builder import IRBuilder

__all__ = [
    "DType",
    "TileType",
    "MemSpace",
    "Layout",
    "dtype",
    "dtype_from_mlir",
    "DTYPES",
    "Value",
    "Block",
    "Region",
    "fresh_value",
    "Effect",
    "TileOp",
    "operand",
    "buffer_operand",
    "attribute",
    "nested_block",
    # Control flow
    "Loop",
    "IfElse",
    "Break",
    "Continue",
    "GridSync",
    # Data movement
    "Copy",
    "TmaCopy",
    "Load",
    "Store",
    "Fill",
    "PartitionView",
    # Compute
    "Gemm",
    "Tcgen05Gemm",
    "Reduce",
    "Cumsum",
    "ThreadAllreduce",
    "Elementwise",
    "Cast",
    "Select",
    "RepeatInterleave",
    # Atomic
    "AtomicRMW",
    "AtomicCAS",
    # Misc
    "Barrier",
    "DeviceAssert",
    "DebugPrint",
    "DecodeI4",
    "DecodeI2",
    "Dp4a",
    "IRBuilder",
]
