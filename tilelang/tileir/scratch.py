"""Layout of per-tile-block scratch supplied by the launch runtime."""

from __future__ import annotations

from math import prod


def scratch_layout(buffers: dict) -> tuple[int, dict]:
    """Pack demoted buffers into disjoint, 16-byte-aligned byte ranges."""
    offsets = {}
    size = 0
    for value, (shape, _dtype) in buffers.items():
        offsets[value] = size
        bits = value.type.dtype.bitwidth
        # Boolean pointers address bytes, even though their Tile IR type is i1.
        bits = 8 if bits == 1 else bits
        size += (prod(int(dim) for dim in shape) * bits + 7) // 8
        size = (size + 15) // 16 * 16
    return size, offsets


def scratch_bytes_for_primfunc(prim_func) -> int:
    """Recompute scratch requirements for compilation and cache ABI validation."""
    from tvm import tirx

    has_local_buffers = False

    def visit(node):
        nonlocal has_local_buffers
        if isinstance(node, tirx.SBlock):
            has_local_buffers |= any(buffer.scope() != "global" for buffer in node.alloc_buffers)

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)
    if not has_local_buffers:
        return 0

    from .ir.builder import IRBuilder
    from .lowering.sem_to_ir import LoweringScope
    from .lowering.tir_to_sem import tir_to_sem

    program = tir_to_sem(prim_func)
    if len(program.kernels) != 1:
        raise ValueError("Scratch layout requires exactly one TileIR kernel.")
    scope = LoweringScope(program.kernels[0], program, builder=IRBuilder())
    return scratch_layout(scope.alloca_buffer_values)[0]
