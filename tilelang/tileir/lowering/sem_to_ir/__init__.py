"""Lower SemanticIR programs to typed TileIR blocks.

``lower_kernel`` is the package entry point. Handler modules register statement
and tile-op implementations when the package is imported.
"""

from __future__ import annotations

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.value import Block
from tilelang.tileir.errors import TileIRLoweringError
from tilelang.tileir.semantic import SemanticKernel, SemanticProgram

# Re-export the lowering primitives used by callers and handler modules.
from ._base import (  # noqa: F401
    IMPL,
    TILE_OP_IMPL,
    LoweringScope,
    impl,
    tile_op_impl,
    lookup_dtype,
    _parse_bool,
    _next_power_of_two,
    _sem_buffer_to_tile_type,
    _is_unsigned_dtype,
    _make_placeholder,
    _scalar_bool_type,
    _scalar_i32_type,
    _tir_dtype_to_tile_type,
    _op_name_from_call,
    _get_binary_op_fn,
    _UNARY_CALL_FN,
    _CMP_FNS,
)
from .expr import (  # noqa: F401
    lower_expr,
    _try_lower_fma,
    _lower_attr_expr,
    _make_elementwise,
)
from .tile_level import _lower_tile_level_expr, _try_lower_fma_tile  # noqa: F401
from .stmt import lower_stmt  # noqa: F401

# Populate the statement and tile-op registries.
from . import parallel  # noqa: F401
from . import tile_ops  # noqa: F401
from . import decoders  # noqa: F401
from . import atomic  # noqa: F401

__all__ = [
    "IMPL",
    "TILE_OP_IMPL",
    "LoweringScope",
    "lower_expr",
    "lower_stmt",
    "lower_kernel",
    "impl",
]


def _bind_dynamic_shape_symbols(
    program: SemanticProgram | None,
    scope: LoweringScope,
    builder: IRBuilder,
) -> None:
    """Bind dynamic shape symbols to placeholder Values.

    A GLOBAL param dim declared as a symbolic name (``T.dynamic``, e.g.
    ``block_indices: (batch, heads_kv, max_selected_blocks)``) already has its
    runtime value in the entry ABI — the buffer's shape argument.  Bind each
    distinct symbol name to a placeholder Value in the scalar scope and record
    (buffer, dim) in ``block.shape_bindings`` so ``emit_module`` can wire the
    placeholder to the materialized buffer's shape tile.
    """

    if program is None:
        return
    seen: set[str] = set()
    for sem_buf in program.params:
        try:
            buf_val = scope.lookup_buffer(sem_buf.name)
        except KeyError:
            continue
        for dim_idx, dim in enumerate(sem_buf.shape):
            if not isinstance(dim, str) or dim in seen:
                continue
            # A string dim that parses as an int is a serialized constant.
            try:
                int(dim)
                continue
            except ValueError:
                pass
            seen.add(dim)
            placeholder = _make_placeholder(builder, _scalar_i32_type(), name=dim)
            scope.bind(dim, placeholder)
            builder.block.shape_bindings[placeholder] = (buf_val, dim_idx)


def lower_kernel(
    sem_kernel: SemanticKernel,
    builder: IRBuilder,
    program: SemanticProgram | None = None,
    fast_math: bool = False,
) -> Block:
    """Lower a SemanticKernel to a TileIR Block.

    Parameters
    ----------
    sem_kernel :
        The SemanticKernel produced by ``tir_to_sem`` / ``extract_semantic_program``.
    builder :
        An ``IRBuilder`` that will receive the ops.
    program :
        The parent ``SemanticProgram`` (needed to resolve param buffers).
        May be ``None`` if kernel-level buffers are sufficient.

    Returns
    -------
    Block
        The ``builder.block`` after all statements have been lowered into it.
        When *program* is provided, ``block.params`` is populated with the
        GLOBAL parameter buffer ``Value`` objects (in ``program.params`` order)
        so that ``emit_module`` can bind entry-function arguments 1-to-1 with
        ``root.params``.
    """
    # The builder owns the monotonic SSA id sequence.
    scope = LoweringScope(sem_kernel, program, builder=builder, fast_math=fast_math)
    _bind_dynamic_shape_symbols(program, scope, builder)
    # Reshape-view aliases: emit redirects alias tile reads/writes through the
    # base buffer's tile (single source of truth) with reshapes.
    builder.block.buffer_aliases.update(scope.buffer_alias_values)
    lower_stmt(sem_kernel.body, scope, builder)

    # Pipeline integration: register the GLOBAL param buffer Values AND scalar
    # param Values as builder.block.params in the original PrimFunc param order.
    # emit_module's positional entry_args alignment (root.params[i] ↔
    # entry_args[i]) requires both buffer and scalar params to appear in this
    # list in the same order as prim_func.params.
    #
    # Merge buffer params and scalar params into one ordered list using the
    # explicit semantic ABI order.  The SemanticProgram intentionally does not
    # retain the original PrimFunc.
    global_param_names: set[str] = set()
    if program is not None and (program.params or getattr(program, "scalar_params", ())):
        param_order = getattr(program, "param_order", ())
        if param_order:
            values_by_name = {**scope._buffers, **scope._scalar_params}
            missing = [name for name in param_order if name not in values_by_name]
            if missing:
                raise TileIRLoweringError("SemanticProgram parameter order references unbound entry parameters: " + ", ".join(missing))
            builder.block.params = [values_by_name[name] for name in param_order]
        else:
            # Hand-built semantic programs may omit an ABI order. Preserve the
            # deterministic buffers-then-scalars convention for those tests.
            buf_values = [scope._buffers[p.name] for p in program.params if p.name in scope._buffers]
            scalar_values = [
                scope._scalar_params[sp.name] for sp in getattr(program, "scalar_params", ()) if sp.name in scope._scalar_params
            ]
            builder.block.params = buf_values + scalar_values
        global_param_names = {p.name for p in program.params}

    # Populate block.alloc_buffers with SHARED/REGISTER Values
    # so that emit_module can materialise them as zero-constant tiles before
    # walking the op list.  Only non-GLOBAL entries (alloc_shared / alloc_fragment)
    # are included; GLOBAL param Values are already in block.params.
    alloc_buf_values: list = []
    for buf in sem_kernel.alloc_buffers:
        if buf.name in scope._buffers and buf.name not in global_param_names:
            val = scope._buffers[buf.name]
            # SIMT-demoted scratch buffers are alloca memory, not SSA tiles.
            if val in scope.alloca_buffer_values:
                continue
            alloc_buf_values.append(val)
    builder.block.alloc_buffers = alloc_buf_values
    builder.block.alloca_buffers.update(scope.alloca_buffer_values)

    return builder.block
