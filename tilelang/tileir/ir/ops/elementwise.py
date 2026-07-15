"""TileIR elementwise / shape ops (constants, casts, selects, broadcasts)."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    operand,
)


@dataclasses.dataclass(eq=False)
class Constant(TileOp, opcode="constant", effect=Effect.NONE):
    """A compile-time scalar constant that produces a 0-d (scalar) tile Value.

    ``value`` — the Python int or float literal.
    ``dtype`` — target element dtype name string (e.g. "int32", "float32").

    ``emit_mlir`` calls ``ct.constant(value, tile_type=TileType<[]×dtype>)``
    and returns the 0-d MLIR tile.  The result is bound to ``results[0]``.
    The caller must pass ``result_types=(scalar_tile_type,)`` to
    ``IRBuilder.create`` so ``results[0]`` gets the correct TileType.
    """

    value: Any = attribute()
    dtype: str = attribute()

    def emit_mlir(self, ctx: Any) -> Any:
        """Emit ``ct.constant`` for a scalar (0-d) tile."""
        from tilelang.tileir.emission_utils import _mlir_element_type

        ct = ctx.ct
        loc = ctx.loc

        # Build a temporary TileType just to resolve the MLIR element type.
        from tilelang.tileir.ir.types import TileType, MemSpace, dtype as _lu

        result_ty = TileType(dtype=_lu(self.dtype), shape=(), space=MemSpace.REGISTER, layout=None)
        elem_ty = _mlir_element_type(ctx, result_ty)
        tile_type = ct.TileType.get([], elem_ty)
        return ct.constant(self.value, tile_type=tile_type, loc=loc)


@dataclasses.dataclass(eq=False)
class Elementwise(TileOp, opcode="elementwise", effect=Effect.NONE):
    """Elementwise function applied to one or more operands.

    ``fn``       — the function name dispatched at emit time.

                   *Unary*   (1 operand): "exp", "exp2", "exp10", "log",
                   "log2", "log10", "log1p", "sqrt", "rsqrt", "sin", "cos",
                   "tan", "sinh", "cosh", "tanh", "ceil", "floor", "abs",
                   "sigmoid", "neg", "not".

                   *Binary*  (2 operands): "add", "sub", "mul", "div",
                   "floordiv", "floormod", "max", "min", "pow", "atan2",
                   "andi", "ori", "xori".

                   *Ternary* (3 operands): "fma" (a*b + c).

    ``inputs``   — tuple of SSA ``Value`` objects passed to the function,
                   in positional order.  The number of inputs must match
                   the arity implied by ``fn``.

                   NOTE: this field is named ``inputs`` (not ``operands``)
                   on purpose.  ``TileOp`` defines an ``operands()`` *method*
                   that returns the declared SSA operands; a field literally
                   named ``operands`` would shadow that method on every
                   ``Elementwise`` instance, so ``op.operands()`` would raise
                   ``TypeError`` (the tuple is not callable).  The inherited
                   ``operands()`` method still works and returns ``(inputs,)``.

    ``inputs`` carries the operand Values so the emit path can look up the
    MLIR values for each input.

    ``unsigned`` — True when the operand(s) are an unsigned integer type
    (mirrors ``Reduce.src_unsigned`` / ``Gemm.lhs_unsigned``: the TileIR type
    registry alias-collapses uint dtypes to signless ints, so the emitted
    MLIR element type alone cannot tell signed from unsigned).  Only
    consulted for "max"/"min"/comparison ("eq"/"ne"/"lt"/"le"/"gt"/"ge") fns
    -- other fns (add/sub/mul/bitwise/...) are signedness-agnostic in two's
    complement, or (like "shr") already carry an explicit unsigned variant
    fn name instead. Set by ``_make_elementwise`` (sem_to_ir/expr.py) from
    the operand TIR expression's dtype.
    """

    fn: str = attribute()
    inputs: Any = operand(default=())
    unsigned: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Elementwise to MLIR.

        Dispatches on ``self.fn`` to the appropriate cuda_tile builder call.

        Unary ops use the ``_UNARY_MATH_TABLE`` in emit_helpers (name→lambda).
        Binary ops use the ``_BINARY_OP_TABLE`` in emit_helpers.
        'fma' (ternary) maps directly to ``ct.fma``.

        Returns the MLIR tile value so emit_module can bind it to results[0].
        """
        from tilelang.tileir.emission_utils import (
            _emit_elementwise,
        )

        return _emit_elementwise(self, ctx)


@dataclasses.dataclass(eq=False)
class Cast(TileOp, opcode="cast", effect=Effect.NONE):
    """Type-cast a tile to a new element type.

    ``src``       — the source tile Value.
    ``dtype``     — target dtype name string (e.g. "float16", "int32").
    ``src_dtype`` — original source dtype NAME string as known at lowering
                    time (e.g. "uint8"), threaded so the emit path can pick
                    zero- vs sign-extension on int widening.  The
                    ``TileType`` alias map collapses uint{N} -> signless
                    Int{N}, so the MLIR element type alone cannot tell signed
                    from unsigned.  Empty when unknown (defaults to the
                    MLIR-derived name, i.e. SIGNED).
    ``bitcast``   — when True, reinterpret the source bits as ``dtype`` (a
                    ``ct.bitcast``) instead of a value-converting numeric cast.
                    Used to lower ``tir.reinterpret``. Requires equal bit width.
    """

    src: Any = operand()
    dtype: str = attribute()
    src_dtype: str = attribute(default="")
    bitcast: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Cast to MLIR.

        Resolves ``src`` from the value_map, then applies the appropriate
        cuTile cast primitive (ftof / itof / ftoi / exti / trunci) based on
        the source and target MLIR element types.

        Returns the cast MLIR tile value for binding to results[0].
        """
        from tilelang.tileir.emission_utils import (
            _emit_cast,
        )

        return _emit_cast(self, ctx)


@dataclasses.dataclass(eq=False)
class Select(TileOp, opcode="select", effect=Effect.NONE):
    """Element-wise select: ``result = cond ? true_val : false_val``.

    ``cond``      — boolean tile Value (predicate mask).
    ``true_val``  — value selected where cond is True.
    ``false_val`` — value selected where cond is False.
    """

    cond: Any = operand()
    true_val: Any = operand()
    false_val: Any = operand()

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Select to MLIR via ``ct.select``.

        Resolves all three operands from the value_map and emits a cuTile
        select op.  Returns the MLIR tile value for binding to results[0].
        """
        from tilelang.tileir.emission_utils import (
            _emit_select,
        )

        return _emit_select(self, ctx)


@dataclasses.dataclass(eq=False)
class Iota(TileOp, opcode="iota", effect=Effect.NONE):
    """Axis-indexed iota tile for use in T.Parallel bodies.

    Produces a tile of shape ``tile_shape`` where each element at position
    ``(..., i, ...)`` has value ``i`` along ``axis``, broadcast across all
    other dimensions.  This is needed to lower T.Parallel loop-variable
    references (e.g. ``i`` in ``m_idx * 64 + i >= k * 64 + j``) into tile
    expressions.

    For example, ``Iota(tile_shape=(64, 64), axis=0)`` produces:
    ``[[0,0,...], [1,1,...], ..., [63,63,...]]``   (shape 64×64, dtype i32)

    And ``Iota(tile_shape=(64, 64), axis=1)`` produces:
    ``[[0,1,...,63], [0,1,...,63], ..., [0,1,...,63]]``  (shape 64×64, dtype i32)
    """

    tile_shape: tuple = attribute()
    axis: int = attribute()

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Iota to MLIR via ct.iota + ct.reshape + ct.broadcast."""
        ct = ctx.ct
        loc = ctx.loc

        shape = list(self.tile_shape)
        axis = self.axis
        n = shape[axis]

        # ct.iota(n, Int32) produces tile<nxi32> = [0, 1, ..., n-1]
        iota_1d = ct.iota(n, ct.Int32, loc=loc)

        if len(shape) == 1:
            return iota_1d

        # Reshape to [1, ..., n, ..., 1] — n at the right axis, 1 elsewhere.
        rank_shape = [1] * len(shape)
        rank_shape[axis] = n
        iota_nd = ct.reshape(rank_shape, iota_1d, loc=loc)

        # Broadcast to full tile_shape.
        return ct.broadcast(shape, iota_nd, loc=loc)


@dataclasses.dataclass(eq=False)
class Broadcast(TileOp, opcode="broadcast", effect=Effect.NONE):
    """Reshape + broadcast a tile to a target shape.

    Embeds the source tile (shape ``src_shape``) into a higher-dimensional
    ``target_shape`` by first reshaping to a rank-N all-ones tensor with the
    source dimension at the indicated axis, then broadcasting.

    For example:

    * ``Broadcast(src, src_shape=(64,), target_shape=(64, 64), axis=0)``
      produces ``tile<64x64>`` where each row is the source tile.
    * ``Broadcast(src, src_shape=(64,), target_shape=(64, 64), axis=1)``
      produces ``tile<64x64>`` where each column is the source tile.

    This is used to align lower-dimensional local-buffer tiles (e.g.
    ``dA_cs_m_local : tile<64xf32>``) to the full 2-D tile context so that
    arithmetic like ``dA_cs_m[i] - dA_cs_k[j]`` produces the correct outer-
    difference matrix.
    """

    src: Any = operand()
    src_shape: tuple = attribute()
    target_shape: tuple = attribute()
    axis: int = attribute()
    # Optional explicit intermediate shape for MULTI-axis embeds (e.g. a 2D
    # ``weights[bq_i, h_i]`` into a 3D parallel tile: (4,32) → (1,4,32) →
    # (256,4,32)). Empty tuple → derive [1,…,src_extent,…,1] from ``axis``
    # (the single-axis form).
    reshape_shape: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Broadcast to MLIR via ct.reshape + ct.broadcast.

        Steps:
        1. Resolve the source tile from ctx.value_map.
        2. Reshape to ``reshape_shape`` when set (multi-axis embed), else to
           [1,...,src_shape[0],...,1] at ``axis``.
        3. Broadcast to target_shape.
        """
        ct = ctx.ct
        loc = ctx.loc

        src_mlir = ctx.lookup(self.src)
        # Ensure it's a Tile object.
        from tilelang.tileir.emission_utils import _as_tile

        src_tile = _as_tile(ctx, src_mlir)

        target = list(self.target_shape)
        rank = len(target)

        if self.reshape_shape:
            reshape_shape = list(self.reshape_shape)
        else:
            # Build intermediate reshape shape: [1, ..., src_extent, ..., 1].
            # The source may be a scalar (shape=[]) → reshape to [1]*rank.
            src_extent = self.src_shape[0] if self.src_shape else 1
            reshape_shape = [1] * rank
            reshape_shape[self.axis] = src_extent

        # Reshape the tile (src may already be rank-1 or rank-0).
        if list(src_tile.shape) != reshape_shape:
            src_tile = ct.reshape(reshape_shape, src_tile, loc=loc)

        # Broadcast to full target_shape.
        if list(src_tile.shape) == target:
            return src_tile
        return ct.broadcast(target, src_tile, loc=loc)


@dataclasses.dataclass(eq=False)
class Permute(TileOp, opcode="permute", effect=Effect.NONE):
    """Permute the axes of a register/shared tile (the tile analogue of
    ``torch.permute``): result axis ``k`` reads source axis ``perm[k]``.

    Realises a *transposed read* of a fragment — e.g. ``x_local[j, i]`` inside
    ``for i, j in T.Parallel(...)``.  The loaded tile carries the buffer's axis
    order, but the surrounding parallel context lays tiles out in
    ``ordered_vars`` order, so the axes must be permuted to match.  This is the
    read-side mirror of ``Store.val_perm`` (which permutes on the write side).
    """

    src: Any = operand()
    perm: tuple = attribute()

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower to ``ct.permute(src_tile, perm)``."""
        from tilelang.tileir.emission_utils import _as_tile

        ct = ctx.ct
        src_tile = _as_tile(ctx, ctx.lookup(self.src))
        return ct.permute(src_tile, list(self.perm), loc=ctx.loc)


@dataclasses.dataclass(eq=False)
class RepeatInterleave(TileOp, opcode="repeat_interleave", effect=Effect.NONE):
    """Repeat every element along one axis of a register tile.

    The operation is collective: it inserts a singleton dimension immediately
    after ``axis``, broadcasts that dimension to ``repeats``, then flattens the
    repeated dimension back into ``axis``.  For example, ``(8, 4)`` with
    ``axis=1`` and ``repeats=2`` becomes ``(8, 8)`` with adjacent duplicates.
    """

    src: Any = operand()
    axis: int = attribute()
    repeats: int = attribute()

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower to reshape + broadcast + reshape."""
        from tilelang.tileir.errors import _UnsupportedTileIRNode
        from tilelang.tileir.emission_utils import _as_tile

        ct = ctx.ct
        loc = ctx.loc
        src_tile = _as_tile(ctx, ctx.lookup(self.src))
        src_shape = list(src_tile.tile_type.shape)

        if self.axis < 0 or self.axis >= len(src_shape):
            raise _UnsupportedTileIRNode(f"RepeatInterleave axis {self.axis} is out of range for shape {src_shape}")
        if self.repeats <= 0:
            raise _UnsupportedTileIRNode(f"RepeatInterleave repeats must be positive, got {self.repeats}")

        expanded_shape = list(src_shape)
        expanded_shape.insert(self.axis + 1, 1)
        broadcast_shape = list(expanded_shape)
        broadcast_shape[self.axis + 1] = self.repeats
        result_shape = list(src_shape)
        result_shape[self.axis] *= self.repeats

        expanded = ct.reshape(expanded_shape, src_tile, loc=loc)
        repeated = ct.broadcast(broadcast_shape, expanded, loc=loc)
        return ct.reshape(result_shape, repeated, loc=loc)
