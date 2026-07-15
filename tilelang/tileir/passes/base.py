"""TileIR pass infrastructure.

Provides:
  - ``Pass``         — type alias for a pass callable.
  - ``PassContext``  — carrier for cross-pass results and analysis caches.
  - ``run_pipeline`` — run an ordered list of passes over a root Block.
  - ``walk_block``   — recursively visit every op in a Block (pre-order).

Design notes
------------
Passes are *mutating* procedures: ``Pass = Callable[[Block, PassContext], None]``.
This mirrors cuTile's ``_passes`` design (``Block -> None``) while threading a
shared ``PassContext`` so analysis results from one pass are visible to later
passes (e.g. dataflow analysis feeds token ordering).

``walk_block`` uses the ``TileOp.nested_blocks()`` API (``_block_names``) to
recurse into control-flow bodies without importing any concrete op class.
It does NOT import the MLIR-emission walker because
that helper is tightly coupled to an MLIR ``EmitContext`` and does MLIR
emission work; this function only does IR traversal.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tilelang.tileir.ir.ops import TileOp
from tilelang.tileir.ir.value import Block

__all__ = [
    "Pass",
    "PassContext",
    "run_pipeline",
    "walk_block",
]


# PassContext — cross-pass result carrier


class PassContext:
    """Carrier for cross-pass results and analysis caches.

    Attributes
    ----------
    results : dict[str, Any]
        Keyed storage for pass outputs.  Each pass stores its result under a
        well-known key (conventionally its module-level name or a short string
        like ``"dataflow"``).  Later passes read this dict to consume earlier
        analysis without recomputing.

    Design intent
    -------------
    Kept deliberately minimal: ``results`` is the extension point for
    cross-pass data, and typed attributes can be added alongside it as
    first-class fields if a pass needs them.
    """

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def __repr__(self) -> str:
        keys = list(self.results.keys())
        return f"PassContext(results={keys!r})"


# Pass type alias

#: A pass is a callable that receives the root Block and the shared
#: PassContext, mutates the Block and/or annotates the context in-place,
#: and returns None.
Pass = Callable[[Block, "PassContext"], None]


# walk_block — recursive pre-order op visitor


def walk_block(block: Block, visitor: Callable[[Any], None]) -> None:
    """Recursively visit every op in *block* in pre-order (depth-first).

    For each op in ``block.ops``:
      1. Call ``visitor(op)``.
      2. Recurse into each nested ``Block`` body returned by
         ``op.nested_blocks()`` (if the op is a ``TileOp`` subclass that
         carries nested bodies such as ``Loop`` or ``IfElse``).

    This traversal relies on the ``TileOp.nested_blocks()`` protocol
    (``_block_names``) rather than on isinstance checks, so it works with
    any future op that carries a ``nested_block()``-marked field.

    Parameters
    ----------
    block :
        The ``Block`` to traverse.
    visitor :
        Called once per op (including ops in nested bodies).
    """
    for op in block.ops:
        visitor(op)
        # Recurse into nested Block bodies only for TileOp subclasses.
        # The nested_blocks() method is part of the TileOp._block_names contract;
        # a non-TileOp object with a stray `nested_blocks` attr would be
        # mis-walked without this guard.
        if not isinstance(op, TileOp):
            continue
        # Every TileOp defines nested_blocks() (TileOp._block_names contract), so
        # no None-fallback is needed after the isinstance guard above.
        for nested_blk in op.nested_blocks():
            if nested_blk is not None and isinstance(nested_blk, Block):
                walk_block(nested_blk, visitor)


# run_pipeline — ordered pass runner


def run_pipeline(
    root: Block,
    passes: list[Pass],
    ctx: PassContext | None = None,
) -> PassContext:
    """Run each pass in *passes* over *root* in declaration order.

    Parameters
    ----------
    root :
        The root ``Block`` to analyse/transform.  Passed unchanged to every
        pass; passes mutate it (or annotate *ctx*) in place.
    passes :
        Ordered list of pass callables.  Each is called as
        ``pass_fn(root, ctx)``; the return value (always ``None``) is
        discarded.
    ctx :
        An existing ``PassContext`` to thread through the pipeline.  When
        ``None`` (the default) a fresh ``PassContext`` is created.

    Returns
    -------
    PassContext
        The context after all passes have run.  The same object is returned
        whether it was supplied by the caller or freshly created.
    """
    if ctx is None:
        ctx = PassContext()
    for pass_fn in passes:
        pass_fn(root, ctx)
    return ctx
