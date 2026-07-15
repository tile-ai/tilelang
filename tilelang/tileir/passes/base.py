"""Pass context, pipeline execution, and recursive TileIR traversal."""

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
    """Carrier for cross-pass results and analysis caches."""

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
    """Visit operations and nested blocks in depth-first pre-order."""
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
