"""IRBuilder — constructs TileIR by managing the insertion block and SSA ids.

Provides:
  - ``IRBuilder`` — owns the id counter, tracks the current insertion block,
    and exposes ``.create()`` and ``.block_scope()`` for building ops.

Design notes
-----------
* **Owner of the id counter.** ``IRBuilder`` holds a concrete ``_Counter``
  object.  Every call to ``fresh_value`` passes this counter so that id
  assignment is fully centralised — no module-global state.

* **create() verifies eagerly.** ``op.verify()`` is called before the op is
  appended to the block.  If it raises, the block is left unchanged, making
  errors visible at IR-build time rather than at a later analysis pass.

* **block_scope() is a context manager.** Nested block construction (loop
  bodies, if/else branches) swaps ``self._block`` to a fresh ``Block`` for
  the duration of the ``with`` body, then unconditionally restores it — even
  if an exception propagates out.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from tilelang.tileir.ir.types import TileType
from tilelang.tileir.ir.value import Block, Value, fresh_value
from tilelang.tileir.ir.ops import TileOp

__all__ = ["IRBuilder"]


# Internal id counter


class _Counter:
    """Monotonically increasing integer counter.

    Satisfies the ``IdCounter`` protocol required by ``fresh_value``.
    Each call to ``.next()`` returns the next integer starting from 0.
    """

    __slots__ = ("_n",)

    def __init__(self) -> None:
        self._n: int = 0

    def next(self) -> int:
        val = self._n
        self._n += 1
        return val


# IRBuilder


class IRBuilder:
    """Constructs typed TileIR inside a managed insertion context.

    Usage::

        b = IRBuilder()
        op = _MyOp(dst=some_buf, val=some_val)
        b.create(op, result_types=(ty,))
        # op.results is now a 1-tuple of fresh Values

        with b.block_scope() as inner_block:
            b.create(inner_op)
        # inner_block holds inner_op; outer block is restored

    Attributes
    ----------
    block : Block
        The current insertion block.  Ops created via ``create()`` are
        appended here.
    """

    def __init__(self) -> None:
        self._counter: _Counter = _Counter()
        self._module_block: Block = Block()
        self._block: Block = self._module_block

    # Public block accessors

    @property
    def block(self) -> Block:
        """The current insertion block."""
        return self._block

    def module_block(self) -> Block:
        """Return the root (module-level) block."""
        return self._module_block

    # create()

    def create(
        self,
        op: TileOp,
        *,
        result_types: tuple[TileType, ...] = (),
    ) -> TileOp:
        """Mint result Values, verify the op, append it, and return it.

        Parameters
        ----------
        op :
            A fully constructed ``TileOp`` instance (operands and attributes
            already set).  Its ``results`` field will be replaced with freshly
            minted ``Value`` objects.
        result_types :
            One ``TileType`` per SSA result the op produces.  May be empty for
            side-effect-only ops.

        Returns
        -------
        TileOp
            The same *op* instance, with ``op.results`` populated.

        Raises
        ------
        Any exception raised by ``op.verify()`` propagates unchanged.  The op
        is NOT appended to the current block if ``verify()`` raises.
        """
        # Allocate a fresh Value for each requested result type.
        results: tuple[Value, ...] = tuple(fresh_value(self._counter, ty) for ty in result_types)
        op.results = results

        # Verify BEFORE appending — fail at build time, not later.
        op.verify()

        self._block.append(op)
        return op

    # block_scope()

    @contextmanager
    def block_scope(self, params: list[Value] | None = None) -> Iterator[Block]:
        """Temporarily switch the insertion point to a fresh ``Block``.

        The new block is yielded.  On exit (normal or exceptional) the
        previous block is unconditionally restored.

        Parameters
        ----------
        params :
            Optional list of ``Value`` objects to use as block entry params
            (e.g. loop iteration variables).

        Yields
        ------
        Block
            The fresh inner block.  Ops created inside the ``with`` body are
            appended to it.

        Example::

            with builder.block_scope(params=[loop_var]) as body:
                builder.create(inner_op)
            # body.ops == [inner_op]
        """
        saved = self._block
        inner = Block(params=params)
        self._block = inner
        try:
            yield inner
        finally:
            self._block = saved
