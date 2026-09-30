"""Tests for pass infrastructure."""

from __future__ import annotations

import dataclasses
from typing import Any


from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block
from tilelang.tileir.ir.ops import TileOp, Effect, attribute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp16_reg() -> TileType:
    return TileType(
        dtype=dtype("float16"),
        shape=(16, 16),
        space=MemSpace.REGISTER,
        layout=None,
    )


@dataclasses.dataclass(eq=False)
class _Marker(TileOp, opcode="marker", effect=Effect.NONE):
    """A simple no-side-effect op used to tag blocks in tests."""

    tag: str = attribute(default="")


# ---------------------------------------------------------------------------
# Import the pass infrastructure under test
# ---------------------------------------------------------------------------

from tilelang.tileir.passes.base import PassContext, run_pipeline, walk_block


# ---------------------------------------------------------------------------
# PassContext tests
# ---------------------------------------------------------------------------


class TestPassContext:
    def test_results_starts_empty(self):
        ctx = PassContext()
        assert ctx.results == {}

    def test_results_can_store_arbitrary_values(self):
        ctx = PassContext()
        ctx.results["my_pass"] = [1, 2, 3]
        assert ctx.results["my_pass"] == [1, 2, 3]

    def test_multiple_passes_share_results(self):
        ctx = PassContext()
        ctx.results["pass_a"] = "alpha"
        ctx.results["pass_b"] = "beta"
        assert ctx.results["pass_a"] == "alpha"
        assert ctx.results["pass_b"] == "beta"


# ---------------------------------------------------------------------------
# run_pipeline — basic contract
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_no_op_pass_leaves_block_unchanged(self):
        """A pass that does nothing must not mutate the block."""
        root = Block()
        _fp16_reg()
        # Pre-populate block with a single op.
        op = _Marker(tag="original")
        root.append(op)

        def noop_pass(block: Block, ctx: PassContext) -> None:
            pass  # intentionally nothing

        run_pipeline(root, [noop_pass])
        assert root.ops == [op]
        assert root.ops[0].tag == "original"  # type: ignore[attr-defined]

    def test_single_pass_records_into_ctx(self):
        """A pass that records into ctx.results should be visible after run."""
        root = Block()

        def recording_pass(block: Block, ctx: PassContext) -> None:
            ctx.results["seen"] = True

        ctx = run_pipeline(root, [recording_pass])
        assert ctx.results["seen"] is True

    def test_two_passes_run_in_order(self):
        """Two passes run in the declared order; second sees first's result."""
        root = Block()
        order: list[str] = []

        def pass_a(block: Block, ctx: PassContext) -> None:
            order.append("a")
            ctx.results["order"] = list(order)

        def pass_b(block: Block, ctx: PassContext) -> None:
            order.append("b")
            ctx.results["order"] = list(order)

        ctx = run_pipeline(root, [pass_a, pass_b])
        assert ctx.results["order"] == ["a", "b"]

    def test_run_pipeline_returns_ctx(self):
        """run_pipeline must return the PassContext."""
        root = Block()
        result = run_pipeline(root, [])
        assert isinstance(result, PassContext)

    def test_run_pipeline_accepts_provided_ctx(self):
        """When a ctx is provided, that same ctx is used and returned."""
        root = Block()
        ctx = PassContext()
        ctx.results["pre"] = "value"

        def pass_a(block: Block, ctx: PassContext) -> None:
            ctx.results["pass_a"] = "done"

        returned = run_pipeline(root, [pass_a], ctx=ctx)
        assert returned is ctx
        assert returned.results["pre"] == "value"
        assert returned.results["pass_a"] == "done"

    def test_empty_pipeline_returns_ctx(self):
        """An empty pass list returns a valid PassContext immediately."""
        root = Block()
        ctx = run_pipeline(root, [])
        assert isinstance(ctx, PassContext)
        assert ctx.results == {}

    def test_pass_receives_root_block(self):
        """Each pass receives the same root Block object."""
        root = Block()
        seen_blocks: list[Block] = []

        def capturing_pass(block: Block, ctx: PassContext) -> None:
            seen_blocks.append(block)

        run_pipeline(root, [capturing_pass, capturing_pass])
        assert all(b is root for b in seen_blocks)

    def test_two_passes_both_results_visible(self):
        """After two-pass pipeline, both results are in ctx.results."""
        root = Block()

        def pass_one(block: Block, ctx: PassContext) -> None:
            ctx.results["one"] = 1

        def pass_two(block: Block, ctx: PassContext) -> None:
            ctx.results["two"] = 2

        ctx = run_pipeline(root, [pass_one, pass_two])
        assert ctx.results["one"] == 1
        assert ctx.results["two"] == 2


# ---------------------------------------------------------------------------
# walk_block — recursive op traversal
# ---------------------------------------------------------------------------


class TestWalkBlock:
    def test_walk_flat_block_visits_all_ops(self):
        """walk_block visits every op in a flat (non-nested) block."""
        root = Block()
        for tag in ("a", "b", "c"):
            root.append(_Marker(tag=tag))

        visited: list[Any] = []

        def visit(op: Any) -> None:
            visited.append(op)

        walk_block(root, visit)
        assert visited == root.ops

    def test_walk_empty_block(self):
        """walk_block on an empty block calls visitor zero times."""
        root = Block()
        visited: list[Any] = []
        walk_block(root, lambda op: visited.append(op))
        assert visited == []

    def test_walk_block_visits_nested_ops(self):
        """walk_block recurses into ops that carry nested Block bodies."""
        from tilelang.tileir.ir.ops import Loop

        root = Block()
        body = Block()
        inner_op = _Marker(tag="inner")
        body.append(inner_op)

        # Build a minimal Loop op with a body block.
        loop_op = Loop(stop=None, body=body)
        root.append(loop_op)

        visited_tags: list[str] = []

        def visit(op: Any) -> None:
            if hasattr(op, "tag"):
                visited_tags.append(op.tag)

        walk_block(root, visit)
        assert "inner" in visited_tags

    def test_walk_block_visits_outer_then_inner(self):
        """Outer op is visited before its nested ops (pre-order)."""
        from tilelang.tileir.ir.ops import Loop

        root = Block()
        body = Block()
        inner_op = _Marker(tag="inner")
        body.append(inner_op)

        outer_op = Loop(stop=None, body=body)
        root.append(outer_op)

        order: list[Any] = []
        walk_block(root, lambda op: order.append(op))

        # outer_op (Loop) must appear before inner_op (_Marker)
        assert order.index(outer_op) < order.index(inner_op)
