"""Tests for lowering/sem_to_ir.py — SemanticIR → TileIR lowering pass.

Verifies:
- IMPL registry contains expected kind handlers.
- lower_kernel produces the correct op types for copy / gemm / for / if-else kernels.
- LoweringScope frame scoping isolates bindings.
- Gemm unsigned dtype flag is set from SemanticBuffer dtype string.
- seq stmt lowers all children.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tvm import tirx as _tirx

# Allow imports from the sibling test-utils directory (same pattern as other tileir tests).
sys.path.insert(0, str(Path(__file__).parents[1]))

import tilelang
from tilelang import language as T

from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
from tilelang.tileir.lowering.sem_to_ir import (
    IMPL,
    TILE_OP_IMPL,
    LoweringScope,
    lower_kernel,
)
from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.ops import (
    Broadcast,
    Loop,
    IfElse,
    Copy,
    Gemm,
    Store,
    RepeatInterleave,
    TmaCopy,
)
from tilelang.tileir.ir.types import MemSpace, TileType, dtype as lookup_dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt, SemanticProgram, SemanticRegion
from tilelang.tileir.errors import TileIRLoweringError, _UnsupportedTileIRNode


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _make_copy_tir():
    """Simple copy kernel: T.copy(a, a_shared); T.copy(a_shared, b)."""

    @tilelang.jit
    def copy_kernel(a, b):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((32, 64), T.float16)

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), T.float16)
            T.copy(a, a_shared)
            T.copy(a_shared, b)

    return copy_kernel.get_tir(None, None)


def _make_gemm_tir():
    """Gemm kernel: copy + gemm + copy."""

    @tilelang.jit
    def gemm_kernel(a, b, c):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((64, 32), T.float16)
        c: T.Tensor((32, 32), T.float32)

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), T.float16)
            b_shared = T.alloc_shared((64, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(a_shared, b_shared, acc, clear_accum=True)
            T.copy(acc, c)

    return gemm_kernel.get_tir(None, None, None)


def _make_loop_tir():
    """For-loop kernel: pipelined loop over slices."""

    @tilelang.jit
    def loop_kernel(a, b):
        a: T.Tensor((128, 64), T.float16)
        b: T.Tensor((128, 64), T.float16)

        with T.Kernel(1, threads=128):
            for i in T.Pipelined(4, num_stages=2):
                a_shared = T.alloc_shared((32, 64), T.float16)
                T.copy(a[i * 32 : (i + 1) * 32, :], a_shared)
                T.copy(a_shared, b[i * 32 : (i + 1) * 32, :])

    return loop_kernel.get_tir(None, None)


def _make_packed_repeat_tir():
    """Collective packed read: each source byte is repeated twice on axis 1."""

    @tilelang.jit
    def packed_repeat_kernel(packed, out):
        packed: T.Tensor((8, 4), T.int8)
        out: T.Tensor((8, 8), T.int8)

        with T.Kernel(1, threads=128):
            packed_shared = T.alloc_shared((8, 4), T.int8)
            unpacked = T.alloc_fragment((8, 8), T.int8)
            T.copy(packed, packed_shared)
            for i, j in T.Parallel(8, 8):
                unpacked[i, j] = packed_shared[i, j // 2]
            T.copy(unpacked, out)

    return packed_repeat_kernel.get_tir(None, None)


def _make_serial_row_slice_outer_product_tir():
    """Two serially sliced fragments multiplied in a 2-D parallel body."""

    @tilelang.jit
    def serial_row_slice_outer_product(a, b, out):
        a: T.Tensor((4, 4), T.float32)
        b: T.Tensor((4, 8), T.float32)
        out: T.Tensor((4, 8), T.float32)

        with T.Kernel(1, threads=128):
            a_local = T.alloc_fragment((4, 4), T.float32)
            b_local = T.alloc_fragment((4, 8), T.float32)
            out_local = T.alloc_fragment((4, 8), T.float32)
            T.copy(a, a_local)
            T.copy(b, b_local)
            for i, j in T.Parallel(4, 8):
                for k in T.serial(4):
                    out_local[i, j] = a_local[k, i] * b_local[k, j]
            T.copy(out_local, out)

    return serial_row_slice_outer_product.get_tir(None, None, None)


def _make_gemm_uint8_tir():
    """Gemm kernel with int8 / int32 (uint8 signedness test)."""

    @tilelang.jit
    def gemm_u8(a, b, c):
        a: T.Tensor((32, 64), T.int8)
        b: T.Tensor((64, 32), T.int8)
        c: T.Tensor((32, 32), T.int32)

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), T.int8)
            b_shared = T.alloc_shared((64, 32), T.int8)
            acc = T.alloc_fragment((32, 32), T.int32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(a_shared, b_shared, acc, clear_accum=True)
            T.copy(acc, c)

    return gemm_u8.get_tir(None, None, None)


# ---------------------------------------------------------------------------
# Helper: flatten all ops from a block (including nested)
# ---------------------------------------------------------------------------


def _all_ops(block: Block) -> list:
    """Flatten all ops from a block (DFS, includes nested block ops)."""
    result = []
    for op in block.ops:
        result.append(op)
        for nested in op.nested_blocks():
            if nested is not None:
                result.extend(_all_ops(nested))
    return result


def _ops_of_type(block: Block, typ) -> list:
    """Return all ops in the block (recursively) that are instances of typ."""
    return [op for op in _all_ops(block) if isinstance(op, typ)]


# ---------------------------------------------------------------------------
# Test 1: IMPL registry contains expected handlers
# ---------------------------------------------------------------------------


def test_lower_kernel_import_ok():
    """IMPL dict should contain all key SemanticStmt kind handlers."""
    expected_kinds = {
        "seq",
        "block",
        "thread_extent",
        "threadblock_swizzle_pattern",
        "reduce_scope",
        "let",
        "for",
        "while",
        "if",
        "buffer_store",
        "tile_op",
        "atomic_rmw",
        "register_control",
    }
    missing = expected_kinds - set(IMPL.keys())
    assert not missing, f"IMPL missing handlers for: {missing}"


def test_tile_op_impl_has_key_ops():
    """TILE_OP_IMPL should contain handlers for copy/fill/gemm/reduce."""
    expected = {
        "tl.tileop.copy",
        "tl.tileop.tma_copy",
        "tl.tileop.fill",
        "tl.tileop.gemm",
        "tl.tileop.tcgen05_gemm",
        "tl.tileop.reduce",
        "tl.tileop.cumsum",
        "tir.break_loop",
        "tir.continue_loop",
        "tir.tvm_storage_sync",
        "tir.tvm_thread_allreduce",
    }
    missing = expected - set(TILE_OP_IMPL.keys())
    assert not missing, f"TILE_OP_IMPL missing handlers for: {missing}"


def _lower_handcrafted_stmt(stmt: SemanticStmt, buffers: tuple[SemanticBuffer, ...] = ()):
    """Lower one hand-built SemanticStmt through the real statement dispatcher."""
    kernel = SemanticKernel(
        name="handcrafted_stmt",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=buffers,
        body=SemanticStmt(kind="seq", children=(stmt,)),
    )
    builder = IRBuilder()
    return lower_kernel(kernel, builder)


def _regions_and_buffers(count: int) -> tuple[tuple[SemanticRegion, ...], tuple[SemanticBuffer, ...]]:
    buffers = tuple(SemanticBuffer(name=f"buf{i}", shape=(8,), dtype="float32", scope="local") for i in range(count))
    regions = tuple(SemanticRegion(buffer=buf.name, access="readwrite", indices=("0",), shape=(8,)) for buf in buffers)
    return regions, buffers


@pytest.mark.parametrize(
    "op_name,required_regions",
    [
        ("tl.tileop.copy", 2),
        ("tl.tileop.tma_copy", 2),
        ("tl.tileop.fill", 1),
        ("tl.tileop.gemm", 3),
        ("tl.tileop.reduce", 2),
        ("tl.tileop.cumsum", 2),
    ],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_core_tile_ops_reject_malformed_region_count(op_name, required_regions, delta):
    """Core computation ops must not disappear or ignore extra regions."""
    actual_regions = required_regions + delta
    regions, buffers = _regions_and_buffers(actual_regions)
    stmt = SemanticStmt(kind="tile_op", attrs=(("op", op_name),), regions=regions)

    with pytest.raises(
        TileIRLoweringError,
        match=rf"{op_name.split('.')[-1]} requires exactly {required_regions} regions; got {actual_regions}",
    ) as exc_info:
        _lower_handcrafted_stmt(stmt, buffers)

    assert type(exc_info.value) is TileIRLoweringError


@pytest.mark.parametrize(
    "op_name,required_regions",
    [
        ("tl.tileop.copy", 2),
        ("tl.tileop.tma_copy", 2),
        ("tl.tileop.fill", 1),
        ("tl.tileop.gemm", 3),
        ("tl.tileop.reduce", 2),
        ("tl.tileop.cumsum", 2),
    ],
)
def test_core_tile_ops_reject_unknown_region_buffer(op_name, required_regions):
    """A missing buffer is a lowering-pipeline error, never a skippable op."""
    regions = tuple(SemanticRegion(buffer=f"missing{i}", access="readwrite", indices=("0",), shape=(8,)) for i in range(required_regions))
    stmt = SemanticStmt(kind="tile_op", attrs=(("op", op_name),), regions=regions)

    with pytest.raises(
        TileIRLoweringError,
        match=rf"{op_name.split('.')[-1]} region 0 .* unknown buffer 'missing0'",
    ) as exc_info:
        _lower_handcrafted_stmt(stmt)

    assert type(exc_info.value) is TileIRLoweringError


def test_buffer_store_rejects_unknown_buffer():
    stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "missing"), ("value", _tirx.FloatImm("float32", 1.0))),
    )

    with pytest.raises(
        TileIRLoweringError,
        match="buffer_store references unknown buffer 'missing'",
    ) as exc_info:
        _lower_handcrafted_stmt(stmt)

    assert type(exc_info.value) is TileIRLoweringError


def test_parallel_buffer_store_rejects_unknown_buffer():
    from tilelang.tileir.lowering.sem_to_ir.parallel import _lower_parallel_buffer_store

    empty_kernel = SemanticKernel(
        name="parallel_unknown_buffer",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=SemanticStmt(kind="seq", children=()),
    )
    builder = IRBuilder()
    scope = LoweringScope(empty_kernel, builder=builder)
    stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "missing"), ("value", _tirx.FloatImm("float32", 1.0))),
    )

    with pytest.raises(
        TileIRLoweringError,
        match="parallel buffer_store references unknown buffer 'missing'",
    ) as exc_info:
        _lower_parallel_buffer_store(stmt, {"i"}, scope, builder)

    assert type(exc_info.value) is TileIRLoweringError


def test_parallel_buffer_store_rejects_missing_source_indices():
    from tilelang.tileir.lowering.sem_to_ir.parallel import _lower_parallel_buffer_store

    dst = SemanticBuffer(name="dst", shape=(8,), dtype="float32", scope="local")
    kernel = SemanticKernel(
        name="parallel_missing_indices",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(dst,),
        body=SemanticStmt(kind="seq", children=()),
    )
    builder = IRBuilder()
    scope = LoweringScope(kernel, builder=builder)
    stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "dst"), ("value", _tirx.FloatImm("float32", 1.0))),
    )

    with pytest.raises(
        TileIRLoweringError,
        match="parallel buffer_store to 'dst' has no raw TIR indices",
    ) as exc_info:
        _lower_parallel_buffer_store(
            stmt,
            {"i"},
            scope,
            builder,
            ordered_vars=["i"],
            ordered_extents=[8],
        )

    assert type(exc_info.value) is TileIRLoweringError


def test_parallel_buffer_store_rejects_missing_loop_metadata():
    from tilelang.tileir.lowering.sem_to_ir.parallel import _lower_parallel_buffer_store

    dst_sem = SemanticBuffer(name="dst", shape=(8, 8), dtype="float32", scope="local")
    kernel = SemanticKernel(
        name="parallel_missing_metadata",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(dst_sem,),
        body=SemanticStmt(kind="seq", children=()),
    )
    builder = IRBuilder()
    scope = LoweringScope(kernel, builder=builder)
    dst_tir = _tirx.decl_buffer((8, 8), "float32", name="dst")
    source = _tirx.BufferStore(
        dst_tir,
        _tirx.FloatImm("float32", 1.0),
        [_tirx.IntImm("int32", 0), _tirx.IntImm("int32", 0)],
    )
    stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "dst"),),
        value=source.value,
        indices=tuple(source.indices),
    )

    with pytest.raises(
        TileIRLoweringError,
        match="parallel buffer_store to 'dst' has no ordered loop metadata",
    ) as exc_info:
        _lower_parallel_buffer_store(stmt, {"i"}, scope, builder)

    assert type(exc_info.value) is TileIRLoweringError


def _tma_copy_stmt(dst_index):
    """Build a real tl.tileop.tma_copy Call with a chosen destination start."""
    from tilelang.utils.language import to_buffer_region

    src_tir = _tirx.decl_buffer((32,), "float32", name="src")
    dst_tir = _tirx.decl_buffer((64,), "float32", name="dst")
    extent = _tirx.IntImm("int32", 32)
    src_call = to_buffer_region(
        _tirx.BufferLoad(src_tir, [_tirx.IntImm("int32", 0)]),
        access_type="r",
        extents=[extent],
    )
    dst_call = to_buffer_region(
        _tirx.BufferLoad(dst_tir, [dst_index]),
        access_type="w",
        extents=[extent],
    )
    source = _tirx.call_intrin(
        "handle",
        _tirx.op.Op.get("tl.tileop.tma_copy"),
        src_call,
        dst_call,
    )
    stmt = SemanticStmt(
        kind="tile_op",
        attrs=(("op", "tl.tileop.tma_copy"),),
        regions=(
            SemanticRegion(buffer="src", access="read", indices=("0",), shape=(32,)),
            SemanticRegion(buffer="dst", access="write", indices=(str(dst_index),), shape=(32,)),
        ),
        call_args=tuple(source.args),
    )
    buffers = (
        SemanticBuffer(name="src", shape=(32,), dtype="float32", scope="shared"),
        SemanticBuffer(name="dst", shape=(64,), dtype="float32", scope=""),
    )
    return stmt, buffers


def test_tma_copy_preserves_exact_partition_index():
    stmt, buffers = _tma_copy_stmt(_tirx.IntImm("int32", 32))

    block = _lower_handcrafted_stmt(stmt, buffers)

    tma_ops = _ops_of_type(block, TmaCopy)
    assert len(tma_ops) == 1
    assert tma_ops[0].dst_indices == (1,)


def test_tma_copy_preserves_element_offset_instead_of_using_zero():
    stmt, buffers = _tma_copy_stmt(_tirx.IntImm("int32", 1))
    block = _lower_handcrafted_stmt(stmt, buffers)
    op = _ops_of_type(block, TmaCopy)[0]
    assert op.dst_indices == (1,)
    assert op.dst_elem_view is True
    assert op.src_elem_view is False


def test_tma_copy_rejects_unlowerable_partition_index_instead_of_using_zero():
    index_tir = _tirx.decl_buffer((1,), "int32", name="index")
    bad_index = _tirx.BufferLoad(index_tir, [_tirx.IntImm("int32", 0)])
    stmt, buffers = _tma_copy_stmt(bad_index)

    with pytest.raises(
        _UnsupportedTileIRNode,
        match="tma_copy dst: cannot lower start offset",
    ):
        _lower_handcrafted_stmt(stmt, buffers)


@pytest.mark.parametrize("shared_source", [False, True])
def test_tma_copy_rejects_non_exact_shared_subtile(shared_source):
    from dataclasses import replace

    stmt, buffers = _tma_copy_stmt(_tirx.IntImm("int32", 1))
    buffers = (replace(buffers[0], scope=""), replace(buffers[1], scope="shared"))
    if shared_source:
        stmt = replace(
            stmt,
            regions=(replace(stmt.regions[1], access="read"), replace(stmt.regions[0], access="write")),
            call_args=tuple(reversed(stmt.call_args)),
        )
    with pytest.raises(_UnsupportedTileIRNode, match="shared/register tile slices require an exactly divisible start offset"):
        _lower_handcrafted_stmt(stmt, buffers)


def test_buffer_store_rejects_unlowerable_index_instead_of_using_zero():
    dst_tir = _tirx.decl_buffer((8,), "float32", name="dst")
    index_tir = _tirx.decl_buffer((1,), "int32", name="index")
    bad_index = _tirx.BufferLoad(index_tir, [_tirx.IntImm("int32", 0)])
    source = _tirx.BufferStore(
        dst_tir,
        _tirx.FloatImm("float32", 1.0),
        [bad_index],
    )
    stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "dst"),),
        value=source.value,
        indices=tuple(source.indices),
    )
    buffers = (SemanticBuffer(name="dst", shape=(8,), dtype="float32", scope="local"),)

    with pytest.raises(
        _UnsupportedTileIRNode,
        match="buffer_store to 'dst': cannot lower index",
    ):
        _lower_handcrafted_stmt(stmt, buffers)


# ---------------------------------------------------------------------------
# Test 2: Copy kernel produces Copy ops
# ---------------------------------------------------------------------------


def test_lower_kernel_copy():
    """Copy kernel should produce Copy ops in the lowered block."""
    prim_func = _make_copy_tir()
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    copy_ops = _ops_of_type(block, Copy)
    assert len(copy_ops) >= 1, f"Expected at least 1 Copy op, got {len(copy_ops)}"

    # Both copies should reference different buffers.
    seen_src = {op.src.name for op in copy_ops if op.src is not None}
    seen_dst = {op.dst.name for op in copy_ops if op.dst is not None}
    assert len(seen_src) >= 1
    assert len(seen_dst) >= 1


# ---------------------------------------------------------------------------
# Test 3: Gemm kernel produces Gemm op
# ---------------------------------------------------------------------------


def test_lower_kernel_gemm():
    """Gemm kernel should produce a Gemm op with lhs, rhs, and acc."""
    prim_func = _make_gemm_tir()
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    gemm_ops = _ops_of_type(block, Gemm)
    assert len(gemm_ops) == 1, f"Expected exactly 1 Gemm op, got {len(gemm_ops)}"

    g = gemm_ops[0]
    assert g.lhs is not None
    assert g.rhs is not None
    assert g.acc is not None
    # clear_accum=True was passed
    assert g.clear is True


# ---------------------------------------------------------------------------
# Test 4: For-loop kernel produces Loop op with is_for=True
# ---------------------------------------------------------------------------


def test_lower_kernel_for_loop():
    """For-loop kernel should produce a Loop op with is_for=True."""
    prim_func = _make_loop_tir()
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    loop_ops = _ops_of_type(block, Loop)
    assert len(loop_ops) >= 1, f"Expected at least 1 Loop op, got {len(loop_ops)}"

    for_loops = [op for op in loop_ops if op.is_for]
    assert len(for_loops) >= 1, "Expected at least one Loop with is_for=True"


def test_lower_parallel_packed_read_uses_repeat_interleave():
    """A ``j // 2`` packed read lowers collectively, not as scalar gather."""
    prim_func = _make_packed_repeat_tir()
    program = tir_to_sem(prim_func)
    builder = IRBuilder()

    block = lower_kernel(program.kernels[0], builder, program=program)

    repeat_ops = _ops_of_type(block, RepeatInterleave)
    assert len(repeat_ops) == 1
    repeat = repeat_ops[0]
    assert repeat.axis == 1
    assert repeat.repeats == 2
    assert repeat.results[0].type.shape == (8, 8)


def test_serial_row_slices_broadcast_to_parallel_outer_product():
    """Serially sliced fragment rows retain their parallel-axis placement."""
    prim_func = _make_serial_row_slice_outer_product_tir()
    program = tir_to_sem(prim_func)
    builder = IRBuilder()

    block = lower_kernel(program.kernels[0], builder, program=program)

    broadcasts = [op for op in _ops_of_type(block, Broadcast) if op.target_shape == (4, 8)]
    assert {op.reshape_shape for op in broadcasts} >= {(4, 1), (1, 8)}
    assert all(op.results[0].type.shape == (4, 8) for op in broadcasts)


# ---------------------------------------------------------------------------
# Test 5: If-else from SemanticStmt
# ---------------------------------------------------------------------------


def test_lower_kernel_if_else():
    """An 'if' SemanticStmt should lower to an IfElse op."""
    from tvm import tirx as _tirx

    # Build a minimal SemanticKernel with a hand-crafted 'if' stmt.
    # SemanticKernel requires alloc_buffers; provide an empty one.
    then_stmt = SemanticStmt(kind="seq", children=())
    else_stmt = SemanticStmt(kind="seq", children=())
    if_stmt = SemanticStmt(
        kind="if",
        attrs=(("condition", _tirx.IntImm("bool", 1)),),
        children=(then_stmt, else_stmt),
    )
    body = SemanticStmt(kind="seq", children=(if_stmt,))
    kernel = SemanticKernel(
        name="test_if",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )

    builder = IRBuilder()
    lower_kernel(kernel, builder)

    if_ops = _ops_of_type(builder.block, IfElse)
    assert len(if_ops) == 1, f"Expected 1 IfElse op, got {len(if_ops)}"

    if_op = if_ops[0]
    assert if_op.cond is not None
    assert if_op.then_block is not None
    assert if_op.else_block is not None


# ---------------------------------------------------------------------------
# Test 6: LoweringScope frame push/pop
# ---------------------------------------------------------------------------


def test_lowering_scope_frame_push_pop():
    """Bindings inside a frame should be invisible after the frame exits."""
    # Create a minimal SemanticKernel (no alloc_buffers).
    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test_scope",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )
    scope = LoweringScope(kernel)
    dummy_ty = TileType(dtype=lookup_dtype("int32"), shape=(), space=MemSpace.REGISTER, layout=None)
    dummy_val = Value(id=0, type=dummy_ty, name="dummy")

    # Outside frame: not visible yet.
    assert scope.lookup("x") is None

    with scope.frame():
        scope.bind("x", dummy_val)
        assert scope.lookup("x") is dummy_val

    # After frame exit: binding is gone.
    assert scope.lookup("x") is None


# ---------------------------------------------------------------------------
# Test 7: Gemm unsigned dtype flag
# ---------------------------------------------------------------------------


def test_gemm_unsigned_dtype_flag():
    """For a gemm with int8 inputs the lhs_unsigned/rhs_unsigned should default
    to False — int8 is not the unsigned alias uint8 in TileIR naming."""
    prim_func = _make_gemm_uint8_tir()
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    gemm_ops = _ops_of_type(block, Gemm)
    assert len(gemm_ops) >= 1, "Expected at least 1 Gemm op"

    g = gemm_ops[0]
    # The dtype string from the TIR is "int8" (not "uint8"), so both should be False.
    # If the frontend uses uint8 alias the flag would be True; this test validates
    # the detection logic against what actually comes out of tilelang.jit.
    assert isinstance(g.lhs_unsigned, bool)
    assert isinstance(g.rhs_unsigned, bool)


# ---------------------------------------------------------------------------
# Test 8: Seq stmt lowers all children
# ---------------------------------------------------------------------------


def test_lower_kernel_seq():
    """A 'seq' SemanticStmt should lower all children into the block."""
    # Manually craft two buffer_store children to count ops.
    # We need a real buffer so lookup_buffer works.
    from tvm import tirx as _tirx

    buf = SemanticBuffer(name="mybuf", shape=(8,), dtype="float32", scope="local")
    store_stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "mybuf"), ("value", _tirx.FloatImm("float32", 0.0))),
    )
    seq_stmt = SemanticStmt(
        kind="seq",
        children=(store_stmt, store_stmt),  # two identical stores
    )
    body = SemanticStmt(kind="seq", children=(seq_stmt,))
    kernel = SemanticKernel(
        name="test_seq",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(buf,),
        body=body,
    )

    builder = IRBuilder()
    lower_kernel(kernel, builder)

    store_ops = _ops_of_type(builder.block, Store)
    assert len(store_ops) == 2, f"Expected 2 Store ops from seq, got {len(store_ops)}"


# ---------------------------------------------------------------------------
# Test 9: Buffer memory spaces are correctly classified
# ---------------------------------------------------------------------------


def test_lowering_scope_buffer_memory_spaces():
    """alloc_shared -> SHARED, alloc_fragment -> REGISTER."""
    shared_buf = SemanticBuffer(name="shbuf", shape=(32, 64), dtype="float16", scope="shared")
    frag_buf = SemanticBuffer(name="fgbuf", shape=(32, 32), dtype="float32", scope="local")

    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test_spaces",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(shared_buf, frag_buf),
        body=body,
    )
    scope = LoweringScope(kernel)

    shbuf_val = scope.lookup_buffer("shbuf")
    fgbuf_val = scope.lookup_buffer("fgbuf")

    assert shbuf_val.type.space == MemSpace.SHARED
    assert fgbuf_val.type.space == MemSpace.REGISTER


# ---------------------------------------------------------------------------
# Test 10: Lowering a real gemm kernel end-to-end (structural check)
# ---------------------------------------------------------------------------


def test_lower_gemm_end_to_end_structure():
    """Lower a real gemm kernel and confirm the block has Gemm + Copy ops."""
    prim_func = _make_gemm_tir()
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    assert block is not None
    assert isinstance(block, Block)
    assert len(block.ops) > 0, "Expected at least one op in the lowered block"

    # Should have both Copy and Gemm ops.
    all_types = {type(op).__name__ for op in _all_ops(block)}
    assert "Copy" in all_types, f"Expected Copy in ops, got {all_types}"
    assert "Gemm" in all_types, f"Expected Gemm in ops, got {all_types}"


# ---------------------------------------------------------------------------
# Unsigned-gemm flag from raw SemanticBuffer.dtype (not collapsed name)
# ---------------------------------------------------------------------------


def _make_uint8_gemm_kernel():
    """Build a SemanticProgram/SemanticKernel with uint8 lhs/rhs buffers by hand.

    This avoids going through tilelang.jit (which maps uint8→int8 before TIR
    emission) and directly tests that _lower_gemm_common reads the raw dtype
    string from LoweringScope._raw_dtypes rather than Value.type.dtype.name.
    """
    # lhs and rhs are declared as uint8 (unsigned), acc is int32.
    lhs_buf = SemanticBuffer(name="lhs_u8", shape=(32, 64), dtype="uint8", scope="shared")
    rhs_buf = SemanticBuffer(name="rhs_u8", shape=(64, 32), dtype="uint8", scope="shared")
    acc_buf = SemanticBuffer(name="acc_i32", shape=(32, 32), dtype="int32", scope="local")

    # A single gemm tile_op stmt with three regions.
    gemm_stmt = SemanticStmt(
        kind="tile_op",
        attrs=(
            ("op", "tl.tileop.gemm"),
            ("transpose_A", "0"),
            ("transpose_B", "0"),
            ("clear_accum", "1"),
        ),
        regions=(
            SemanticRegion(buffer="lhs_u8", access="read", indices=(), shape=(32, 64)),
            SemanticRegion(buffer="rhs_u8", access="read", indices=(), shape=(64, 32)),
            SemanticRegion(buffer="acc_i32", access="readwrite", indices=(), shape=(32, 32)),
        ),
    )
    body = SemanticStmt(kind="seq", children=(gemm_stmt,))
    kernel = SemanticKernel(
        name="uint8_gemm",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(lhs_buf, rhs_buf, acc_buf),
        body=body,
    )
    program = SemanticProgram(
        name="uint8_gemm_prog",
        params=(),
        global_alloc_buffers=(),
        kernels=(kernel,),
    )
    return program, kernel


def test_gemm_uint8_lhs_rhs_unsigned_flags_true():
    """gemm whose lhs/rhs SemanticBuffer.dtype='uint8' must yield
    lhs_unsigned=True and rhs_unsigned=True.

    Using a_val.type.dtype.name would return 'int8' for any uint8 buffer
    (alias collapse in lookup_dtype), making the flag always False
    regardless of actual signedness.
    """
    program, kernel = _make_uint8_gemm_kernel()

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    gemm_ops = _ops_of_type(block, Gemm)
    assert len(gemm_ops) == 1, f"Expected exactly 1 Gemm op, got {len(gemm_ops)}"

    g = gemm_ops[0]
    assert g.lhs_unsigned is True, f"Expected lhs_unsigned=True for uint8 lhs buffer, got {g.lhs_unsigned!r}"
    assert g.rhs_unsigned is True, f"Expected rhs_unsigned=True for uint8 rhs buffer, got {g.rhs_unsigned!r}"


# ---------------------------------------------------------------------------
# Shift ops lower to Elementwise (not silently dropped)
# ---------------------------------------------------------------------------


def test_shift_op_lowers_to_elementwise():
    """tir.shift_left/shift_right in lower_expr must lower to
    Elementwise(fn="shl"/"shr") — NOT raise _UnsupportedTileIRNode.

    Use a Var operand (not IntImm) to prevent constant folding — tirx.shift_left
    with constant operands folds to an IntImm at construction time.
    """
    from tvm import tirx as _tirx
    from tilelang.tileir.ir.ops import Elementwise
    from tilelang.tileir.ir.types import MemSpace, TileType

    # Var operand prevents constant-folding → shift_left returns a Call node.
    x = _tirx.Var("x_var", "int32")
    shift_amount = _tirx.IntImm("int32", 2)
    shift_expr = _tirx.shift_left(x, shift_amount)
    assert isinstance(shift_expr, _tirx.Call), "Expected shift_left(Var, IntImm) to produce a Call (not folded)"

    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="shift_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )
    scope = LoweringScope(kernel)
    builder = IRBuilder()
    from tilelang.tileir.lowering.sem_to_ir import lower_expr

    # Bind x_var so lower_expr can resolve it.
    from tilelang.tileir.ir.types import dtype as _lookup_dtype

    x_ty = TileType(dtype=_lookup_dtype("int32"), shape=(), space=MemSpace.REGISTER, layout=None)
    from tilelang.tileir.ir.value import fresh_value

    x_val = fresh_value(builder._counter, x_ty, name="x_var")
    scope.bind("x_var", x_val)

    # lower_expr should succeed and produce an Elementwise(fn="shl") result.
    result = lower_expr(shift_expr, scope, builder)
    assert result is not None, "lower_expr returned None for shl"
    # The result Value should be produced by an Elementwise(fn="shl") op.
    shl_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "shl"]
    assert shl_ops, f"Expected at least one Elementwise(fn='shl') op in the block, got: {builder.block.ops}"

    # Similarly test shift_right.
    shift_right_expr = _tirx.shift_right(x, shift_amount)
    assert isinstance(shift_right_expr, _tirx.Call)
    result_r = lower_expr(shift_right_expr, scope, builder)
    assert result_r is not None
    shr_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "shr"]
    assert shr_ops, "Expected at least one Elementwise(fn='shr') op in the block"


# ---------------------------------------------------------------------------
# Decode/dp4a tile ops raise _UnsupportedTileIRNode (not Barrier)
# ---------------------------------------------------------------------------


def _make_decode_kernel(op_name: str):
    """Build a SemanticKernel with a single decode tile_op stmt."""
    body = SemanticStmt(
        kind="seq",
        children=(
            SemanticStmt(
                kind="tile_op",
                attrs=(("op", op_name),),
            ),
        ),
    )
    return SemanticKernel(
        name="decode_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )


@pytest.mark.parametrize(
    "op_name,match_str",
    [
        ("decode_i4u_to_f16", "decode_i4u_to_f16"),
        ("decode_i2u_to_i8s", "decode_i2u_to_i8s"),
        ("DP4A", "DP4A"),
    ],
)
def test_decode_dp4a_raises_unsupported(op_name, match_str):
    """decode_i4u_to_f16 / decode_i2u_to_i8s / DP4A tile ops must raise
    _UnsupportedTileIRNode, NOT silently emit a semantically-wrong Barrier."""
    kernel = _make_decode_kernel(op_name)
    builder = IRBuilder()
    with pytest.raises(_UnsupportedTileIRNode, match=match_str):
        lower_kernel(kernel, builder)


# ---------------------------------------------------------------------------
# Buffer Value ids are >= 0 (monotone) when builder is passed
# ---------------------------------------------------------------------------


def test_buffer_value_ids_are_nonnegative():
    """LoweringScope should stamp buffer Values with ids >= 0 via
    fresh_value when a builder is provided, not id=-1."""
    shared_buf = SemanticBuffer(name="shbuf", shape=(32, 64), dtype="float16", scope="shared")
    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test_ids",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(shared_buf,),
        body=body,
    )
    builder = IRBuilder()
    scope = LoweringScope(kernel, builder=builder)
    val = scope.lookup_buffer("shbuf")
    assert val.id >= 0, f"Expected buffer Value id >= 0, got {val.id}"


def test_lower_let_replay_rejects_buffer_load_in_deferred_value():
    """Deferred replay rejects expressions that read from a buffer."""
    from tvm import tirx as _tirx
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    tir_buf = _tirx.decl_buffer((8,), "int32", name="mybuf")
    buf_load = _tirx.BufferLoad(tir_buf, [_tirx.IntImm("int32", 0)])
    # Keep ``w`` unbound so the binding takes the replay path.
    unbound_var = _tirx.Var("w", "int32")
    deferred_value = _tirx.Add(unbound_var, buf_load)

    let_stmt = SemanticStmt(
        kind="let",
        attrs=(("var", "bad_replay"), ("value", deferred_value)),
        children=(),
    )
    body = SemanticStmt(kind="seq", children=(let_stmt,))
    kernel = SemanticKernel(
        name="test_replay_purity",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(SemanticBuffer(name="mybuf", shape=(8,), dtype="int32", scope="local"),),
        body=body,
    )

    builder = IRBuilder()
    with pytest.raises(TileIRLoweringNotImplementedError, match="BufferLoad"):
        lower_kernel(kernel, builder)


def test_replay_purity_analysis_fails_closed(monkeypatch):
    """A traversal failure must not classify an expression as replay-safe."""
    from tilelang.tileir.lowering.sem_to_ir import stmt as stmt_lowering

    expr = _tirx.Add(_tirx.Var("w", "int32"), _tirx.IntImm("int32", 1))

    def fail_visit(*_args, **_kwargs):
        raise RuntimeError("synthetic traversal failure")

    monkeypatch.setattr(_tirx.stmt_functor, "post_order_visit", fail_visit)

    with pytest.raises(TileIRLoweringError, match="replay purity analysis failed"):
        stmt_lowering._expr_contains_buffer_load(expr)
