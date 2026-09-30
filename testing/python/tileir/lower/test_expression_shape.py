"""Expression metadata must retain the participant shape used by gather lowering."""

import pytest
from tvm import tirx

from tilelang.tileir.checks import has_cuda_tile_ir_bindings
from tilelang.tileir.emission_utils import _as_tile, _broadcast_source_shape
from tilelang.tileir.errors import TileIRLoweringError, _UnsupportedTileIRNode
from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.ops import Broadcast
from tilelang.tileir.ir.types import MemSpace, TileType, dtype as lookup_dtype
from tilelang.tileir.ir.value import Value
from tilelang.tileir.lowering.mlir_emit import emit_module
from tilelang.tileir.lowering.sem_to_ir import LoweringScope, lower_expr
from tilelang.tileir.lowering.sem_to_ir.tile_level import _classify_gather_dims, _lower_tile_level_expr
from tilelang.tileir.semantic import SemanticKernel, SemanticStmt


def _scope(*bindings):
    kernel = SemanticKernel(
        name="shape",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=SemanticStmt(kind="seq", children=()),
    )
    scope = LoweringScope(kernel)
    for var, value in bindings:
        scope.bind(var, value)
    return scope


def _value(name, shape, dtype="int32"):
    return Value(-1, TileType(dtype=lookup_dtype(dtype), shape=shape, space=MemSpace.REGISTER, layout=None), name=name)


def _emit_input(builder, name, shape):
    # CUDA Tile entry arguments must be scalar. Build shaped inputs inside the
    # kernel, so these tests verify the entire module rather than an invalid ABI.
    value = _value(name, (), "float32")
    builder.block.params.append(value)
    if not shape:
        return value
    op = builder.create(
        Broadcast(src=value, src_shape=(), target_shape=shape, axis=0, reshape_shape=(1,) * len(shape)),
        result_types=(value.type.with_shape(shape),),
    )
    return op.results[0]


@pytest.mark.parametrize("tile_context", [False, True])
@pytest.mark.parametrize(
    "kind", ["add", "compare", "call_select", "condition_select", "bitwise", "shift", "not", "cast", "reinterpret", "unary"]
)
def test_expression_preserves_bound_tile_shape(kind, tile_context):
    x = tirx.Var("selected", "int32")
    xf = tirx.Cast("float32", x)
    expressions = {
        "add": x + 1,
        "compare": x >= 64,
        "call_select": tirx.if_then_else(x >= 0, x, x),
        "condition_select": tirx.Select(x >= 64, 1, 0),
        "bitwise": tirx.bitwise_and(x, 127),
        "shift": tirx.shift_right(x, 1),
        "not": tirx.bitwise_not(x),
        "cast": xf,
        "reinterpret": tirx.reinterpret("float32", x),
        "unary": tirx.exp(xf),
    }
    scope = _scope((x, _value("selected", (128,))))
    builder = IRBuilder()
    if tile_context:
        result = _lower_tile_level_expr(expressions[kind], scope, builder, {"lane"}, ordered_vars=["lane"], ordered_extents=[128])
    else:
        result = lower_expr(expressions[kind], scope, builder)
    assert result.type.shape == (128,)
    assert result.type.dtype.name == (
        "bool" if kind in ("compare",) else "float32" if kind in ("cast", "reinterpret", "unary") else "int32"
    )


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,expected", [((4,), (4, 8), (4, 8)), ((8,), (4, 8), (4, 8)), ((), (4, 8), (4, 8)), ((4, 8), (4, 8), (4, 8))]
)
@pytest.mark.skipif(not has_cuda_tile_ir_bindings(), reason="cuda_tile MLIR bindings unavailable")
def test_binary_metadata_matches_emitted_broadcast(lhs_shape, rhs_shape, expected):
    x, y = tirx.Var("x", "float32"), tirx.Var("y", "float32")
    builder = IRBuilder()
    lhs, rhs = _emit_input(builder, "x", lhs_shape), _emit_input(builder, "y", rhs_shape)
    result = lower_expr(x + y, _scope((x, lhs), (y, rhs)), builder)
    module, ctx = emit_module(
        builder.block, kernel_name="shape", entry_args=[(v.name, v.type) for v in builder.block.params], return_ctx=True
    )
    assert module.operation.verify()
    assert result.type.shape == expected
    assert tuple(_as_tile(ctx, ctx.lookup(result)).tile_type.shape) == expected
    # The row vector uses the first Parallel axis; a column vector uses the last.
    assert _broadcast_source_shape((4,), (4, 8)) == (4, 1)
    assert _broadcast_source_shape((8,), (4, 8)) == (1, 8)


@pytest.mark.skipif(not has_cuda_tile_ir_bindings(), reason="cuda_tile MLIR bindings unavailable")
def test_fma_broadcasts_scalar_product_to_shaped_addend():
    x, y, z = (tirx.Var(name, "float32") for name in ("x", "y", "z"))
    builder = IRBuilder()
    values = [_emit_input(builder, "x", ()), _emit_input(builder, "y", ()), _emit_input(builder, "z", (32,))]
    scope = _scope(*zip((x, y, z), values))
    scope.fast_math = True
    result = lower_expr(x * y + z, scope, builder)
    module, ctx = emit_module(
        builder.block, kernel_name="fma_shape", entry_args=[(v.name, v.type) for v in builder.block.params], return_ctx=True
    )
    assert module.operation.verify()
    assert builder.block.ops[-1].fn == "fma"
    assert result.type.shape == (32,)
    assert tuple(_as_tile(ctx, ctx.lookup(result)).tile_type.shape) == (32,)


def test_incompatible_binary_shapes_fail_before_gather_classification():
    x, y = tirx.Var("x", "int32"), tirx.Var("y", "int32")
    scope = _scope((x, _value("x", (4,))), (y, _value("y", (8,))))
    with pytest.raises(TileIRLoweringError, match="cannot broadcast tile shape"):
        lower_expr(x + y, scope, IRBuilder())


def test_confirmed_shaped_gather_index_does_not_fall_back():
    x = tirx.Var("selected", "int32")
    scope = _scope((x, _value("selected", (64,))))
    with pytest.raises(_UnsupportedTileIRNode, match="expected participant shape"):
        _classify_gather_dims([x + 1], scope, IRBuilder(), {"lane"}, ["lane"], [128])
