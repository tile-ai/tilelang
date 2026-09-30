"""Tests for tilelang.tileir.ir.types."""

from tilelang.tileir.ir.types import TileType, MemSpace, dtype, DTYPES


def test_dtype_singletons_are_unique_and_indexed():
    assert dtype("float16") is dtype("float16")
    assert dtype("float16").bitwidth == 16
    assert "bfloat16" in DTYPES


def test_tiletype_scalar_vs_tile():
    s = TileType(dtype("float32"), (), MemSpace.REGISTER, None)
    t = TileType(dtype("float32"), (128, 64), MemSpace.SHARED, None)
    assert s.is_scalar() and not t.is_scalar()
    assert t.with_shape((64, 64)).shape == (64, 64)


def test_tiletype_eq_hash():
    a = TileType(dtype("int32"), (8,), MemSpace.GLOBAL, None)
    b = TileType(dtype("int32"), (8,), MemSpace.GLOBAL, None)
    assert a == b and hash(a) == hash(b)
