import tilelang  # noqa: F401  (sets up the TVM runtime paths)
import tvm

from tilelang.utils.language import bits_product, prim_expr_equal


def test_bits_product_large_shape_uses_int64():
    shape = [tvm.tirx.IntImm("int64", 248320), tvm.tirx.IntImm("int64", 2560)]
    bits = bits_product(shape, "uint8")
    assert str(bits.dtype) == "int64"
    assert int(bits) == 248320 * 2560 * 8


def test_bits_product_casts_int32_shape_factors():
    shape = [tvm.tirx.IntImm("int32", 248320), tvm.tirx.IntImm("int32", 2560)]
    bits = bits_product(shape, "uint8")
    assert str(bits.dtype) == "int64"
    # The product stays symbolic when the factors need a cast, but comparing the
    # same expression must still use the normalized constants.
    assert prim_expr_equal(bits, bits_product(shape, "uint8"))


def test_bits_product_counts_vector_lanes():
    bits = bits_product([tvm.tirx.IntImm("int64", 2)], "uint8x2")
    assert str(bits.dtype) == "int64"
    assert int(bits) == 2 * 8 * 2


def test_prim_expr_equal_large_constants():
    large = 248320 * 2560 * 8
    assert prim_expr_equal(large, tvm.tirx.IntImm("int64", large))
    assert not prim_expr_equal(large, tvm.tirx.IntImm("int64", large + 1))


def test_prim_expr_equal_small_constants():
    assert prim_expr_equal(1, tvm.tirx.IntImm("int32", 1))
    assert not prim_expr_equal(1, tvm.tirx.IntImm("int32", 2))


def test_int32_bit_counts_match_for_narrow_dtype_view():
    src_shape = [tvm.tirx.IntImm("int32", 248320), tvm.tirx.IntImm("int32", 2560)]
    dst_shape = [tvm.tirx.IntImm("int32", 248320), tvm.tirx.IntImm("int32", 5120)]
    src_bits = bits_product(src_shape, "uint8")
    dst_bits = bits_product(dst_shape, "uint4")
    assert str(src_bits.dtype) == "int64"
    assert str(dst_bits.dtype) == "int64"
    # The same 5,088,563,200 bits are reached through different shape factors;
    # T.view/T.reshape rely on the analyzer to prove this.
    assert tvm.arith.Analyzer().can_prove_equal(src_bits, dst_bits)
