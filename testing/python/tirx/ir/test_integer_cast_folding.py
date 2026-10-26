import pytest

from tilelang._compat import ir, sym, tirx


@pytest.mark.parametrize(
    "source,value,destination,expected",
    [
        ("int32", 128, "int8", -128),
        ("int32", 255, "int8", -1),
        ("int32", -129, "int8", 127),
        ("int32", -1, "uint8", 255),
        ("int32", 256, "uint8", 0),
        ("int32", 0, "bool", 0),
        ("int32", 2, "bool", 1),
        ("int32", -2, "bool", 1),
        ("uint64", 2**64 - 1, "int64", -1),
        ("int64", -1, "uint64", 2**64 - 1),
        ("int64", 2500000000, "int32", -1794967296),
        ("int128", 2**100 + 127, "int8", 127),
        ("int8", -1, "uint128", 2**128 - 1),
        ("int32", -5, "int64", -5),
    ],
)
def test_integer_cast_constant_matches_fixed_width_conversion(source, value, destination, expected):
    expression = ir.prim.Cast(destination, ir.IntImm(source, value))
    folded = sym.Analyzer().simplify(expression)
    assert isinstance(folded, ir.IntImm)
    assert folded.ty == ir.PrimType(destination)
    assert int(folded) == expected


def test_broadcast_integer_cast_wraps_each_lane():
    expression = ir.prim.Cast("int8x4", ir.prim.Broadcast(ir.const(255, "int32"), 4))
    folded = sym.Analyzer().simplify(expression)
    assert folded.ty == ir.PrimType("int8x4")
    assert isinstance(folded, ir.prim.Broadcast)
    assert int(folded.value) == -1


def test_out_of_range_literals_are_still_rejected():
    with pytest.raises(ValueError, match="range"):
        ir.IntImm("int8", 128)
    with pytest.raises(ValueError, match="negative"):
        ir.IntImm("uint8", -1)


@pytest.mark.parametrize("operation", ["and", "or", "xor"])
def test_python_bitwise_constants_preserve_unsigned_operand_width(operation):
    variable = ir.Var("variable", "uint32")
    expression = {
        "and": lambda: tirx.bitwise_and(variable, 0xFFFFFFFF),
        "or": lambda: tirx.bitwise_or(variable, 0x80000000),
        "xor": lambda: tirx.bitwise_xor(variable, 0xABCDEF01),
    }[operation]()
    assert expression.ty == ir.PrimType("uint32")
    assert int(expression.b) == {"and": 0xFFFFFFFF, "or": 0x80000000, "xor": 0xABCDEF01}[operation]
