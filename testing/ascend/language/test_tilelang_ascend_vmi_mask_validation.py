"""VMI mask operands are validated at trace time (#3858).

The unary, vector-scalar, and hand-written VMI factories used to accept any
object as ``mask``; code generation then failed far from the call site.  These
are frontend-only checks: every case either raises while the ``T.prim_func``
body is traced or builds the ``tl.vmi.*`` call, so no Ascend device is needed.
"""

import pytest

from tilelang.ascend import language as T
from tvm import tirx
from testing.ascend._ir import calls

pytestmark = pytest.mark.pto

LANES = 128

UNARY_OPS = ("vabs", "vneg", "vrelu", "vexp", "vln", "vsqrt", "vnot")
VEC_SCALAR_OPS = ("vadds", "vmuls", "vmaxs", "vmins", "vshls", "vshrs")


class _NotAMask:
    def __repr__(self):
        return "<not-a-mask>"


def _build(call, *, dtype="float32", lanes=LANES):
    """Trace one VMI call and store its vector result."""

    @T.prim_func
    def main(A: T.Tensor((lanes,), dtype), C: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            source = T.alloc_shared((lanes,), dtype)
            dest = T.alloc_shared((lanes,), dtype)
            T.copy(A, source)
            with T.SimdVF():
                value = T.vmi.vload(source[0], size=lanes)
                result = call(value)
                mask = T.vmi.create_mask(lanes, size=lanes)
                T.vmi.vstore(result, dest[0], mask)
            T.copy(dest, C)

    return main


def _mask_error(name, got, *, lanes=LANES):
    return rf"T\.vmi\.{name}\(\.\.\.\): mask must be a predicate with {lanes} lanes, got {got}"


@pytest.mark.parametrize("name", UNARY_OPS)
def test_unary_rejects_non_predicate_mask(name):
    with pytest.raises(TypeError, match=_mask_error(name, f"float32x{LANES}")):
        _build(lambda value: getattr(T.vmi, name)(value, value))


@pytest.mark.parametrize("name", UNARY_OPS)
def test_unary_rejects_mask_with_wrong_lane_count(name):
    with pytest.raises(TypeError, match=_mask_error(name, "boolx64")):
        _build(lambda value: getattr(T.vmi, name)(value, T.vmi.create_mask(64, size=64)))


@pytest.mark.parametrize("name", UNARY_OPS)
def test_unary_accepts_matching_predicate(name):
    program = _build(lambda value: getattr(T.vmi, name)(value, T.vmi.create_mask(LANES, size=LANES)))
    (call,) = calls(program, f"tl.vmi.{name}")
    assert str(call.dtype) == f"float32x{LANES}"


@pytest.mark.parametrize("name", UNARY_OPS)
def test_unary_keeps_mask_optional(name):
    program = _build(lambda value: getattr(T.vmi, name)(value))
    (call,) = calls(program, f"tl.vmi.{name}")
    assert len(call.args) == 1


def test_unary_rejects_unconvertible_mask_operand():
    with pytest.raises(TypeError, match=_mask_error("vabs", "<not-a-mask>")):
        _build(lambda value: T.vmi.vabs(value, _NotAMask()))


@pytest.mark.parametrize("name", VEC_SCALAR_OPS)
def test_vec_scalar_rejects_non_predicate_mask(name):
    with pytest.raises(TypeError, match=_mask_error(name, f"float32x{LANES}")):
        _build(lambda value: getattr(T.vmi, name)(value, tirx.const(1, "int32"), value))


@pytest.mark.parametrize("name", VEC_SCALAR_OPS)
def test_vec_scalar_rejects_mask_with_wrong_lane_count(name):
    with pytest.raises(TypeError, match=_mask_error(name, "boolx64")):
        _build(lambda value: getattr(T.vmi, name)(value, tirx.const(1, "int32"), T.vmi.create_mask(64, size=64)))


def test_vec_scalar_accepts_matching_predicate():
    program = _build(lambda value: T.vmi.vadds(value, tirx.const(1.0, "float32"), T.vmi.create_mask(LANES, size=LANES)))
    (call,) = calls(program, "tl.vmi.vadds")
    assert str(call.dtype) == f"float32x{LANES}"


@pytest.mark.parametrize(
    ("call", "dtype"),
    [
        pytest.param(lambda value: T.vmi.vexpdif(value, value, value), "float32", id="vexpdif"),
        pytest.param(lambda value: T.vmi.vaxpy(value, value, tirx.const(0.5, "float32"), value), "float32", id="vaxpy"),
        pytest.param(lambda value: T.vmi.vlrelu(value, tirx.const(0.5, "float32"), value), "float32", id="vlrelu"),
        pytest.param(lambda value: T.vmi.vprelu(value, value, value), "float32", id="vprelu"),
        pytest.param(lambda value: T.vmi.vmula(value, value, value, value), "int32", id="vmula"),
    ],
)
def test_handwritten_ops_reject_non_predicate_mask(call, dtype):
    with pytest.raises(TypeError, match=f"mask must be a predicate with {LANES} lanes, got {dtype}x{LANES}"):
        _build(call, dtype=dtype)


def test_handwritten_vexpdif_accepts_matching_predicate():
    program = _build(lambda value: T.vmi.vexpdif(value, value, T.vmi.create_mask(LANES, size=LANES)))
    (call,) = calls(program, "tl.vmi.vexpdif")
    assert str(call.dtype) == f"float32x{LANES}"
