"""Explicit cast controls keep native types and reject malformed raw IR."""

import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
import tilelang.testing


@pytest.mark.parametrize("rounding", ["", "rn", "rz", "rp", "rm", "rs"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_controlled_cast_native_roundtrip(rounding, lanes):
    suffix = f"x{lanes}" if lanes > 1 else ""
    value = ir.Var("value", "float32" + suffix)
    bits = ir.Var("bits", "uint32")
    controls = {"round": rounding, "sat": bool(rounding)}
    if rounding == "rs":
        controls["rbits"] = bits
    call = T.cast(value, "float16" + suffix, **controls)
    assert call.op.name == "tl.cast"
    assert call.ty == ir.PrimType("float16" + suffix)
    assert call.args[0].same_as(value)
    assert call.attrs["round"] == rounding
    assert call.attrs["sat"] == controls["sat"]
    if rounding == "rs":
        assert call.args[1].same_as(bits)
    for expression in (call, ir.load_json(ir.save_json(call))):
        expression.validate()
        assert ir.reinfer_type(expression) == call.ty
    # A transformed call may encode the destination in its explicit result
    # type, without retaining the original optional type argument.
    rebuilt = ir.Call(call.op, call.args, attrs=call.attrs, ty=call.ty)
    rebuilt.validate()
    assert ir.reinfer_type(rebuilt) == call.ty
    function = tirx.Function([value, bits], [ir.Evaluate(call)], ir.NoneType())
    assert tirx.analysis.check_well_formed(function)


@pytest.mark.parametrize(
    "kind,expected,message",
    [
        ("missing_type", TypeError, "destination"),
        ("void_type", TypeError, "destination"),
        ("pointer_type", TypeError, "destination"),
        ("extra_type", TypeError, "destination"),
        ("mismatched_type", TypeError, "disagrees"),
        ("mismatched_lanes", TypeError, "lanes"),
        ("missing_value", ValueError, "expects a value"),
        ("extra_value", ValueError, "expects a value"),
        ("missing_attrs", TypeError, "attributes"),
        ("wrong_round_type", TypeError, None),
        ("wrong_sat_type", TypeError, None),
        ("bad_round", ValueError, "rounding mode"),
        ("missing_rbits", ValueError, "requires an rbits"),
        ("unexpected_rbits", ValueError, "only valid"),
        ("void_rbits", TypeError, "rbits"),
        ("pointer_rbits", TypeError, None),
        ("vector_rbits", TypeError, "rbits"),
    ],
)
def test_controlled_cast_rejects_malformed_native_calls(kind, expected, message):
    args = [ir.Var("value", "float32")]
    attrs = {"round": "", "sat": False}
    types = [ir.PrimType("float16")]
    kwargs = {}
    if kind == "missing_type":
        types = []
    elif kind == "void_type":
        types = [ir.PrimType("void")]
    elif kind == "pointer_type":
        types = [ir.PtrType(ir.PrimType("float16"))]
    elif kind == "extra_type":
        types.append(ir.PrimType("float32"))
    elif kind == "mismatched_type":
        kwargs["ty"] = ir.PrimType("float32")
    elif kind == "mismatched_lanes":
        types = [ir.PrimType("float16x2")]
    elif kind == "missing_value":
        args = []
    elif kind == "extra_value":
        args += args * 2
    elif kind == "missing_attrs":
        attrs = None
    elif kind == "wrong_round_type":
        attrs["round"] = 7
    elif kind == "wrong_sat_type":
        attrs["sat"] = "false"
    elif kind == "bad_round":
        attrs["round"] = "invalid"
    elif kind == "missing_rbits":
        attrs["round"] = "rs"
    elif kind == "unexpected_rbits":
        args.append(ir.Var("bits", "uint32"))
    else:
        attrs["round"] = "rs"
        bits_type = {
            "void_rbits": "void",
            "pointer_rbits": ir.PtrType(ir.PrimType("uint32")),
            "vector_rbits": "uint32x2",
        }[kind]
        args.append(ir.Var("bits", bits_type))
    with pytest.raises(expected, match=message):
        call = ir.Call("tl.cast", args, attrs=attrs, ty_args=types, **kwargs)
        call.validate()


@pytest.mark.parametrize(
    "rbits_dtype",
    ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64"],
)
def test_stochastic_cast_scalar_random_bits_survive_validation_and_reconstruction(rbits_dtype):
    value = ir.Var("value", "float32")
    bits = ir.Var("bits", rbits_dtype)
    call = T.cast(value, "float16", round="rs", sat=True, rbits=bits)
    rebuilt = ir.Call(call.op, call.args, attrs=call.attrs, ty=call.ty)
    for expression in (call, ir.load_json(ir.save_json(call)), rebuilt):
        expression.validate()
        assert expression.args[1].ty == ir.PrimType(rbits_dtype)
        assert ir.reinfer_type(expression) == ir.PrimType("float16")
    assert call.args[1].same_as(bits)
    function = tirx.Function([value, bits], [ir.Evaluate(call)], ir.NoneType())
    assert tirx.analysis.check_well_formed(function)


@tilelang.testing.requires_cuda
def test_explicit_non_saturating_integer_cast_executes_with_wraparound():
    import torch

    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "uint8")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                B[i] = T.cast(A[i], "uint8", sat=False)

    source = torch.tensor([0, 1, 127, 128, 255, 256, 511, -1] * 4, dtype=torch.int32, device="cuda")
    expected = torch.remainder(source.to(torch.int64), 256).to(torch.uint8)
    compiled = tilelang.compile(kernel, out_idx=[1], target="cuda", execution_backend="tvm_ffi")
    actual = compiled(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# This execution case builds an architecture-specific sm_100a cubin.
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_stochastic_cast_executes_on_exactly_representable_values():
    import torch

    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float16")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                B[i] = T.cast(A[i], "float16", round="rs", rbits=T.uint32(0x12345678))

    # Exact half values are invariant to stochastic random bits, providing an
    # independent numeric oracle without relying on a random-state emulator.
    source = torch.tensor(
        [-8.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 8.0] * 4,
        dtype=torch.float32,
        device="cuda",
    )
    compiled = tilelang.compile(kernel, out_idx=[1], target=Target({"kind": "cuda", "arch": "sm_100a"}), execution_backend="tvm_ffi")
    actual = compiled(source)
    torch.testing.assert_close(actual, source.to(torch.float16), rtol=0, atol=0)
    assert "__tl_cvt_f32x1_to_f16x1_rs_sat" in compiled.get_kernel_source()
