import pytest

import tilelang.testing
from tilelang import tvm
from tvm import tirx


FAST_MATH_OPS = ["exp", "exp10", "log", "log2", "log10", "tan", "cos", "sin"]
IEEE_MATH_OPS = [
    ("ieee_add", "fadd", 2),
    ("ieee_sub", "fsub", 2),
    ("ieee_mul", "fmul", 2),
    ("ieee_fmaf", "fmaf", 3),
    ("ieee_frcp", "frcp", 1),
    ("ieee_fsqrt", "fsqrt", 1),
    ("ieee_fdiv", "fdiv", 2),
]
PACKED_MATH_OPS = [
    ("add2", 2),
    ("sub2", 2),
    ("mul2", 2),
    ("fma2", 3),
    ("max2", 2),
    ("min2", 2),
    ("max2_nan", 2),
    ("min2_nan", 2),
    ("abs2", 1),
]
BINARY_OPS = [(tirx.Add, "add2"), (tirx.Sub, "sub2"), (tirx.Mul, "mul2"), (tirx.Min, "min2"), (tirx.Max, "max2")]


def _intrinsic(dtype, name, *arguments):
    return tirx.call_intrin(dtype, tvm.ir.Op.get(name), *arguments)


def _source(expression, parameters, arch="sm_90"):
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("CUDA codegen is not enabled")
    function = tirx.PrimFunc(parameters, tirx.Evaluate(expression))
    function = function.with_attr("global_symbol", "math_dispatch")
    function = function.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return build(tvm.IRModule({"math_dispatch": function}), target).inspect_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation", FAST_MATH_OPS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_fast_math_names_and_headers(operation, dtype):
    value = tirx.Var("value", dtype)
    expression = _intrinsic(dtype, f"tl.__{operation}", value)
    source = _source(expression, [value])
    if dtype == "float32":
        expected = f"__{operation}f"
    elif dtype == "float64":
        expected = operation
    else:
        expected = f"h{operation}"
    assert f"{expected}(value)" in source
    assert ("#include <tl_templates/cuda/math.h>" in source) == (operation in {"exp", "log", "cos", "sin"})


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation,base_name,operand_count", IEEE_MATH_OPS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rounding_mode", ["rn", "rz", "ru", "rd"])
def test_ieee_math_names_arguments_and_rounding(operation, base_name, operand_count, dtype, rounding_mode):
    parameters = [tirx.Var(f"operand_{index}", dtype) for index in range(operand_count)]
    expression = _intrinsic(dtype, f"tl.{operation}", *parameters, rounding_mode)
    source = _source(expression, parameters)
    if dtype == "float32":
        expected = f"__{base_name}_{rounding_mode}"
    elif base_name == "fmaf":
        expected = f"__fma_{rounding_mode}"
    else:
        expected = f"__d{base_name[1:]}_{rounding_mode}"
    arguments = ", ".join(parameter.name for parameter in parameters)
    assert f"{expected}({arguments})" in source
    assert '"' + rounding_mode + '"' not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "operation,operand_count,expected",
    [
        ("ieee_add", 2, "__hadd_rn"),
        ("ieee_sub", 2, "__hsub_rn"),
        ("ieee_mul", 2, "__hmul_rn"),
        ("ieee_fmaf", 3, "__hfma"),
        ("ieee_frcp", 1, "hrcp"),
        ("ieee_fsqrt", 1, "hsqrt"),
        ("ieee_fdiv", 2, "__hdiv"),
    ],
)
def test_half_ieee_math_names(operation, operand_count, expected, dtype):
    parameters = [tirx.Var(f"operand_{index}", dtype) for index in range(operand_count)]
    expression = _intrinsic(dtype, f"tl.{operation}", *parameters, "rn")
    source = _source(expression, parameters)
    arguments = ", ".join(parameter.name for parameter in parameters)
    assert f"{expected}({arguments})" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation,base_name,operand_count", IEEE_MATH_OPS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("rounding_mode", ["rz", "ru", "rd"])
def test_half_ieee_math_rejects_other_rounding(operation, base_name, operand_count, dtype, rounding_mode):
    parameters = [tirx.Var(f"operand_{index}", dtype) for index in range(operand_count)]
    expression = _intrinsic(dtype, f"tl.{operation}", *parameters, rounding_mode)
    with pytest.raises(tvm.error.InternalError, match="Only rounding mode 'rn'"):
        _source(expression, parameters)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,expected", [("float32", "__frsqrt_rn"), ("float16", "hrsqrt"), ("bfloat16", "hrsqrt")])
def test_ieee_rsqrt_implicit_rounding(dtype, expected):
    value = tirx.Var("value", dtype)
    source = _source(_intrinsic(dtype, "tl.ieee_frsqrt", value), [value])
    assert f"{expected}(value)" in source


@tilelang.testing.requires_cuda
def test_ieee_rsqrt_rejects_float64():
    value = tirx.Var("value", "float64")
    with pytest.raises(tvm.error.InternalError, match="frsqrt is not supported for float64"):
        _source(_intrinsic("float64", "tl.ieee_frsqrt", value), [value])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation,operand_count", PACKED_MATH_OPS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_explicit_packed_math_dispatch(operation, operand_count, dtype):
    vector_dtype = f"{dtype}x2"
    parameters = [tirx.Var(f"operand_{index}", vector_dtype) for index in range(operand_count)]
    expression = _intrinsic(vector_dtype, f"tl.{operation}", *parameters)
    source = _source(expression, parameters)
    assert source.count(f"tl::{operation}(") == 1
    if dtype == "float32":
        assert "tl::from_uint1<" not in source
        assert "tl::to_uint1(" not in source
    else:
        native_type = "__half2" if dtype == "float16" else "__nv_bfloat162"
        assert source.count(f"tl::from_uint1<{native_type}>") == operand_count
        assert source.count("tl::to_uint1(") == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor,packed_name", BINARY_OPS)
@pytest.mark.parametrize("dtype,arch", [("float16", "sm_90"), ("bfloat16", "sm_90"), ("float32", "sm_100")])
@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_binary_packed_math_dispatch(constructor, packed_name, dtype, arch, lanes):
    lhs = tirx.Var("lhs", f"{dtype}x{lanes}")
    rhs = tirx.Var("rhs", f"{dtype}x{lanes}")
    source = _source(constructor(lhs, rhs), [lhs, rhs], arch)
    assert source.count(f"tl::{packed_name}(") == lanes // 2


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor,packed_name", BINARY_OPS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lanes", [12, 16])
def test_wide_half_binary_packed_math_dispatch(constructor, packed_name, dtype, lanes):
    lhs = tirx.Var("lhs", f"{dtype}x{lanes}")
    rhs = tirx.Var("rhs", f"{dtype}x{lanes}")
    source = _source(constructor(lhs, rhs), [lhs, rhs])
    assert source.count(f"tl::{packed_name}(") == lanes // 2
    assert f"ulonglong{lanes // 4}" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor,packed_name", BINARY_OPS)
@pytest.mark.parametrize("dtype,arch", [("float32", "sm_90"), ("int32", "sm_100"), ("float64", "sm_100")])
def test_binary_math_scalar_fallback(constructor, packed_name, dtype, arch):
    lhs = tirx.Var("lhs", f"{dtype}x2")
    rhs = tirx.Var("rhs", f"{dtype}x2")
    source = _source(constructor(lhs, rhs), [lhs, rhs], arch)
    assert f"tl::{packed_name}(" not in source
    assert "lhs.x" in source and "lhs.y" in source
    assert "rhs.x" in source and "rhs.y" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor,packed_name", BINARY_OPS)
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
def test_binary_math_odd_lanes_scalar_fallback(constructor, packed_name, dtype):
    lhs = tirx.Var("lhs", f"{dtype}x3")
    rhs = tirx.Var("rhs", f"{dtype}x3")
    source = _source(constructor(lhs, rhs), [lhs, rhs], "sm_100")
    assert f"tl::{packed_name}(" not in source
    for field in "xyz":
        assert f"lhs.{field}" in source
        assert f"rhs.{field}" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,arch", [("float16", "sm_90"), ("bfloat16", "sm_90"), ("float32", "sm_100")])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("commuted", [False, True])
def test_packed_fma_fusion(dtype, arch, explicit, commuted):
    vector_dtype = f"{dtype}x2"
    lhs, rhs, addend = [tirx.Var(name, vector_dtype) for name in ("lhs", "rhs", "addend")]
    product = _intrinsic(vector_dtype, "tl.mul2", lhs, rhs) if explicit else tirx.Mul(lhs, rhs)
    arguments = [addend, product] if commuted else [product, addend]
    expression = _intrinsic(vector_dtype, "tl.add2", *arguments) if explicit else tirx.Add(*arguments)
    source = _source(expression, [lhs, rhs, addend], arch)
    assert source.count("tl::fma2(") == 1
    assert "tl::mul2(" not in source
    assert "tl::add2(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,arch", [("float16", "sm_90"), ("bfloat16", "sm_90"), ("float32", "sm_100")])
@pytest.mark.parametrize("lanes", [4, 6, 8])
@pytest.mark.parametrize("commuted", [False, True])
def test_wide_packed_fma_fusion(dtype, arch, lanes, commuted):
    vector_dtype = f"{dtype}x{lanes}"
    lhs, rhs, addend = [tirx.Var(name, vector_dtype) for name in ("lhs", "rhs", "addend")]
    product = tirx.Mul(lhs, rhs)
    expression = tirx.Add(addend, product) if commuted else tirx.Add(product, addend)
    source = _source(expression, [lhs, rhs, addend], arch)
    assert source.count("tl::fma2(") == lanes // 2
    assert "tl::mul2(" not in source
    assert "tl::add2(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor", [tirx.EQ, tirx.NE, tirx.LT, tirx.LE, tirx.GT, tirx.GE, tirx.Div])
def test_vector_arguments_are_not_reemitted_for_fallback(constructor):
    lhs = tirx.call_extern("float16x2", "read_lhs")
    rhs = tirx.call_extern("float16x2", "read_rhs")
    source = _source(constructor(lhs, rhs), [])
    assert source.count("read_lhs(") == 2
    assert source.count("read_rhs(") == 2


@tilelang.testing.requires_cuda
def test_unhandled_call_keeps_default_codegen():
    value = tirx.Var("value", "float32")
    source = _source(tirx.call_pure_extern("float32", "user_math", value), [value])
    assert "user_math(value)" in source
    assert "#include <tl_templates/cuda/math.h>" not in source


@tilelang.testing.requires_cuda
def test_fast_math_preserves_nested_header_requirement():
    value = tirx.Var("value", "float32")
    inner = _intrinsic("float32", "tl.__exp", value)
    outer = _intrinsic("float32", "tl.__log2", inner)
    source = _source(outer, [value])
    assert "__log2f(__expf(value))" in source
    assert "#include <tl_templates/cuda/math.h>" in source


@tilelang.testing.requires_cuda
def test_fast_math_preserves_outer_header_requirement():
    value = tirx.Var("value", "float32")
    inner = _intrinsic("float32", "tl.__log2", value)
    outer = _intrinsic("float32", "tl.__exp", inner)
    source = _source(outer, [value])
    assert "__expf(__log2f(value))" in source
    assert "#include <tl_templates/cuda/math.h>" in source
