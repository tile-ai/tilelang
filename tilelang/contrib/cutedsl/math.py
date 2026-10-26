__all__ = [
    "abs2",
    "exp",
    "exp2",
    "exp10",
    "log",
    "log1p",
    "log2",
    "log10",
    "tan",
    "cos",
    "sin",
    "sqrt",
    "rsqrt",
    "fabsf",
    "max2",
    "min2",
    "max2_nan",
    "min2_nan",
    "max_nan",
    "min_nan",
    "cuda_max",
    "cuda_min",
    "fast_max",
    "fast_min",
    "copysignf",
    "divf",
    "isfinite",
    "__habs",
    "__float2half_rz",
    "tanh",
]

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from cutlass.cute.typing import Union, Numeric
from cutlass.cute.tensor import TensorSSA

from cutlass.experimental import primitives as prims
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import BFloat16, Float16, Float32, Float64, Int16, Int32, Uint16, Uint32
from cutlass.cutlass_dsl import dsl_user_op


def _scalar_arg_type(arg):
    if isinstance(arg, Numeric):
        return type(arg)
    if hasattr(arg, "type"):
        return Numeric.from_mlir_type(arg.type)
    return None


def _scalar_type_name(scalar_type):
    return getattr(scalar_type, "__name__", str(scalar_type))


def _scalar_result_type(*args):
    result_type = None
    for arg in args:
        arg_type = _scalar_arg_type(arg)
        if arg_type is None:
            continue
        if result_type is None:
            result_type = arg_type
        elif arg_type is not result_type:
            raise TypeError(f"Mixed scalar dtypes are not supported: {_scalar_type_name(result_type)} and {_scalar_type_name(arg_type)}")
    if result_type is None:
        raise TypeError("Expected at least one scalar Numeric or MLIR value")
    return result_type


def _tl_math_op(func, fastmath: bool, *args, **kwargs):
    return func(*args, fastmath=fastmath, **kwargs)


def exp(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.exp, fastmath, x, **kwargs)


def exp2(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.exp2, fastmath, x, **kwargs)


def log(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.log, fastmath, x, **kwargs)


def log1p(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.log1p, fastmath, x, **kwargs)


def log2(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.log2, fastmath, x, **kwargs)


def log10(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.log10, fastmath, x, **kwargs)


def tan(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.tan, fastmath, x, **kwargs)


def cos(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.cos, fastmath, x, **kwargs)


def sin(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.sin, fastmath, x, **kwargs)


def sqrt(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.sqrt, fastmath, x, **kwargs)


def rsqrt(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.rsqrt, fastmath, x, **kwargs)


def exp10(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    """Compute 10^x using exp2(x * log2(10))."""
    _LOG2_10 = 3.3219280948873626  # log2(10)
    return exp2(x * _LOG2_10, fastmath=fastmath)


def fabsf(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.absf, fastmath, x)


def abs2(x: Union[TensorSSA, Numeric]) -> Union[TensorSSA, Numeric]:
    return fabsf(x)


# PTX type suffixes of the 16-bit float max/min instructions (sm_80+).
_PTX_16BIT_FLOAT = {Float16: "f16", BFloat16: "bf16"}


def _minmax_operand_type(x, y):
    """Element type of a min/max pair; Python literals take the other operand's type."""
    types = [t for t in (_scalar_arg_type(x), _scalar_arg_type(y)) if t is not None]
    if not types:
        return None
    if len(types) == 2 and types[0] is not types[1]:
        raise TypeError(f"Mixed scalar dtypes are not supported: {_scalar_type_name(types[0])} and {_scalar_type_name(types[1])}")
    return types[0]


def _minmax_asm(op, ptx_type, reg, ir_type, a, b, *, loc=None, ip=None):
    return llvm.inline_asm(
        ir_type,
        [a, b],
        f"{op}.{ptx_type} $0, $1, $2;",
        f"={reg},{reg},{reg}",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _minmax_16bit(op, x, y, dtype, *, loc=None, ip=None):
    """``{max,min}{.NaN}.{f16,bf16}``, the instruction behind ``__hmax``/``__hmax_nan``."""
    bits = [dtype(v).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip) for v in (x, y)]
    res = _minmax_asm(op, _PTX_16BIT_FLOAT[dtype], "h", Int16.mlir_type, *bits, loc=loc, ip=ip)
    return Int16(res).bitcast(dtype, loc=loc, ip=ip)


@dsl_user_op
def _minmax_16bit_x2(op, x, y, dtype, *, loc=None, ip=None):
    """``{max,min}{.NaN}.{f16x2,bf16x2}``, the instruction behind ``__hmax2``/``__hmax2_nan``."""
    packed = [llvm.bitcast(Int32.mlir_type, v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip) for v in (x, y)]
    res = _minmax_asm(op, _PTX_16BIT_FLOAT[dtype] + "x2", "r", Int32.mlir_type, *packed, loc=loc, ip=ip)
    vec_type = ir.VectorType.get([2], dtype.mlir_type)
    return TensorSSA(llvm.bitcast(vec_type, res, loc=loc, ip=ip), (2,), dtype)


@dsl_user_op
def _minmax_f64(op, x, y, *, loc=None, ip=None):
    """``{max,min}.f64``, the instruction behind ``fmax``/``fmin`` on doubles."""
    args = [Float64(v).ir_value(loc=loc, ip=ip) for v in (x, y)]
    return Float64(_minmax_asm(op, "f64", "d", Float64.mlir_type, *args, loc=loc, ip=ip))


def _cuda_minmax(is_max, x, y):
    """Scalar float min/max as the CUDA codegen lowers ``T.max``/``T.min``.

    float16/bfloat16 use ``__hmax``/``__hmin``, float32 ``fmaxf``/``fminf`` and
    float64 ``fmax``/``fmin``. All ignore a NaN operand and order +0 above -0.
    Returns ``None`` for other types.
    """
    dtype = _minmax_operand_type(x, y)
    op = "max" if is_max else "min"
    if dtype in _PTX_16BIT_FLOAT:
        return _minmax_16bit(op, x, y, dtype)
    if dtype is Float32:
        hw = cute.arch.fmax if is_max else cute.arch.fmin
        return hw(x, y)
    if dtype is Float64:
        return _minmax_f64(op, x, y)
    return None


def cuda_max(x, y):
    """``T.max`` with CUDA semantics: hardware max for floats, ``cutlass.max`` otherwise."""
    result = _cuda_minmax(True, x, y)
    return cutlass.max(x, y) if result is None else result


def cuda_min(x, y):
    """``T.min`` with CUDA semantics: hardware min for floats, ``cutlass.min`` otherwise."""
    result = _cuda_minmax(False, x, y)
    return cutlass.min(x, y) if result is None else result


def fast_max(x, y):
    """``tl::fast_max``: ``__hmax`` for 16-bit floats, ``fmaxf`` for float32, ``a < b ? b : a`` otherwise."""
    if _minmax_operand_type(x, y) in (Float16, BFloat16, Float32):
        return _cuda_minmax(True, x, y)
    return cutlass.select_(x < y, y, x)


def fast_min(x, y):
    """``tl::fast_min``: ``__hmin`` for 16-bit floats, ``fminf`` for float32, ``b < a ? b : a`` otherwise."""
    if _minmax_operand_type(x, y) in (Float16, BFloat16, Float32):
        return _cuda_minmax(False, x, y)
    return cutlass.select_(y < x, y, x)


def max_nan(x, y):
    """``tl.max_nan``: ``__hmax_nan`` (``max.NaN``) for 16-bit floats, ``tl::fast_max`` otherwise.

    A NaN operand yields the canonical NaN; +0 is ordered above -0. Vectors
    are handled lane by lane.
    """
    if isinstance(x, TensorSSA):
        return _lanewise(max_nan, x, y)
    dtype = _minmax_operand_type(x, y)
    if dtype in _PTX_16BIT_FLOAT:
        return _minmax_16bit("max.NaN", x, y, dtype)
    return fast_max(x, y)


def min_nan(x, y):
    """``tl.min_nan``: ``__hmin_nan`` (``min.NaN``) for 16-bit floats, ``tl::fast_min`` otherwise."""
    if isinstance(x, TensorSSA):
        return _lanewise(min_nan, x, y)
    dtype = _minmax_operand_type(x, y)
    if dtype in _PTX_16BIT_FLOAT:
        return _minmax_16bit("min.NaN", x, y, dtype)
    return fast_min(x, y)


def _lanewise(fn, x, y):
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in range(cute.size(x.shape)):
        out[i] = fn(x[i], y[i])
    return out.load()


def _packed_minmax(op, scalar_fn, x, y):
    if not any(isinstance(arg, TensorSSA) for arg in (x, y)):
        return scalar_fn(x, y)
    if not (isinstance(x, TensorSSA) and isinstance(y, TensorSSA)):
        raise TypeError(f"{op}2 expects two vectors or two scalars")
    if x.dtype in _PTX_16BIT_FLOAT and cute.size(x.shape) == 2:
        return _minmax_16bit_x2(op, x, y, x.dtype)
    return _lanewise(scalar_fn, x, y)


def max2(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric]) -> Union[TensorSSA, Numeric]:
    """``tl::max2``: ``__hmax2`` for half2/bfloat162, lane-wise ``fmaxf`` for float2."""
    return _packed_minmax("max", cuda_max, x, y)


def min2(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric]) -> Union[TensorSSA, Numeric]:
    """``tl::min2``: ``__hmin2`` for half2/bfloat162, lane-wise ``fminf`` for float2."""
    return _packed_minmax("min", cuda_min, x, y)


def max2_nan(x: TensorSSA, y: TensorSSA) -> TensorSSA:
    """``tl::max2_nan``: ``__hmax2_nan`` (``max.NaN.{f16x2,bf16x2}``); CUDA defines it for half2/bfloat162 only."""
    if not (isinstance(x, TensorSSA) and x.dtype in _PTX_16BIT_FLOAT and cute.size(x.shape) == 2):
        raise TypeError(f"max2_nan expects a float16x2 or bfloat16x2 vector, got {x!r}")
    return _minmax_16bit_x2("max.NaN", x, y, x.dtype)


def min2_nan(x: TensorSSA, y: TensorSSA) -> TensorSSA:
    """``tl::min2_nan``: ``__hmin2_nan`` (``min.NaN.{f16x2,bf16x2}``); CUDA defines it for half2/bfloat162 only."""
    if not (isinstance(x, TensorSSA) and x.dtype in _PTX_16BIT_FLOAT and cute.size(x.shape) == 2):
        raise TypeError(f"min2_nan expects a float16x2 or bfloat16x2 vector, got {x!r}")
    return _minmax_16bit_x2("min.NaN", x, y, x.dtype)


def copysignf(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    if any(isinstance(arg, TensorSSA) for arg in (x, y)):
        return cute_math.copysign(x, y, fastmath=fastmath)

    result_type = _scalar_result_type(x, y)
    if result_type is not Float32:
        raise TypeError(f"copysignf scalar lowering only supports Float32 inputs; got {_scalar_type_name(result_type)}")

    x_bits = Float32(x).bitcast(Uint32)
    y_bits = Float32(y).bitcast(Uint32)
    magnitude = x_bits & Uint32(0x7FFFFFFF)
    sign = y_bits & Uint32(0x80000000)
    return (magnitude | sign).bitcast(Float32)


def isfinite(x: Numeric) -> cutlass.Boolean:
    return cute_math.isfinite(x)


def divf(
    x: Union[TensorSSA, Numeric],
    y: Union[TensorSSA, Numeric],
    fastmath: bool = False,
) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(cute_math.div, fastmath, x, y)


def __habs(x: Numeric) -> Numeric:
    result_type = _scalar_result_type(x)
    if result_type not in (Float16, BFloat16):
        raise TypeError(f"__habs expects Float16 or BFloat16 input, but got {_scalar_type_name(result_type)}")

    bits = result_type(x).bitcast(Uint16)
    return (bits & Uint16(0x7FFF)).bitcast(result_type)


@dsl_user_op
def __float2half_rz(x: Union[float, Float32], *, loc=None, ip=None) -> Float16:
    return Float16(
        prims.inline_ptx(
            "cvt.rz.f16.f32 {$w0}, {$r0};",
            write_only_types=[Float16],
            read_only_args=[Float32(x)],
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def __tanhf(x: Union[float, Float32], *, fastmath, loc=None, ip=None) -> Float32:
    return Float32(
        prims.inline_ptx(
            "tanh.approx.f32 {$w0}, {$r0};",
            write_only_types=[Float32],
            read_only_args=[Float32(x)],
            loc=loc,
            ip=ip,
        )
    )


def tanh(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    return cute_math.tanh(x, fastmath=fastmath)
