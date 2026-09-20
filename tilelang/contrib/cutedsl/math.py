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
from cutlass.base_dsl.typing import BFloat16, Float16, Float32, Uint16, Uint32
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


def max2(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric]) -> Union[TensorSSA, Numeric]:
    if any(isinstance(arg, TensorSSA) for arg in (x, y)):
        return cute.where(x > y, x, y)
    return cutlass.max(x, y)


def min2(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric]) -> Union[TensorSSA, Numeric]:
    if any(isinstance(arg, TensorSSA) for arg in (x, y)):
        return cute.where(x < y, x, y)
    return cutlass.min(x, y)


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
