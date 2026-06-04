__all__ = [
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "tan",
    "cos",
    "sin",
    "sqrt",
    "rsqrt",
    "fabsf",
    "copysignf",
    "divf",
    "tanh",
]

from cutlass.cute.typing import Union, Numeric
from cutlass.cute.tensor import TensorSSA
from cutlass._mlir.dialects import arith, math
from cutlass.cute.math import _math_op as _cute_math_op

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op


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
    if any(isinstance(arg, TensorSSA) for arg in args):
        return _cute_math_op(func, fastmath, *args, **kwargs)

    result_type = _scalar_result_type(*args)
    if not result_type.is_float:
        raise TypeError(f"Expected scalar float inputs, but got {result_type}")

    ir_args = []
    for arg in args:
        if isinstance(arg, Numeric):
            if not type(arg).is_float:
                raise TypeError(f"Expected scalar float inputs, but got {type(arg)}")
            if type(arg) is result_type:
                ir_args.append(arg.ir_value())
            else:
                ir_args.append(result_type(arg).ir_value())
        elif hasattr(arg, "ir_value"):
            ir_args.append(result_type(arg.ir_value()).ir_value())
        else:
            ir_args.append(result_type(arg).ir_value())

    fastmath_flag = arith.FastMathFlags.fast if fastmath else arith.FastMathFlags.none
    return result_type(func(*ir_args, fastmath=fastmath_flag, **kwargs))


def exp(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.exp, fastmath, x, **kwargs)


def exp2(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.exp2, fastmath, x, **kwargs)


def log(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.log, fastmath, x, **kwargs)


def log2(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.log2, fastmath, x, **kwargs)


def log10(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.log10, fastmath, x, **kwargs)


def tan(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.tan, fastmath, x, **kwargs)


def cos(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.cos, fastmath, x, **kwargs)


def sin(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.sin, fastmath, x, **kwargs)


def sqrt(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.sqrt, fastmath, x, **kwargs)


def rsqrt(x: Union[TensorSSA, Numeric], fastmath: bool = False, **kwargs) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.rsqrt, fastmath, x, **kwargs)


def exp10(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    """Compute 10^x using exp2(x * log2(10))."""
    _LOG2_10 = 3.3219280948873626  # log2(10)
    return exp2(x * _LOG2_10, fastmath=fastmath)


def fabsf(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(math.absf, fastmath, x)


def copysignf(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    if any(isinstance(arg, TensorSSA) for arg in (x, y)):
        return _cute_math_op(math.copysign, fastmath, x, y)

    result_type = _scalar_result_type(x, y)
    if result_type is not Float32:
        raise TypeError(f"copysignf scalar lowering only supports Float32 inputs; got {_scalar_type_name(result_type)}")

    x_bits = Float32(x).bitcast(Uint32)
    y_bits = Float32(y).bitcast(Uint32)
    magnitude = x_bits & Uint32(0x7FFFFFFF)
    sign = y_bits & Uint32(0x80000000)
    return (magnitude | sign).bitcast(Float32)


def divf(
    x: Union[TensorSSA, Numeric],
    y: Union[TensorSSA, Numeric],
    fastmath: bool = False,
) -> Union[TensorSSA, Numeric]:
    return _tl_math_op(arith.divf, fastmath, x, y)


@dsl_user_op
def __tanhf(x: Union[float, Float32], *, fastmath, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value()],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def tanh(x: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    tanh_op = __tanhf if fastmath else math.tanh
    return _tl_math_op(tanh_op, False, x)
