import cutlass.cute as cute
from cutlass.cute.typing import Union, Numeric
from cutlass.cute.tensor import TensorSSA
from cutlass._mlir.dialects import arith


def divf(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric], fastmath: bool = True) -> Union[TensorSSA, Numeric]:
    return cute.math._math_op(arith.divf, fastmath, x, y)


def exp(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.exp(x, fastmath=fastmath)


def exp2(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.exp2(x, fastmath=fastmath)


def log(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.log(x, fastmath=fastmath)


def log2(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.log2(x, fastmath=fastmath)


def log10(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.log10(x, fastmath=fastmath)


def tan(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.tan(x, fastmath=fastmath)


def cos(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.cos(x, fastmath=fastmath)


def sin(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.sin(x, fastmath=fastmath)


def sqrt(x: Union[TensorSSA, Numeric], fastmath: bool = True):
    return cute.math.sqrt(x, fastmath=fastmath)
