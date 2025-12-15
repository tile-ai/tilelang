import cutlass.cute as cute
from cutlass.cute.typing import Union, Numeric
from cutlass.cute.tensor import TensorSSA
from cutlass._mlir.dialects import arith
from cutlass.cute.math import exp, exp2, log, log2, log10, tan, cos, sin, sqrt  # noqa: F401


def divf(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    return cute.math._math_op(arith.divf, fastmath, x, y)
