import cutlass.cute as cute
from cutlass.cute.typing import Union, Numeric
from cutlass.cute.tensor import TensorSSA
from cutlass._mlir.dialects import arith, math
from cutlass.cute.math import exp, exp2, log, log2, log10, tan, cos, sin, sqrt  # noqa: F401

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Float32
from cutlass.cutlass_dsl import T, dsl_user_op


def divf(x: Union[TensorSSA, Numeric], y: Union[TensorSSA, Numeric], fastmath: bool = False) -> Union[TensorSSA, Numeric]:
    return cute.math._math_op(arith.divf, fastmath, x, y)


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
    return cute.math._math_op(tanh_op, False, x)
