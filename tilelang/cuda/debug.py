from __future__ import annotations

import warnings

from tvm import tirx

import tilelang.language as T
from tilelang.cuda.target import check_cuda_availability
from tilelang.language.eager.builder import Builder, macro

_IS_CUDA_AVAILABLE = check_cuda_availability()


def get_stack_str(msg, stacklevel=1):
    stack = Builder.current().get_fileline_stack(stacklevel)
    msg = msg + "\n"
    for fileline, lineno, macro_name in stack:
        msg += f"  at {fileline}:{lineno} in {macro_name}\n"
    return msg


@macro
def device_assert(condition: tirx.PrimExpr, msg: str = "", no_stack_info=False):
    """
    Device-side assert emulation for CUDA targets.
    """
    if _IS_CUDA_AVAILABLE:
        if no_stack_info:
            if msg == "":
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert"), condition)
            else:
                warnings.warn("Non-empty msg may slightly slow down the kernel", stacklevel=2)
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, msg)
        else:
            T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, get_stack_str(msg, stacklevel=2))
