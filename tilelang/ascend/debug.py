"""Ascend dialect device assert.

The Ascend toolkit's assert() macro works in both aicore and SIMT code, so
device asserts lower through the same tl.device_assert builtins as CUDA.
Mirroring the CUDA dialect's policy, the assert compiles to a no-op on hosts
without an NPU toolkit so host-only tracing and tests still run.
"""

from __future__ import annotations

import warnings

from tvm import tirx

import tilelang.language as T
from tilelang.cuda.debug import get_stack_str
from tilelang.language.eager.builder import macro

__all__ = ["device_assert"]


def _check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


_IS_ASCEND_AVAILABLE = _check_ascend_availability()


@macro
def device_assert(condition: tirx.PrimExpr, msg: str = "", no_stack_info=False):
    """
    Device-side assert emulation for Ascend targets.
    """
    if _IS_ASCEND_AVAILABLE:
        if no_stack_info:
            if msg == "":
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert"), condition)
            else:
                warnings.warn("Non-empty msg may slightly slow down the kernel", stacklevel=2)
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, msg)
        else:
            T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, get_stack_str(msg, stacklevel=2))
