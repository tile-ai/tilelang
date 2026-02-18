"""
Simple wrappers that delegate to cutlass.cute.arch implementations.
We use the existing implementations from cutlass rather than reinventing the wheel.
"""

from cutlass.cute.typing import Pointer, Int, Int32, Boolean  # noqa: F401
from cutlass.cutlass_dsl import CuTeDSL, dsl_user_op  # noqa: F401
from cutlass._mlir.dialects import nvvm, llvm

from cutlass.cute.arch import mbarrier_init, mbarrier_expect_tx, mbarrier_arrive  # noqa: F401
from cutlass.cute.arch import mbarrier_arrive_and_expect_tx as arrive_and_expect_tx  # noqa: F401
from cutlass.cute.arch import cp_async_mbarrier_arrive_noinc as mbarrier_cp_async_arrive_noinc  # noqa: F401

import cutlass.cute.arch as arch


@dsl_user_op
def mbarrier_wait(mbar_ptr: Pointer, phase: Int, timeout_ns: Int = 10000000, *, loc=None, ip=None) -> None:
    """Waits on a mbarrier with a specified phase (blocking loop).

    Uses inline PTX to loop until the try_wait succeeds.
    The CUDA backend does: while (!mbar.try_wait(parity)) {}
    """
    llvm.inline_asm(
        None,
        [mbar_ptr.llvm_ptr, Int32(phase).ir_value(loc=loc, ip=ip), Int32(timeout_ns).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, $2;\n@!p bra LAB_WAIT;\n}",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_cp_async_arrive(mbar_ptr: Pointer, *, loc=None, ip=None) -> None:
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    nvvm.cp_async_mbarrier_arrive_shared(
        mbar_llvm_ptr,
        noinc=False,
        loc=loc,
        ip=ip,
    )


def fence_proxy_async():
    arch.fence_proxy(arch.ProxyKind.async_shared, space=arch.SharedSpace.shared_cta)


def fence_barrier_init():
    arch.mbarrier_init_fence()
