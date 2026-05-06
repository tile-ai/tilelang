from __future__ import annotations

from tvm import tir


def mma(
    A: tir.Buffer,
    B: tir.Buffer,
    C: tir.Buffer,
    *,
    extern_name: str = "hmx_mma_placeholder",
) -> None:
    """
    Emit an HMX matrix-multiply-accumulate intrinsic: C += A × B.

    The function emits a ``T.evaluate(tir.call_extern(...))`` that lowers
    to an ``hmx_mma_*`` C/C++ extern call.  When targeting a real device
    you can implement that extern using the Qualcomm Hexagon Kernel Library
    (HKL) or inline assembly.

    Parameters
    ----------
    A : tir.Buffer
        Input buffer in VTCM (int8 / uint8 / float16).
    B : tir.Buffer
        Weight buffer in VTCM (int8 / uint8 / float16).
    C : tir.Buffer
        Accumulator buffer in HMX accumulator scope (int32 / float32).
    extern_name : str
        Name of the C extern to call.  Defaults to ``hmx_mma_i8i8i32``.
        Override to ``hmx_mma_f16f16f32`` for FP16 HMX (v75+).
    """
    from tilelang import language as T

    T.evaluate(
        tir.call_extern(
            "handle",
            extern_name,
            A.data,
            B.data,
            C.data,
        )
    )


def mma_fp16(A: tir.Buffer, B: tir.Buffer, C: tir.Buffer) -> None:
    """
    FP16 × FP16 → FP32 HMX MMA (requires Hexagon v75+ with HMX-FP).
    """
    mma(A, B, C, extern_name="hmx_mma_f16f16f32")


def vtcm_dma_copy(src: tir.Buffer, dst: tir.Buffer) -> None:
    """
    Emit an async DMA copy from DDR (*src*) to VTCM (*dst*).

    In a real kernel this maps to a ``dma_move`` or similar HKL call.
    The placeholder keeps the TIR graph well-formed during development.
    """
    tir.Evaluate(
        tir.call_extern(
            "handle",
            "hexagon_dma_copy",
            src.access_ptr("r"),
            dst.access_ptr("w"),
        )
    )


class HMXBuilder:
    """
    Namespace object that groups HMX intrinsics, analogous to tilelang's
    existing MMABuilder for CUDA tensor-core operations.

    Example
    -------
    ::

        from tilelang.intrinsics.hexagon import hmx

        hmx.mma(A_vtcm, B_vtcm, C_acc)          # int8 MMA
        hmx.mma_fp16(A_vtcm, B_vtcm, C_acc)     # fp16 MMA (v75+)
        hmx.vtcm_dma_copy(A_host, A_vtcm)        # async DMA
    """

    mma = staticmethod(mma)
    mma_fp16 = staticmethod(mma_fp16)
    vtcm_dma_copy = staticmethod(vtcm_dma_copy)


hmx = HMXBuilder()


__all__ = [
    "hmx",
    "HMXBuilder",
    "register_hexagon_memory_info",
    "mma",
    "mma_fp16",
    "vtcm_dma_copy",
]
