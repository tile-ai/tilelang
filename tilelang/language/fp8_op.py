"""FP8 scaled matmul intrinsic exposed on the TileLang language surface.

This module provides ``T.fp8_scaled_matmul`` — a TileLang macro that mirrors
the audiohacking/fp8-mps-metal scaled-matmul kernel signature:

    fp8_scaled_matmul(A_fp8, A_scale, B_fp8, B_scale, C_out)
        # Equivalent to: C_out += (A_fp8.float() * A_scale) @
        #                          (B_fp8.float() * B_scale)
        # with A_fp8 / B_fp8 stored as uchar (e4m3 / e5m2) and the scales
        # broadcast either per-tensor (shape == (1,)) or per-row.

Design
------

The intrinsic is a hygienic ``@T.macro`` that expands inline to the
audiohacking pattern: a scalar K-loop over a dequantize-multiply-accumulate
body with the per-tensor (or per-row) scale fused in. On Metal the inner
``T.cast(fp8_byte, fp32)`` is lowered by Agent C's storage-only patch in
``codegen_metal.cc`` to the ``__tvm_fp8_e4m3_to_half`` /
``__tvm_fp8_e5m2_to_half`` helpers; on CUDA ``T.cast`` lowers to
``__nv_fp8_e4m3_to_half`` etc. The exact same TIR is emitted on every
target — only the codegen of the scalar cast differs.

The reference kernel that this op mirrors is the
``fp8_scaled_matmul_kernel`` published in the audiohacking project:

    https://github.com/audiohacking/fp8-mps-metal
    commit d4fbd40c48aa2a243e600d06627c7dd818150636
    license: MIT

A LUT-decoded variant of the same algorithm ships in
``cppmega_mlx.nn._tilelang.fp8_msl_kernels`` (port of
``AppMana/mps-fp8-for-torch-and-comfyui-python-package`` commit
``a902571eca5362f5e2496cf33dcce52c8bac6a15``, Apache 2.0). Both upstream
projects are credited in the patch comment header.

Why a macro and not a registered TIR op
---------------------------------------

A registered ``tl.fp8_scaled_matmul`` op would buy us:

* a stable IR-level representation (legible in IR-dump traces, addressable
  by passes),
* a single point at which to switch lowering between scalar-emulation,
  cuTe FP8 GEMM (CUDA/Hopper/Blackwell), and any future Metal cooperative
  tensor instruction (Apple has no native FP8 ALU through the M5
  generation — see the Apple WWDC 2025 cooperative-tensors session).

It would cost a C++ rebuild and a parallel scheduler-pass extension. The
hygienic macro form gives us the same user-facing surface today
(``T.fp8_scaled_matmul(...)`` parses cleanly inside ``@T.prim_func``) and
the same MSL output as the C++ approach would, because all the lowering
work (FP8 storage allocation, scalar dequant cast, simdgroup-buffer
exclusion) is already done by the patches that landed earlier:

* ``docs/upstream/tilelang_metal_fp8/`` (Agent C) — storage-only FP8 in
  ``codegen_metal.cc``.
* ``docs/upstream/tilelang_metal_fp8_vector/`` (Agent F-1) — vector FP8
  cast lowering.
* ``docs/upstream/tilelang_metal_fp8_gemm/`` (Agent E) — Metal scalar
  fallback dispatcher for FP8 ``T.gemm``.

Scaled GEMM differs from plain ``T.gemm(fp8, fp8, fp32)`` only by the
extra per-element multiply by ``A_scale * B_scale``; the dispatching and
codegen path is identical. Mirroring the audiohacking scalar K-loop
verbatim therefore reduces to: take the ``GemmMetalScalar`` body that
Agent E already validated and add the scale multiplications in the
inner-most product — which is exactly what this macro emits.

Behaviour
---------

Within ``@T.prim_func`` the call expands to::

    for i, j in T.grid(M, N):
        for k in T.serial(K):
            a_val = T.cast(A_fp8[i, k], accum_dtype)   # FP8 -> fp32
            b_val = T.cast(B_fp8[k, j], accum_dtype)   # FP8 -> fp32
            sa = A_scale[0] if A_scale.shape == (1,) else A_scale[i]
            sb = B_scale[0] if B_scale.shape == (1,) else B_scale[j]
            C[i, j] = C[i, j] + a_val * b_val * sa * sb

Per-tensor vs per-row dispatch happens at macro-expansion time based on
the static shape of the scale operand; the resulting MSL has no runtime
branch.

Public attribution
------------------

* audiohacking/fp8-mps-metal (MIT) — algorithm: scalar dequant, fp32 fma,
  per-tensor / per-row scale broadcast.
* AppMana/mps-fp8-for-torch-and-comfyui-python-package (Apache 2.0) — the
  cppmega.mlx vendor ``mx.fast.metal_kernel`` port that uses a 256-entry
  LUT instead of bit-extraction; functionally equivalent.
"""

from __future__ import annotations

from typing import Optional

from tilelang import tvm as _tvm  # noqa: F401
import tilelang.language as T
from tilelang._typing import BufferLikeType
from tvm import tir
from tvm.target import Target

from .blockscaled_layout import (
    BlockScaledLayout,
    E8M0_BLOCK_K32,
    E8M0_BLOCK_SIZE,
    e8m0_to_float,
)

__all__ = [
    "fp8_scaled_matmul",
    "FP8_DTYPES",
]


# Storage-level FP8 dtype tags accepted by this intrinsic. Any other dtype
# in the A / B operands raises a TypeError at parse time. ``float8_e8m0fnu``
# is the block-scale-factor format and is intentionally excluded — it is
# carried by the sf_a / sf_b operands of the block-scaled GEMM, not by A / B.
FP8_DTYPES: tuple[str, ...] = ("float8_e4m3", "float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz")


def _is_fp8_dtype(dt) -> bool:
    """Return True if a dtype string / object names an FP8 storage variant."""
    s = str(dt or "")
    return any(s.startswith(t) for t in ("float8", "fp8"))


def _shape_extent(buffer, axis: int) -> int:
    """Return a constant integer extent for ``buffer.shape[axis]``.

    Used at macro-expansion time to dispatch per-tensor vs per-row
    behaviour. Falls back to ``-1`` if the extent is symbolic, which the
    caller treats as "assume per-row".
    """
    shape = getattr(buffer, "shape", None)
    if shape is None or len(shape) <= axis:
        return -1
    extent = shape[axis]
    if isinstance(extent, int):
        return extent
    if hasattr(extent, "value"):
        try:
            return int(extent.value)
        except (TypeError, ValueError):
            return -1
    if isinstance(extent, tir.IntImm):
        return int(extent.value)
    return -1


def _normalize_block_scale_layout(
    block_scale_layout: BlockScaledLayout | None,
    *,
    scale_format: str | None,
    scale_block_size: int | None,
) -> BlockScaledLayout | None:
    if block_scale_layout is not None:
        if not isinstance(block_scale_layout, BlockScaledLayout):
            raise TypeError(
                "T.fp8_scaled_matmul block_scale_layout must be a "
                "T.BlockScaledLayout instance"
            )
        if scale_format is not None and scale_format != block_scale_layout.scale_format:
            raise ValueError(
                "T.fp8_scaled_matmul scale_format conflicts with block_scale_layout"
            )
        if scale_block_size is not None and int(scale_block_size) != block_scale_layout.block_size:
            raise ValueError(
                "T.fp8_scaled_matmul scale_block_size conflicts with block_scale_layout"
            )
        return block_scale_layout
    if scale_format is None and scale_block_size is None:
        return None
    if scale_format != E8M0_BLOCK_K32:
        raise ValueError(
            "T.fp8_scaled_matmul e8m0 block-scale metadata requires "
            "scale_format='e8m0_block_k32'"
        )
    if scale_block_size is None or int(scale_block_size) != E8M0_BLOCK_SIZE:
        raise ValueError(
            "T.fp8_scaled_matmul e8m0_block_k32 metadata requires scale_block_size=32"
        )
    return BlockScaledLayout.e8m0_k32()


def _block_scale_value(scale, *, axis: str, col, k):
    # Path C E8M0 is explicitly contracted-K-block indexed: kb = k // 32.
    kb = k // 32
    if axis == "B" and len(getattr(scale, "shape", ())) == 2:
        return e8m0_to_float(scale[col, kb])
    return e8m0_to_float(scale[kb])


def _validate_buffers(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_out,
    *,
    transpose_B: bool,
    accum_dtype: str,
    block_scale_layout: BlockScaledLayout | None = None,
) -> None:
    """Sanity-check operand dtypes and 2D shape compatibility.

    Raises ``TypeError`` / ``ValueError`` early so misuse surfaces at the
    macro call-site rather than deep inside the parser. The macro proper
    re-derives the same shape information at expansion time; this helper
    is the public-facing validator.
    """
    A_dtype = str(getattr(A_fp8, "dtype", "")) if hasattr(A_fp8, "dtype") else ""
    B_dtype = str(getattr(B_fp8, "dtype", "")) if hasattr(B_fp8, "dtype") else ""
    C_dtype = str(getattr(C_out, "dtype", "")) if hasattr(C_out, "dtype") else ""
    sa_dtype = str(getattr(A_scale, "dtype", "")) if hasattr(A_scale, "dtype") else ""
    sb_dtype = str(getattr(B_scale, "dtype", "")) if hasattr(B_scale, "dtype") else ""

    if not _is_fp8_dtype(A_dtype):
        raise TypeError(
            f"T.fp8_scaled_matmul: A_fp8 must be FP8 (e4m3 or e5m2), got dtype={A_dtype!r}"
        )
    if not _is_fp8_dtype(B_dtype):
        raise TypeError(
            f"T.fp8_scaled_matmul: B_fp8 must be FP8 (e4m3 or e5m2), got dtype={B_dtype!r}"
        )
    scale_prefixes = ("float32", "float16", "bfloat")
    if block_scale_layout is not None:
        scale_prefixes = ("uint8",)
    if sa_dtype and not sa_dtype.startswith(scale_prefixes):
        raise TypeError(
            f"T.fp8_scaled_matmul: A_scale must be a {'uint8 E8M0 block-scale' if block_scale_layout is not None else 'floating-point scalar'} buffer, got dtype={sa_dtype!r}"
        )
    if sb_dtype and not sb_dtype.startswith(scale_prefixes):
        raise TypeError(
            f"T.fp8_scaled_matmul: B_scale must be a {'uint8 E8M0 block-scale' if block_scale_layout is not None else 'floating-point scalar'} buffer, got dtype={sb_dtype!r}"
        )
    if C_dtype and not (C_dtype.startswith("float32") or C_dtype.startswith("float16") or C_dtype.startswith("bfloat")):
        raise TypeError(
            f"T.fp8_scaled_matmul: C output must be float32 / float16 / bfloat16 (got {C_dtype!r})"
        )

    A_shape = getattr(A_fp8, "shape", None)
    B_shape = getattr(B_fp8, "shape", None)
    C_shape = getattr(C_out, "shape", None)
    if A_shape is None or B_shape is None or C_shape is None:
        return  # opaque buffer types — defer to runtime
    if len(A_shape) < 2 or len(B_shape) < 2 or len(C_shape) < 2:
        raise ValueError(
            "T.fp8_scaled_matmul: operands must be at least 2D"
        )

    M = _shape_extent(A_fp8, 0)
    K = _shape_extent(A_fp8, 1)
    if transpose_B:
        N = _shape_extent(B_fp8, 0)
        K_b = _shape_extent(B_fp8, 1)
    else:
        K_b = _shape_extent(B_fp8, 0)
        N = _shape_extent(B_fp8, 1)
    M_c = _shape_extent(C_out, 0)
    N_c = _shape_extent(C_out, 1)

    if K > 0 and K_b > 0 and K != K_b:
        raise ValueError(
            f"T.fp8_scaled_matmul: K mismatch — A is {M}x{K}, "
            f"B is {'NxK' if transpose_B else 'KxN'} = {K_b}x{N}; "
            "the contracted dimension must agree"
        )
    if M > 0 and M_c > 0 and M != M_c:
        raise ValueError(
            f"T.fp8_scaled_matmul: M mismatch — A has {M} rows but C has {M_c} rows"
        )
    if N > 0 and N_c > 0 and N != N_c:
        raise ValueError(
            f"T.fp8_scaled_matmul: N mismatch — B has {N} columns but C has {N_c} columns"
        )

    sa_size = _shape_extent(A_scale, 0)
    sb_size = _shape_extent(B_scale, 0)
    if block_scale_layout is not None:
        block_scale_layout.validate_scale_shapes(
            k_extent=K,
            a_scale_shape=tuple(int(v) for v in A_scale.shape),
            b_scale_shape=tuple(int(v) for v in B_scale.shape),
            n_extent=N,
        )
        return
    if M > 0 and sa_size > 0 and sa_size != 1 and sa_size != M:
        raise ValueError(
            f"T.fp8_scaled_matmul: A_scale must be per-tensor (size 1) or "
            f"per-row (size M={M}); got size {sa_size}"
        )
    if N > 0 and sb_size > 0 and sb_size != 1 and sb_size != N:
        raise ValueError(
            f"T.fp8_scaled_matmul: B_scale must be per-tensor (size 1) or "
            f"per-col (size N={N}); got size {sb_size}"
        )

    # accum_dtype currently must be wider than FP8; we don't accept FP16
    # accumulators because the scaled-FMA reference always accumulates in
    # FP32 (the scales themselves are typically out-of-range for FP16).
    if accum_dtype not in ("float32", "float", "float64"):
        raise ValueError(
            f"T.fp8_scaled_matmul: accum_dtype must be float32 (or wider); got {accum_dtype!r}"
        )


@T.macro
def _fp8_scaled_matmul_macro(A_fp8, A_scale, B_fp8, B_scale, C_local, block_scale_layout=None):
    """Hygienic body of ``T.fp8_scaled_matmul``: dequant + per-element scale + FMA.

    The body is parsed once at macro-decoration time and re-substituted at
    each call. Static integer extents — including ``A_scale.shape[0]`` and
    ``B_scale.shape[0]`` — drive the per-tensor-vs-per-row branch at
    expansion time, so the resulting MSL contains no runtime predicate.

    The outer ``(i, j)`` loop is ``T.Parallel`` so the layout-inference
    engine distributes the M*N output cells across ``threads`` cleanly:
    each thread owns a small slice of ``C_local`` and runs its private
    K-loop. Without ``T.Parallel`` the layout pass falls back to a
    replicated layout (every thread does the full work) which gives
    correct results but wastes work; ``T.Parallel`` matches the
    audiohacking kernel's threadgroup-tiling pattern exactly. Mirrors the
    ``fp8_scaled_matmul_kernel`` reference body line-for-line up to the
    macro variable substitutions.
    """
    M_dim, K_dim = A_fp8.shape
    K_dim_b, N_dim = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]

    # The accumulation matches the audiohacking ``fp8_scaled_matmul_kernel``
    # algorithm: per-element FP8 dequant, fp32 FMA, scale broadcast through
    # the multiply. ``T.cast(fp8 -> fp32)`` lowers to ``__tvm_fp8_*_to_half``
    # on Metal (Agent C's storage-only patch) or ``__nv_fp8_*_to_half`` on
    # CUDA (TVM's existing FP8 type lowering).
    for i, j in T.Parallel(M_dim, N_dim):
        for k in T.serial(K_dim):
            a_val = T.cast(A_fp8[i, k], "float32")
            b_val = T.cast(B_fp8[k, j], "float32")
            if block_scale_layout is not None:
                sa = _block_scale_value(A_scale, axis="A", col=j, k=k)
                sb = _block_scale_value(B_scale, axis="B", col=j, k=k)
            else:
                sa = A_scale[0] if sa_size == 1 else A_scale[i]
                sb = B_scale[0] if sb_size == 1 else B_scale[j]
            C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb


@T.macro
def _fp8_scaled_matmul_macro_trans_b(A_fp8, A_scale, B_fp8, B_scale, C_local, block_scale_layout=None):
    """``transpose_B=True`` variant: B is (N, K) row-major, indexed B[j, k]."""
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]

    for i, j in T.Parallel(M_dim, N_dim):
        for k in T.serial(K_dim):
            a_val = T.cast(A_fp8[i, k], "float32")
            b_val = T.cast(B_fp8[j, k], "float32")
            if block_scale_layout is not None:
                sa = _block_scale_value(A_scale, axis="A", col=j, k=k)
                sb = _block_scale_value(B_scale, axis="B", col=j, k=k)
            else:
                sa = A_scale[0] if sa_size == 1 else A_scale[i]
                sb = B_scale[0] if sb_size == 1 else B_scale[j]
            C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb


def fp8_scaled_matmul(
    A_fp8: BufferLikeType,
    A_scale: BufferLikeType,
    B_fp8: BufferLikeType,
    B_scale: BufferLikeType,
    C_out: BufferLikeType,
    *,
    transpose_B: bool = False,
    accum_dtype: str = "float32",
    target: Optional[Target] = None,  # accepted for API compat, currently unused
    scale_format: str | None = None,
    scale_block_size: int | None = None,
    block_scale_layout: BlockScaledLayout | None = None,
):
    """Scaled FP8 matmul intrinsic — accumulate scaled FP8 product into ``C``.

    Computes::

        C_out += (A_fp8 * A_scale) @ (B_fp8 * B_scale)

    where ``A_fp8`` and ``B_fp8`` are FP8 (``e4m3`` or ``e5m2``) storage
    buffers and the scales are floating-point scalars (per-tensor when
    shape is ``(1,)``, per-row / per-col otherwise). Mirrors the
    ``fp8_scaled_matmul_kernel`` algorithm from
    ``audiohacking/fp8-mps-metal`` (MIT).

    The accumulator ``C_out`` is read-modify-write — callers typically
    ``T.clear(C_local)`` once and then call this op inside the K-tile
    loop, exactly like ``T.gemm`` semantics.

    Behaviour by target
    ~~~~~~~~~~~~~~~~~~~

    The macro emits the same TIR on every target. The output MSL / PTX
    differs only in the codegen of the FP8-to-fp32 cast:

    * **Metal** — ``T.cast(fp8 byte, fp32)`` lowers via
      ``__tvm_fp8_e4m3_to_half`` / ``__tvm_fp8_e5m2_to_half`` from Agent
      C's storage-only patch, then a half-to-float promotion. The
      resulting MSL is functionally identical to the audiohacking
      ``fp8_scaled_matmul_kernel`` (one branch + a few shifts per byte
      per dequantization + fp32 fma).
    * **CUDA / ROCm** — ``T.cast`` uses TVM's native FP8 path
      (``__nv_fp8_e4m3_to_half`` etc.). For Hopper / Blackwell, callers
      who want the tensor-core FP8 FMA path should use
      ``T.tcgen05_gemm_blockscaled(...)`` directly (PRs #202 / #1600);
      those gemms ingest the ``e8m0fnu`` block-scale operand explicitly
      and don't fit this op's per-tensor / per-row scale signature.
    * **CPU / fallback** — same scalar TIR; ``T.cast(fp8, fp32)`` lowers
      via TVM's CPU FP8 helpers.

    Args:
        A_fp8: Input A in FP8 storage. Shape ``(M, K)`` row-major.
        A_scale: Per-tensor (shape ``(1,)``) or per-row (shape ``(M,)``)
            fp32 scale for A.
        B_fp8: Input B in FP8 storage. Shape ``(K, N)`` row-major when
            ``transpose_B`` is False, otherwise ``(N, K)`` row-major.
        B_scale: Per-tensor (shape ``(1,)``) or per-col (shape ``(N,)``)
            fp32 scale for B.
        C_out: Accumulator output. Shape ``(M, N)``, fp32.
        transpose_B: Mirror ``T.gemm`` semantics. Defaults to ``False``.
        accum_dtype: Accumulator dtype for the inner GEMM (and the cast
            target for FP8 dequant). Defaults to ``"float32"``.
        target: Currently accepted for API compatibility; the macro emits
            the same TIR on every target.

    Returns:
        The handle returned by the underlying ``@T.macro`` invocation,
        which the TileLang parser inlines as a ``tir.SeqStmt`` at the
        call site.

    Raises:
        TypeError: If ``A_fp8`` / ``B_fp8`` are not FP8 dtypes, or any
            scale / accumulator dtype is not a real-valued type.
        ValueError: If shapes don't agree (``K`` mismatch, ``M`` /
            ``N`` mismatch, or scale shapes that are neither 1 nor
            matching).
    """
    layout = _normalize_block_scale_layout(
        block_scale_layout,
        scale_format=scale_format,
        scale_block_size=scale_block_size,
    )
    _validate_buffers(
        A_fp8, A_scale, B_fp8, B_scale, C_out,
        transpose_B=transpose_B, accum_dtype=accum_dtype, block_scale_layout=layout,
    )

    if transpose_B:
        return _fp8_scaled_matmul_macro_trans_b(A_fp8, A_scale, B_fp8, B_scale, C_out, layout)
    return _fp8_scaled_matmul_macro(A_fp8, A_scale, B_fp8, B_scale, C_out, layout)
