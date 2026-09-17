"""CUDA dialect of the GEMM operators.

Shadows the common ``gemm`` / ``gemm_blockscaled`` / ``gemm_sp`` with
CUDA-specific parameters and hosts the explicit, instruction-pinned CUDA
variants (``wgmma_gemm``, ``tcgen05_gemm``, ``tcgen05_gemm_blockscaled``,
``mma_gemm_blockscaled``) that have no target-neutral meaning.
"""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.experimental.gemm_sp_op import _gemm_sp_impl
from tilelang.language.gemm_op import BarrierType, GemmWarpPolicy, _gemm_blockscaled_impl, _gemm_impl
from tilelang.language.utils import _normalize_annotations
from tilelang.layout import Layout
from tilelang.utils.language import retrieve_shape, to_buffer_region

__all__ = [
    "gemm",
    "gemm_blockscaled",
    "gemm_sp",
    "make_blockscaled_gemm_layout",
    "mma_gemm_blockscaled",
    "tcgen05_gemm",
    "tcgen05_gemm_blockscaled",
    "wgmma_gemm",
]


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    mbar: BarrierType | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """TileLang GEMM operator for CUDA.

    Same semantics as the common :func:`tilelang.language.gemm_op.gemm`: the
    default synchronous GEMM. On Hopper, if the compiler selects WGMMA
    lowering, TileLang inserts the corresponding wait implicitly. On Blackwell
    TCGEN5MMA, TileLang inserts the corresponding
    ``mbarrier_wait_parity(...)`` implicitly after issue.

    For manual asynchronous scheduling, use ``T.wgmma_gemm(...)`` with
    ``T.wait_wgmma(...)`` on Hopper, or ``T.tcgen05_gemm(...)`` with
    ``T.mbarrier_wait_parity(...)`` on Blackwell.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell.
            Required when this GEMM lowers to TCGEN5MMA. Defaults to None.
        annotations (Optional[dict]): Additional annotations.

    Returns:
        tirx.Call: A handle to the GEMM operation.
    """
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        annotations=annotations,
    )


def gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    *,
    k_start: int | tirx.PrimExpr,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    mbar: BarrierType | None = None,
    use_2cta: bool = False,
    sf_layout: str | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Block-scaled GEMM for CUDA: ``C (+)= (A * SFA) @ (B * SFB)``.

    Extends :func:`tilelang.language.gemm_op.gemm_blockscaled` with CUDA
    completion and scale-layout parameters. Like ``T.gemm``, this is the
    synchronous interface. The compiler selects the block-scaled instruction
    from the target and operand scopes:

    - Blackwell SM100, ``C`` in tensor memory: TCGEN05
      ``kind::mxf8f6f4.block_scale``. A/B are FP8/FP6/FP4 in shared memory,
      and SFA/SFB are E8M0 scale factors already in tensor memory. ``mbar``
      is required: the MMA posts completion to it and TileLang inserts the
      matching ``mbarrier_wait_parity`` implicitly after issue, so the
      barrier must flip once per call. ``use_2cta=True`` requests
      ``cta_group::2`` and requires ``cluster_dims`` of ``(2,1,1)`` or
      ``(1,2,1)``.
    - SM120, ``C`` in a fragment: synchronous
      ``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`` with E2M1 operands,
      UE4M3 scale factors and FP32 accumulation. A/B and packed scale words
      reside in shared memory. ``sf_layout`` describes the scale words;
      this path ignores ``mbar``.

    Unsupported combinations fail compilation instead of dropping scales.
    Use ``T.tcgen05_gemm_blockscaled`` for explicit asynchronous TCGEN05
    scheduling without the implicit wait, or ``T.mma_gemm_blockscaled`` to
    pin the warp-level path.

    Args:
        A: Left operand tile in shared memory.
        B: Right operand tile in shared memory.
        C: Accumulator in tensor memory on SM100 or a fragment on SM120.
        SFA: Scale factors for A.
        SFB: Scale factors for B.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy for the warp-level path.
        clear_accum: Whether to zero the accumulator before accumulating.
        k_start: Logical K-axis start offset of this tile.
        sf_a_granularity_k: K elements covered by one A scale factor.
        sf_b_granularity_k: K elements covered by one B scale factor.
        mbar: Completion barrier, required by TCGEN05; waited on implicitly.
        use_2cta: Request the 2CTA TCGEN05 variant.
        sf_layout: SM120 scale layout, ``"rowmajor"`` or
            ``"blockscaled_chunk_kmajor"``.
        annotations: Additional annotations.
    """
    ann = dict(annotations or {})
    if use_2cta:
        ann["use_2cta"] = 1
    if sf_layout is not None:
        ann["sf_layout"] = sf_layout
    return _gemm_blockscaled_impl(
        "tl.tileop.gemm_blockscaled",
        A,
        B,
        C,
        SFA,
        SFB,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        k_start=k_start,
        sf_a_granularity_k=sf_a_granularity_k,
        sf_b_granularity_k=sf_b_granularity_k,
        annotations=ann,
    )


def gemm_sp(
    A_sparse: BufferLikeType | tirx.Var,
    E: BufferLikeType | tirx.Var,
    B: BufferLikeType | tirx.Var,
    C: BufferLikeType | tirx.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    wg_wait: int = 0,
    annotations: dict | None = None,
) -> tirx.Call:
    """Sparse GEMM (2:4 structured sparsity) for CUDA.

    Same semantics as the common :func:`tilelang.language.experimental.gemm_sp_op.gemm_sp`.
    ``wg_wait`` is the Hopper warpgroup wait count consumed when the WGMMA SP
    lowering is selected (``-1`` defers the wait to an explicit
    ``T.wait_wgmma``); it rides in the tile-op annotations.

    Args:
        A_sparse: Compressed sparse matrix containing only non-zero elements.
        E: Metadata tensor encoding the sparsity pattern of A.
        B: Dense input matrix.
        C: Output accumulator matrix.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_E: Whether to transpose E. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy. Defaults to GemmWarpPolicy.Square.
        clear_accum: Whether to zero the accumulator before computation. Defaults to False.
        wg_wait: Warp group wait count. Defaults to 0.
        annotations: Additional annotations; values in it take precedence.

    Returns:
        tirx.Call: A handle to the sparse GEMM operation.
    """
    ann = dict(annotations) if annotations is not None else {}
    if wg_wait != 0:
        ann.setdefault("wg_wait", wg_wait)
    return _gemm_sp_impl(
        "tl.tileop.gemm_sp",
        A_sparse,
        E,
        B,
        C,
        transpose_A,
        transpose_E,
        transpose_B,
        policy,
        clear_accum,
        annotations=ann or None,
    )


def wgmma_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Explicit Hopper WGMMA GEMM without an implicit wait.

    This is the explicit asynchronous Hopper WGMMA counterpart to the default
    synchronous `T.gemm(...)` interface, with two stricter guarantees:
    - it always requests the WGMMA lowering path
    - it never auto-emits an inlined `warpgroup_wait`

    If the current target or operand pattern cannot use Hopper WGMMA,
    compilation fails instead of silently falling back to MMA.
    """

    ann = _normalize_annotations(annotations)
    # Explicit async WGMMA: never auto-emit the warpgroup wait.
    ann.setdefault("wg_wait", -1)
    return _gemm_impl(
        "tl.tileop.wgmma_gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        annotations=ann,
    )


def tcgen05_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    *,
    mbar: BarrierType | None,
    use_2cta: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Explicit Blackwell TCGEN05 GEMM without an implicit wait.

    This is the explicit asynchronous Blackwell TCGEN5MMA counterpart to the
    default synchronous `T.gemm(...)` interface, with two stricter guarantees:
    - it always requests the TCGEN5MMA lowering path
    - it never auto-emits an inlined `mbarrier_wait_parity`

    ``mbar=None`` omits the completion arrival for an intermediate issue.  A
    later TCGEN05 operation remains ordered in the same issue stream and may
    publish the completion event for the whole sequence.

    When ``use_2cta=True``, the instruction is lowered to the 2CTA variant
    which requires ``cluster_dims`` to be ``(2,1,1)`` or ``(1,2,1)``.

    If the current target or operand pattern cannot use Blackwell TCGEN5MMA,
    compilation fails instead of silently falling back to another GEMM path.
    """

    ann = _normalize_annotations(annotations)
    ann["is_tcgen05"] = 1
    if use_2cta:
        ann["use_2cta"] = 1
    return _gemm_impl(
        "tl.tileop.tcgen05_gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        annotations=ann,
    )


def tcgen05_gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA_tmem: BufferLikeType,
    SFB_tmem: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
    *,
    k_start: int | tirx.PrimExpr,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    use_2cta: bool = False,
) -> tirx.PrimExpr:
    """Explicit Blackwell TCGEN05 block-scaled GEMM without an implicit wait.

    This is the explicit counterpart of `T.gemm_blockscaled(...)` for
    Blackwell TCGEN5MMA, with the same guarantees as `T.tcgen05_gemm(...)`:
    it always requests the TCGEN5MMA lowering path and compilation fails
    instead of silently falling back if that path is unavailable. It never
    auto-emits an inlined `mbarrier_wait_parity`.

    ``mbar=None`` omits the completion arrival for this issue. The caller
    must publish completion with a later TCGEN05 operation or an explicit
    ``T.tcgen05_mma_arrive`` before waiting and consuming the result.

    With ``use_2cta=True``, this lowers to the true 2CTA block-scaled TCGEN05
    path only; there is no fallback or emulation. That mode requires
    ``cluster_dims`` to be ``(2,1,1)`` or ``(1,2,1)``.

    A and B are FP8/FP6/FP4 mxf8f6f4 operands in shared memory, C is the
    accumulator in tensor memory, and SFA/SFB are E8M0 scale factors already
    resident in tensor memory. The API is explicit-async: it issues the MMA
    and leaves synchronization to the user schedule.

    ``k_start`` is the logical K-axis start offset for this MMA tile.
    ``sf_a_granularity_k`` and ``sf_b_granularity_k`` describe how many K
    elements one packed scale factor covers. The compiler derives the PTX
    scale-factor A/B IDs for each internal K32 MMA atom from these values.

    Args:
        A: FP8/FP6/FP4 input buffer A in shared memory.
        B: FP8/FP6/FP4 input buffer B in shared memory.
        C: Accumulator in tensor memory.
        SFA_tmem: Scale factors for A in tensor memory.
        SFB_tmem: Scale factors for B in tensor memory.
        transpose_A: Whether A is MN-major. Default: False (K-major).
        transpose_B: Whether B is K-major. Default: False (MN-major).
        clear_accum: Whether to zero the accumulator.
        wg_wait: Warp group wait identifier.
        mbar: Completion barrier, or None to defer the completion arrival.
        k_start: Logical K-axis start offset for this MMA tile.
        sf_a_granularity_k: K elements covered by one A scale factor.
        sf_b_granularity_k: K elements covered by one B scale factor.
        use_2cta: Whether to request true ``cta_group::2`` lowering.
    """

    ann: dict = {"is_tcgen05": 1}
    if use_2cta:
        ann["use_2cta"] = 1
    if wg_wait != 0:
        ann["wg_wait"] = wg_wait
    return _gemm_blockscaled_impl(
        "tl.tileop.tcgen05_gemm_blockscaled",
        A,
        B,
        C,
        SFA_tmem,
        SFB_tmem,
        transpose_A,
        transpose_B,
        # Block-scaled TCGEN05 always uses a 1x1 warp partition.
        GemmWarpPolicy.Square,
        clear_accum,
        mbar,
        k_start=k_start,
        sf_a_granularity_k=sf_a_granularity_k,
        sf_b_granularity_k=sf_b_granularity_k,
        annotations=ann,
    )


def mma_gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    *,
    k_start: int | tirx.PrimExpr,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    sf_layout: str | None = None,
) -> tirx.PrimExpr:
    """Explicit SM120 warp-level block-scaled MMA GEMM.

    This is the explicit counterpart of `T.gemm_blockscaled(...)` for the
    SM120 warp-level path and follows the same scale-factor model: users pass
    the scale tensors, logical ``k_start``, and K granularity, while the
    lowering derives the low-level scale addressing. Unlike TCGEN05, this
    path is synchronous warp-level ``mma.sync`` and does not use tensor memory
    or mbarriers, so ``C`` must be a fragment.

    The current supported instruction is SM120 NVF4:
    ``m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`` with E2M1 operands,
    FP32 accumulation, and UE4M3 scale factors.
    """

    ann: dict = {}
    if sf_layout is not None:
        ann["sf_layout"] = sf_layout
    return _gemm_blockscaled_impl(
        "tl.tileop.gemm_blockscaled",
        A,
        B,
        C,
        SFA,
        SFB,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        k_start=k_start,
        sf_a_granularity_k=sf_a_granularity_k,
        sf_b_granularity_k=sf_b_granularity_k,
        annotations=ann,
    )


def make_blockscaled_gemm_layout(
    C: BufferLikeType,
    A: BufferLikeType,
    transpose_A: bool = False,
) -> Layout:
    """Build the TMEM store layout for the C accumulator of a block-scaled GEMM.

    Users must call ``T.annotate_layout({C_tmem: layout})`` with the returned layout
    so that subsequent ``T.copy(C_tmem, ...)`` can be lowered correctly.

    Args:
        C: The TMEM accumulator buffer (block_M, block_N).
        A: The FP8 operand A buffer (used to infer K and dtype).
        transpose_A: Whether A is MN-major.

    Returns:
        A Layout object for C's TMEM storage.
    """
    from tilelang.cuda.intrinsics.macro.tcgen05_macro_generator import TensorCoreIntrinEmitter

    C_region = to_buffer_region(C)
    A_region = to_buffer_region(A)

    C_shape = retrieve_shape(C_region)
    A_shape = retrieve_shape(A_region)

    M, N = int(C_shape[0]), int(C_shape[1])
    K = int(A_shape[-2] if transpose_A else A_shape[-1])
    a_dtype = str(A_region.buffer.dtype)
    accum_dtype = str(C_region.buffer.dtype)

    emitter = TensorCoreIntrinEmitter(
        a_dtype=a_dtype,
        b_dtype=a_dtype,
        accum_dtype=accum_dtype,
        a_transposed=transpose_A,
        b_transposed=False,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=M,
        warp_col_tiles=N,
        chunk=K,
    )
    # Block-scaled GEMM is 1CTA dense (no .ws), matching _lower_blockscaled.
    emitter.get_tcgen5_mma_meta(M, N, K, disable_2cta=True, disable_ws=True)

    c_buf = C_region.buffer if isinstance(C_region, tirx.BufferRegion) else C
    return emitter.make_mma_store_layout(c_buf)
