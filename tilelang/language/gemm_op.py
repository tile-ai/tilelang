"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tilelang.layout import Layout
from tvm import tirx
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    prim_expr_equal,
)
from tilelang.language.utils import (
    _normalize_annotations,
    buffer_region_to_tile_region,
)


def _legalize_buffer_arg(arg: BufferLikeType | tirx.Var) -> BufferLikeType:
    """Convert let-bound variables to their corresponding buffers.

    Args:
        arg (Union[tirx.Buffer, tirx.Var]): Input argument to legalize

    Returns:
        Union[tirx.Buffer, tirx.Var]: The legalized argument
    """
    if isinstance(arg, tirx.Var) and T.has_let_value(arg):
        return T.get_let_value(arg).buffer
    return arg


def _gemm_dense_slots(
    api_name: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool,
    transpose_B: bool,
    policy: GemmWarpPolicy,
    clear_accum: bool,
    mbar: BarrierType | None,
    use_2cta: bool,
) -> list:
    """Validate the A/B/C operands and build the 13 positional slots every GEMM
    tile op starts with (see the protocol documented at ``Gemm::Gemm``).

    ``api_name`` names the user-facing entry point in error messages.
    """

    A = _legalize_buffer_arg(A)
    B = _legalize_buffer_arg(B)
    C = _legalize_buffer_arg(C)
    mbar = _legalize_buffer_arg(mbar) if mbar is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    for shape, name in ((A_shape, "A"), (B_shape, "B"), (C_shape, "C")):
        assert len(shape) >= 2, f"current only support {name} as a 2D or higher-order tensor"
        for i in range(len(shape) - 2):
            assert shape[i] == 1, (
                f"current only support {name} as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape[-2], C_shape[-1]
    M_A = A_shape[-1] if transpose_A else A_shape[-2]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    N_B = B_shape[-2] if transpose_B else B_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(M_A, M), f"{api_name} M shape check failed: M_A = {M_A}, M_C = {M}"
    assert prim_expr_equal(K, K_B), f"{api_name} K shape check failed: K_A = {K}, K_B = {K_B}"
    if use_2cta:
        # In 2CTA mode each CTA holds half of B along N, so N_B should be N // 2
        assert prim_expr_equal(N_B * 2, N), f"{api_name} N shape check failed for 2CTA: N_B = {N_B}, expected N_C / 2 = {N} / 2"
    else:
        assert prim_expr_equal(N_B, N), f"{api_name} N shape check failed: N_B = {N_B}, N_C = {N}"

    for name, dim in (("M", M), ("N", N), ("K", K)):
        if not isinstance(dim, tirx.IntImm):
            raise ValueError(f"{api_name} requires static tile dimensions, but {name} is symbolic: {dim}")

    if mbar is not None:
        assert isinstance(mbar, (tirx.Buffer, tirx.BufferLoad)), (
            f"mbar for {api_name} must be a tirx.Buffer or tirx.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")
    C_coords = [r.min for r in C_region.region[-2:]]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", list(A_shape))
    B_arg = buffer_region_to_tile_region(B_region, "r", list(B_shape))
    C_arg = buffer_region_to_tile_region(C_region, "rw", list(C_shape))
    # When mbar is None, pass a placeholder constant (0). The C++ side only
    # accepts the mbar slot when it is a BufferLoadNode, so the placeholder is
    # correctly ignored.
    mbar_arg = mbar if mbar is not None else tirx.const(0, dtype="int32")
    return [
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        mbar_arg,
        C_coords[0],
        C_coords[1],
    ]


def _gemm_impl(
    op_key: str,
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
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key. Backend lowering knobs
    such as ``k_pack`` and ``wg_wait`` ride in ``annotations``; the dialect
    wrappers and the CUDA gemm variants put them there.
    """

    annotations = _normalize_annotations(annotations)
    slots = _gemm_dense_slots(
        "T.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        use_2cta=bool(annotations.get("use_2cta", 0)),
    )
    return tirx.call_intrin("handle", tirx.op.Op.get(op_key), *slots, annotations=annotations)


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """TileLang GEMM operator.

    This is the default synchronous GEMM interface. On Hopper, if the compiler
    selects WGMMA lowering, TileLang inserts the corresponding wait implicitly.
    On Blackwell TCGEN5MMA, TileLang inserts the corresponding
    `mbarrier_wait_parity(...)` implicitly after issue.

    For manual asynchronous scheduling, use `T.wgmma_gemm(...)` with
    `T.wait_wgmma(...)` on Hopper, or `T.tcgen05_gemm(...)` with
    `T.mbarrier_wait_parity(...)` on Blackwell.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        annotations (Optional[dict]): Additional annotations.

    Backend dialects extend this signature with their hardware's knobs:
    ``tilelang.cuda.language.gemm`` adds ``mbar`` (Blackwell TCGEN5MMA
    barrier), ``tilelang.rocm.language.gemm`` adds ``k_pack`` (packed MFMA).

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
        None,
        annotations=annotations,
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


def _gemm_blockscaled_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool,
    transpose_B: bool,
    policy: GemmWarpPolicy,
    clear_accum: bool,
    mbar: BarrierType | None,
    *,
    k_start: int | tirx.PrimExpr,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    annotations: dict | None,
) -> tirx.PrimExpr:
    """Shared block-scaled GEMM implementation.

    Emits a 16-slot block-scaled GEMM call: the 13 dense GEMM
    slots followed by the SFA region, the SFB region and the logical K-axis
    start offset. Which instruction consumes it is decided by the backend from
    the target and the operand scopes, unless the wrapper requests an explicit
    instruction family through its op key.
    """

    ann = _normalize_annotations(annotations)
    ann["sf_a_granularity_k"] = int(sf_a_granularity_k)
    ann["sf_b_granularity_k"] = int(sf_b_granularity_k)

    slots = _gemm_dense_slots(
        "T.gemm_blockscaled",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        use_2cta=bool(ann.get("use_2cta", 0)),
    )

    SFA_region = to_buffer_region(_legalize_buffer_arg(SFA))
    SFB_region = to_buffer_region(_legalize_buffer_arg(SFB))
    SFA_arg = buffer_region_to_tile_region(SFA_region, "r", list(retrieve_shape(SFA_region)))
    SFB_arg = buffer_region_to_tile_region(SFB_region, "r", list(retrieve_shape(SFB_region)))
    if not isinstance(k_start, tirx.PrimExpr):
        k_start = tirx.const(k_start, dtype="int32")

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get(op_key),
        *slots,
        SFA_arg,
        SFB_arg,
        k_start,
        annotations=ann,
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
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Target-neutral block-scaled GEMM: ``C (+)= (A * SFA) @ (B * SFB)``.

    Scale factors apply to blocks along the reduction axis: ``k_start`` is
    the logical K-axis start of this tile and ``sf_*_granularity_k`` gives
    the number of K elements covered by one scale factor. The backend owns
    the supported dtypes, operand scopes, scale representation and lowering.
    Compilation fails when the backend has no block-scaled implementation;
    lowering to an unscaled GEMM would change the result.

    The CUDA dialect extends this signature with ``mbar``, ``use_2cta`` and
    ``sf_layout``. Its TCGEN05 path requires the CUDA entry point with an
    explicit completion barrier and leaves waiting to the caller.

    Args:
        A: Left operand tile.
        B: Right operand tile.
        C: Accumulator tile.
        SFA: Scale factors for A.
        SFB: Scale factors for B.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: GEMM warp partition policy.
        clear_accum: Whether to zero the accumulator before accumulating.
        k_start: Logical K-axis start offset for this tile.
        sf_a_granularity_k: K elements covered by one A scale factor.
        sf_b_granularity_k: K elements covered by one B scale factor.
        annotations: Additional annotations.
    """
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
        annotations=annotations,
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
