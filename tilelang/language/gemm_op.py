"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
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

    Like `T.gemm(...)`, this is the synchronous interface: the result is
    complete when the call returns. On Blackwell TCGEN5MMA, TileLang inserts
    the corresponding `mbarrier_wait_parity(...)` implicitly after issue, so
    that path needs a completion barrier, which the CUDA dialect accepts as
    ``mbar`` (it also adds ``use_2cta`` and ``sf_layout``). For manual
    asynchronous scheduling use `T.tcgen05_gemm_blockscaled(...)`.

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
