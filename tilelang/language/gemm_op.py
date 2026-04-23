"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tilelang.layout import Layout
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_offset,
    prim_expr_equal,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)


def _gemm_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
    annotations: dict | None = None,
) -> tir.PrimExpr:
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key.
    """

    def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    mbar = legalize_arguments(mbar) if mbar is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape
    M_A = A_shape[-1] if transpose_A else A_shape[-2]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    N_B = B_shape[-2] if transpose_B else B_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(M_A, M), f"T.gemm M shape check failed: M_A = {M_A}, M_C = {M}"
    assert prim_expr_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"
    use_2cta = annotations is not None and annotations.get("use_2cta", 0)
    if use_2cta:
        # In 2CTA mode each CTA holds half of B along N, so N_B should be N // 2
        assert prim_expr_equal(N_B * 2, N), f"T.gemm N shape check failed for 2CTA: N_B = {N_B}, expected N_C / 2 = {N} / 2"
    else:
        assert prim_expr_equal(N_B, N), f"T.gemm N shape check failed: N_B = {N_B}, N_C = {N}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    if mbar is not None:
        assert isinstance(mbar, (tir.Buffer, tir.BufferLoad)), (
            f"mbar for tcgen5mma must be a tir.Buffer or tir.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")
    C_coords = [r.min for r in C_region.region]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    # When mbar is None, pass a placeholder constant (0).
    # The C++ side checks if arg 16 is a BufferLoadNode before using it,
    # so a non-BufferLoad value will be correctly ignored.
    mbar_arg = mbar if mbar is not None else tir.const(0, dtype="int32")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_key),
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
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
        mbar_arg,
        C_coords[0],
        C_coords[1],
        annotations=annotations,
    )


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
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
        k_pack (int): Numbers of packed matrix cores, for ROCm only. Defaults to 1.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell.
            Required when this GEMM lowers to TCGEN5MMA. Defaults to None.

    Returns:
        tir.Call: A handle to the GEMM operation.
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
        k_pack,
        0,
        mbar,
    )


def wgmma_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tir.PrimExpr:
    """Explicit Hopper WGMMA GEMM without an implicit wait.

    This is the explicit asynchronous Hopper WGMMA counterpart to the default
    synchronous `T.gemm(...)` interface, with two stricter guarantees:
    - it always requests the WGMMA lowering path
    - it never auto-emits an inlined `warpgroup_wait`

    If the current target or operand pattern cannot use Hopper WGMMA,
    compilation fails instead of silently falling back to MMA.
    """

    return _gemm_impl(
        "tl.tileop.wgmma_gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        1,
        -1,
        None,
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
    mbar: BarrierType,
    use_2cta: bool = False,
) -> tir.PrimExpr:
    """Explicit Blackwell TCGEN05 GEMM without an implicit wait.

    This is the explicit asynchronous Blackwell TCGEN5MMA counterpart to the
    default synchronous `T.gemm(...)` interface, with two stricter guarantees:
    - it always requests the TCGEN5MMA lowering path
    - it never auto-emits an inlined `mbarrier_wait_parity`

    When ``use_2cta=True``, the instruction is lowered to the 2CTA variant
    which requires ``cluster_dims`` to be ``(2,1,1)`` or ``(1,2,1)``.

    If the current target or operand pattern cannot use Blackwell TCGEN5MMA,
    compilation fails instead of silently falling back to another GEMM path.
    """

    ann = {"is_tcgen05": 1}
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
        1,
        0,
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
    clear_accum=False,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
    sf_a_id: int = 0,
    sf_b_id: int = 0,
    *,
    use_2cta: bool = False,
) -> tir.PrimExpr:
    """Explicit Blackwell TCGEN05 block-scaled GEMM without an implicit wait.

    This is the explicit asynchronous Blackwell TCGEN5MMA block-scaled
    counterpart to `T.tcgen05_gemm(...)`. It never auto-emits an inlined
    `mbarrier_wait_parity`, and compilation fails instead of silently falling
    back if the requested ISA path is unavailable.

    With ``use_2cta=True``, this lowers to the true 2CTA block-scaled TCGEN05
    path only; there is no fallback or emulation. That mode requires
    ``cluster_dims`` to be ``(2,1,1)`` or ``(1,2,1)``.

    A and B are FP8 (E4M3/E5M2) in shared memory, C is the accumulator in
    tensor memory, and SFA/SFB are E8M0 scale factors already resident in
    tensor memory. As with `T.tcgen05_gemm(...)`, this API is explicit-async:
    it issues the MMA and leaves synchronization to the user schedule.

    Args:
        A: FP8 input buffer A in shared memory.
        B: FP8 input buffer B in shared memory.
        C: Accumulator in tensor memory.
        SFA_tmem: Scale factors for A in tensor memory.
        SFB_tmem: Scale factors for B in tensor memory.
        transpose_A: Whether A is MN-major. Default: False (K-major).
        transpose_B: Whether B is K-major. Default: False (MN-major).
        clear_accum: Whether to zero the accumulator.
        wg_wait: Warp group wait identifier.
        mbar: Mbarrier for MMA completion signaling.
        sf_a_id: Scale factor ID for A (0-3).
        sf_b_id: Scale factor ID for B (0-3).
        use_2cta: Whether to request true ``cta_group::2`` lowering.
    """

    ann = {"use_2cta": int(use_2cta)} if use_2cta else None

    # Re-read normalized regions below after let legalization.

    def legalize(arg):
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize(A)
    B = legalize(B)
    C = legalize(C)
    SFA_tmem = legalize(SFA_tmem)
    SFB_tmem = legalize(SFB_tmem)
    mbar = legalize(mbar) if mbar is not None else None

    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)
    SFA_region = to_buffer_region(SFA_tmem)
    SFB_region = to_buffer_region(SFB_tmem)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"

    M, N = C_shape
    M_A = A_shape[-1] if transpose_A else A_shape[-2]
    N_B = B_shape[-2] if transpose_B else B_shape[-1]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.tcgen05_gemm_blockscaled K shape check failed: K_A = {K}, K_B = {K_B}"
    if use_2cta:
        assert prim_expr_equal(M_A, M) and prim_expr_equal(N_B * 2, N), (
            f"T.tcgen05_gemm_blockscaled 2CTA shape check failed: M_A = {M_A}, expected M_C = {M}; N_B = {N_B}, expected N_C / 2 = {N} / 2"
        )
    else:
        assert prim_expr_equal(N_B, N), f"T.tcgen05_gemm_blockscaled N shape check failed: N_B = {N_B}, N_C = {N}"

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)
    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    if mbar is not None:
        assert isinstance(mbar, (tir.Buffer, tir.BufferLoad)), (
            f"mbar for tcgen5mma must be a tir.Buffer or tir.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")

    C_coords = [r.min for r in C_region.region]

    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    SFA_arg = buffer_region_to_tile_region(SFA_region, "r", list(retrieve_shape(SFA_region)))
    SFB_arg = buffer_region_to_tile_region(SFB_region, "r", list(retrieve_shape(SFB_region)))

    assert mbar is not None, "mbar is required for tcgen05_gemm_blockscaled"

    # Ensure sf_a_id and sf_b_id are PrimExpr
    if not isinstance(sf_a_id, tir.PrimExpr):
        sf_a_id = tir.const(sf_a_id, dtype="int32")
    if not isinstance(sf_b_id, tir.PrimExpr):
        sf_b_id = tir.const(sf_b_id, dtype="int32")

    # Block-scaled always uses Square policy (1x1 warp partition)
    policy = GemmWarpPolicy.Square

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.gemm"),
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
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        1,  # k_pack
        wg_wait,
        mbar,
        C_coords[0],
        C_coords[1],
        SFA_arg,  # arg 19
        SFB_arg,  # arg 20
        sf_a_id,  # arg 21
        sf_b_id,  # arg 22
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
    from tilelang.intrinsics.tcgen05_macro_generator import TensorCoreIntrinEmitter

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

    c_buf = C_region.buffer if isinstance(C_region, tir.BufferRegion) else C
    return emitter.make_mma_store_layout(c_buf)
