"""Ascend two-AIV partitioned copy frontend."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy
from tilelang.utils.language import to_buffer_region
from tvm import tirx

__all__ = ["dual_copy"]


def _get_shape(buf: BufferLikeType) -> list:
    """Extract shape list from Buffer, BufferLoad, or BufferRegion."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, tirx.BufferLoad):
        return list(buf.buffer.shape)
    if isinstance(buf, tirx.BufferRegion):
        return [r.extent for r in buf.region]
    raise TypeError(f"Cannot get shape from {type(buf)}")


def _get_allocation_shape(buf: BufferLikeType) -> list:
    """Extract the underlying allocation shape rather than region extents."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, (tirx.BufferLoad, tirx.BufferRegion)):
        return list(buf.buffer.shape)
    raise TypeError(f"Cannot get allocation shape from {type(buf)}")


def _is_half(full_dim, half_dim) -> bool:
    """Check whether full_dim is exactly twice half_dim."""
    if isinstance(full_dim, (int, tirx.IntImm)) and isinstance(half_dim, (int, tirx.IntImm)):
        full_val = full_dim if isinstance(full_dim, int) else full_dim.value
        half_val = half_dim if isinstance(half_dim, int) else half_dim.value
        return half_val * 2 == full_val
    try:
        from tvm.ir import structural_equal

        return structural_equal(half_dim * 2, full_dim)
    except Exception:
        return False


def dual_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    l2_cache_ctrl: int | str = 0,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy a region using an M- or N-split across the two AIVs.

    ``dual_copy`` is a mixed-kernel operation: the user must also provide the
    Cube-side work in the enclosing kernel. It is not supported as a way to
    turn an otherwise pure Vector kernel into a two-AIV launch; the compiler
    assumes this contract and does not synthesize Cube-side work.

    With auto-scheduling disabled, write a manual mixed kernel as ``T.Kernel``
    containing explicit ``T.Cube()`` and ``T.Vector()`` blocks, and place each
    software dual copy inside the ``T.Vector()`` block. The rewrite pass reuses
    that block's ``cthread`` sid and rejects unscoped software dual copies.

    Software dual copies also accept one-dimensional regions whose sole extent
    has a 2:1 ratio. For regions with rank at least two, exactly one trailing
    dimension must have a 2:1 extent ratio between source and destination. The
    supported memory paths are:

    - L0C→UB: lowered as a hardware dual-destination copy.
    - GM→UB, UB→GM, and UB→L1: lowered as an ordinary per-AIV copy
      indexed by the ``cthread`` sid.

    For UB→L1, the InsertNd2Nz pass also handles ND→NZ format conversion.

    - ``[M, N]`` ↔ ``[M/2, N]``: M-split (``dual_dst_ctl=0b01``)
    - ``[M, N]`` ↔ ``[M, N/2]``: N-split (``dual_dst_ctl=0b10``)

    Hardware L0C→UB requires rank at least two. Its N-split additionally
    requires the full N extent to be a multiple of 32.

    The larger region is the full logical tile. Each AIV processes the
    corresponding half of that tile along the inferred split dimension.
    A statically known full extent must therefore be even; odd extents are
    rejected rather than rounded into unequal partitions.

    Args:
        src: Source region in GM, L0C, or UB, according to the supported paths.
        dst: Destination region in UB, GM, or L1, according to the supported paths.
        unit_flag_ctrl: Unit flag control (0=manual sync, 3=pipelined with mad).
            ``None`` omits the annotation and lowers as 0.
        l2_cache_ctrl (int | str): Ascend L2 cache control policy for the UB→GM
            store path. Accepts an integer or case-insensitive string name
            (e.g. ``"notalloc_clean"``). Defaults to 0 (``"normal_fv"``).
            Ascend only; ignored on other backends.

    Raises:
        ValueError: If the split direction cannot be inferred from shapes.
    """
    src_shape = _get_shape(src)
    dst_shape = _get_shape(dst)

    alloc_src_shape = _get_allocation_shape(src)
    alloc_dst_shape = _get_allocation_shape(dst)

    if len(src_shape) == 1 and len(dst_shape) == 1:
        split_candidates = [(-1, 2)]
        ratio_requirement = "The sole dimension must have an exact 2:1 extent ratio."
    elif len(src_shape) >= 2 and len(dst_shape) >= 2:
        split_candidates = [(-2, 1), (-1, 2)]
        ratio_requirement = "Exactly one trailing dimension must be halved."
    else:
        raise ValueError("dual_copy requires source and destination regions that are both one-dimensional or both have rank at least two")

    def infer_halved_axes(lhs_shape, rhs_shape):
        return [_is_half(lhs_shape[axis], rhs_shape[axis]) or _is_half(rhs_shape[axis], lhs_shape[axis]) for axis, _ in split_candidates]

    halved_axes = infer_halved_axes(src_shape, dst_shape)
    # Runtime tail regions may not retain a structural 2:1 relationship with
    # the fixed allocation. Fall back to allocation shapes for split inference
    # while preserving the actual region extents for the copy.
    if sum(halved_axes) != 1:
        halved_axes = infer_halved_axes(alloc_src_shape, alloc_dst_shape)
    if sum(halved_axes) != 1:
        raise ValueError(f"Cannot infer dual_copy split direction: src shape {src_shape} vs dst shape {dst_shape}. " + ratio_requirement)

    split_axis, dual_dst_ctl = split_candidates[halved_axes.index(True)]

    # Convert both to buffer regions to bypass the shape equality check in copy()
    # This allows src[M,N] to be copied to dst[M/2,N] or dst[M,N/2]
    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)

    # WARNING: dav-950 does NOT support quant_pre (type conversion) with
    # fixpipe (cc->ub) dual-destination copies. If src and dst dtypes differ, the hardware may
    # silently produce incorrect results.
    src_dtype = src_region.buffer.dtype
    dst_dtype = dst_region.buffer.dtype
    if src_dtype != dst_dtype and src_region.buffer.scope() == "shared.l0c":
        raise ValueError(
            f"dual_copy: src dtype ({src_dtype}) != dst dtype ({dst_dtype}). "
            "Type conversion during cc->ub dual-destination copy is NOT supported on dav-950. "
            "Use a separate copy or cast inside a VF block instead."
        )

    annotations = {"dual_dst_ctl": dual_dst_ctl}
    if _is_half(dst_shape[split_axis], src_shape[split_axis]) or _is_half(alloc_dst_shape[split_axis], alloc_src_shape[split_axis]):
        annotations["double"] = True

    return copy(
        src_region,
        dst_region,
        annotations=annotations,
        unit_flag_ctrl=unit_flag_ctrl,
        l2_cache_ctrl=l2_cache_ctrl,
    )
