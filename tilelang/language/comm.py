"""Communication intrinsics wrappers for TileLang.

This module provides small helper functions that prepare arguments and
emit TIR intrinsics for inter-core communication on a target mesh.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import tvm_ffi
from tvm import arith, ir, tir
import tilelang.utils.target as _target_utils
import tilelang.language as T
from tilelang._typing import BufferLikeType
from tilelang.language.mesh_tensor import _unwrap_mesh_tensor
from tilelang.utils.language import prim_expr_contains_mesh_symbol, prim_expr_equal, to_buffer_region

from tilelang.carver.arch.driver import get_sunmmio_device_mesh_config

# Mirror of kAttrSrcOffsetByte in src/target/sunmmio_utils.h. Resolved at
# import time via FFI so the single source of truth stays in C++.
ATTR_SRC_OFFSET_BYTE = str(tvm_ffi.get_global_func("tl.target.GetAttrSrcOffsetByte")())

DIRECTION_MAP = {"horizontal": 0, "h": 0, "vertical": 1, "v": 1, "all": 2, "a": 2}
REDUCE_TYPE_LIST = (
    "sum",
    "abssum",
    "max",
    "min",
    "absmax",
    "bitand",
    "bitor",
    "bitxor",
)

CoreCoord = int | tir.PrimExpr
CoreSpec = CoreCoord | tuple[CoreCoord, CoreCoord]
ACCESS_MASK = {"r": 1, "w": 2, "rw": 3}
_OperandKind = Literal["buffer", "region", "load"]

# Keep operand extraction, clipping, rank matching, and destination shrinking
# aligned with the SunMMIO path in copy_op.py.


@dataclass(frozen=True)
class _CommRegionSpec:
    kind: _OperandKind
    buffer: tir.Buffer
    mins: list[tir.PrimExpr]
    extents: list[tir.PrimExpr] | None
    explicit_extents: bool


@dataclass(frozen=True)
class _NormalizedCommRegion:
    spec: _CommRegionSpec
    mins: list[tir.PrimExpr]
    extents: list[tir.PrimExpr]


@dataclass(frozen=True)
class _PreparedCommRegion:
    buffer: tir.Buffer
    extents: list[tir.PrimExpr]
    region: tir.PrimExpr | tir.BufferRegion


def get_target_mesh_shape() -> dict[str, int]:
    """Get the target mesh shape as a dictionary with 'nrow' and 'ncol' keys."""
    nrow, ncol = get_sunmmio_device_mesh_config()
    return {"nrow": nrow, "ncol": ncol}


def _check_core_coord(coord: CoreCoord, limit: int, name: str):
    if isinstance(coord, bool):
        raise TypeError(f"{name} must be an integer or TIR PrimExpr, got bool.")
    coord_int = _const_int(coord)
    if coord_int is not None:
        assert 0 <= coord_int < limit, f"{name} {coord_int} out of bounds for limit {limit}."
    elif not isinstance(coord, tir.PrimExpr):
        raise TypeError(f"{name} must be an integer or TIR PrimExpr, got {type(coord)}.")


def core_to_id(core_id: CoreSpec, name: str = "core") -> CoreCoord:
    """Normalize a linear core id or 2D mesh coordinate into a linear core id.

    Parameters
    ----------
    core_id : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Either a linear core id, or a tuple specifying the (row, col)
        coordinates of the core on the mesh.
    name : str
        User-facing argument name used in diagnostics.

    Returns
    -------
    int | tir.PrimExpr
        The normalized linear core id.

    Notes
    -----
    Dynamic TIR expressions are allowed. Compile-time bounds checks are only
    performed when the id or coordinate is statically known.
    """
    mesh_shape = get_target_mesh_shape()
    if isinstance(core_id, tuple):
        assert len(core_id) == 2, f"{name} must be a linear core id or a tuple of (row, col)."
        row, col = core_id
        _check_core_coord(row, mesh_shape["nrow"], f"{name} row")
        _check_core_coord(col, mesh_shape["ncol"], f"{name} col")
        return row * mesh_shape["ncol"] + col

    _check_core_coord(core_id, mesh_shape["nrow"] * mesh_shape["ncol"], name)
    return core_id


def core_tuple_to_id(core_id: tuple[CoreCoord, CoreCoord]) -> CoreCoord:
    """Convert 2D (row, col) coordinates on the mesh into a linear core id."""
    assert isinstance(core_id, tuple) and len(core_id) == 2, "core_id must be a tuple of (row, col)."
    return core_to_id(core_id)


def _const_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _extent_equal(lhs, rhs) -> bool:
    if prim_expr_equal(lhs, rhs):
        return True
    try:
        return bool(arith.Analyzer().can_prove_equal(lhs, rhs))
    except (TypeError, ValueError):
        return False


def _extent_is_dynamic(extent) -> bool:
    return isinstance(extent, tir.PrimExpr) and _const_int(extent) is None


def _compact_extent_equal(lhs, rhs) -> bool:
    lhs_int = _const_int(lhs)
    rhs_int = _const_int(rhs)
    if lhs_int is not None and rhs_int is not None:
        return lhs_int == rhs_int
    try:
        return bool(ir.structural_equal(lhs, rhs))
    except (TypeError, ValueError):
        return False


def _compact_extent_is_one(extent) -> bool:
    extent_int = _const_int(extent)
    if extent_int is not None:
        return extent_int == 1
    return _compact_extent_equal(extent, tir.IntImm("int32", 1))


def _compact_shape_matches(lhs, rhs, *, allow_broadcast: bool) -> bool:
    if len(lhs) != len(rhs):
        return False

    for lhs_extent, rhs_extent in zip(lhs, rhs):
        matches = _compact_extent_equal(lhs_extent, rhs_extent)
        if allow_broadcast:
            matches = matches or _compact_extent_is_one(lhs_extent) or _compact_extent_is_one(rhs_extent)
        if matches:
            continue
        if prim_expr_contains_mesh_symbol(lhs_extent) or prim_expr_contains_mesh_symbol(rhs_extent):
            continue
        return False

    return True


def _compact_shape_equal(lhs, rhs) -> bool:
    return _compact_shape_matches(lhs, rhs, allow_broadcast=False)


def _compact_shape_compatible(lhs, rhs) -> bool:
    return _compact_shape_matches(lhs, rhs, allow_broadcast=True)


def _prepare_comm_region_compact(obj: BufferLikeType, access_type: str) -> _PreparedCommRegion:
    region = to_buffer_region(obj, access_type=access_type)
    if not isinstance(region, tir.BufferRegion):
        raise TypeError(f"Expected a buffer-like object, got {type(obj)}.")
    extents = [rng.extent for rng in region.region]
    encoded_region: tir.PrimExpr | tir.BufferRegion = region
    if any(prim_expr_contains_mesh_symbol(extent) for extent in extents):
        # BufferRegion call arguments cannot represent symbolic extents, so keep
        # the original shape in an explicit tl.region call until mesh resolution.
        encoded_region = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.tileop.region"),
            tir.BufferLoad(region.buffer, [rng.min for rng in region.region]),
            tir.IntImm("int32", ACCESS_MASK[access_type]),
            *extents,
        )
    return _PreparedCommRegion(
        buffer=region.buffer,
        extents=extents,
        region=encoded_region,
    )


def _resolve_let_value(obj: Any) -> Any:
    from tilelang.language.frame import get_let_value, has_let_value

    if isinstance(obj, tir.Var) and has_let_value(obj):
        return get_let_value(obj)
    return obj


def _extract_comm_region_spec(obj: BufferLikeType, op_name: str) -> _CommRegionSpec:
    obj = _unwrap_mesh_tensor(_resolve_let_value(obj))
    if isinstance(obj, tir.Buffer):
        mins = [tir.IntImm("int32", 0) for _ in obj.shape]
        return _CommRegionSpec("buffer", obj, mins, list(obj.shape), False)
    if isinstance(obj, tir.BufferRegion):
        mins = [region.min for region in obj.region]
        extents = [region.extent for region in obj.region]
        return _CommRegionSpec("region", obj.buffer, mins, extents, True)
    if isinstance(obj, tir.BufferLoad):
        return _CommRegionSpec("load", obj.buffer, list(obj.indices), None, False)
    raise TypeError(f"Unsupported argument type for {op_name}: {type(obj)}")


def _extent_is_squeezable_one(extent: tir.PrimExpr) -> bool:
    if prim_expr_equal(extent, 1):
        return True
    if isinstance(extent, tir.Min):
        return prim_expr_equal(extent.a, 1) or prim_expr_equal(extent.b, 1)
    return False


def _extent_gt(lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> bool:
    lhs_int = _const_int(lhs)
    rhs_int = _const_int(rhs)
    return lhs_int is not None and rhs_int is not None and lhs_int > rhs_int


def _warn_explicit_oob(
    op_name: str,
    buffer: tir.Buffer,
    dim: int,
    min_value: tir.PrimExpr,
    extent: tir.PrimExpr,
    shape: tir.PrimExpr,
) -> None:
    warnings.warn(
        f"{op_name} explicit BufferRegion exceeds buffer shape and will be clipped: "
        f"{buffer.name}[dim={dim}], min={min_value}, extent={extent}, shape={shape}",
        stacklevel=4,
    )


def _clip_extent_to_shape(
    spec: _CommRegionSpec,
    op_name: str,
    dim: int,
    min_value: tir.PrimExpr,
    extent: tir.PrimExpr,
    shape: tir.PrimExpr,
) -> tir.PrimExpr:
    min_int = _const_int(min_value)
    extent_int = _const_int(extent)
    shape_int = _const_int(shape)

    if min_int is not None and shape_int is not None and (min_int < 0 or min_int >= shape_int):
        raise ValueError(
            f"{op_name} region starts outside buffer shape: {spec.buffer.name}[dim={dim}], min={min_value}, extent={extent}, shape={shape}"
        )

    if min_int is not None and extent_int is not None and shape_int is not None:
        available = shape_int - min_int
        clipped = min(extent_int, available)
        if clipped < extent_int and spec.explicit_extents:
            _warn_explicit_oob(op_name, spec.buffer, dim, min_value, extent, shape)
        return tir.IntImm(extent.dtype if hasattr(extent, "dtype") else "int32", clipped)

    return extent


def _clip_region_to_shape(
    spec: _CommRegionSpec,
    mins: list[tir.PrimExpr],
    extents: list[tir.PrimExpr],
    op_name: str,
) -> _NormalizedCommRegion:
    if len(mins) != len(extents) or len(extents) != len(spec.buffer.shape):
        raise ValueError(
            f"{op_name} region rank does not match buffer rank before clipping: "
            f"{spec.buffer.name}, mins={len(mins)}, extents={len(extents)}, shape={len(spec.buffer.shape)}"
        )
    clipped_extents = [
        _clip_extent_to_shape(spec, op_name, dim, min_value, extent, shape)
        for dim, (min_value, extent, shape) in enumerate(zip(mins, extents, spec.buffer.shape))
    ]
    return _NormalizedCommRegion(spec, list(mins), clipped_extents)


def _int_one() -> tir.IntImm:
    return tir.IntImm("int32", 1)


def _infer_load_extents_from_peer(spec: _CommRegionSpec, peer_extents: list[tir.PrimExpr]) -> list[tir.PrimExpr]:
    rank = len(spec.mins)
    extents = list(peer_extents)
    if len(extents) < rank:
        return [_int_one() for _ in range(rank - len(extents))] + extents
    return extents[-rank:]


def _normalize_comm_regions(
    src: _CommRegionSpec,
    dst: _CommRegionSpec,
    op_name: str,
) -> tuple[_NormalizedCommRegion, _NormalizedCommRegion]:
    if src.kind == "load" and dst.kind == "load":
        raise ValueError(f"{op_name} cannot infer extents when both operands are BufferLoad values.")

    if src.kind == "load" and dst.kind == "region":
        assert dst.extents is not None
        dst_region = _clip_region_to_shape(dst, dst.mins, list(dst.extents), op_name)
        src_extents = _infer_load_extents_from_peer(src, dst_region.extents)
        src_region = _clip_region_to_shape(src, src.mins, src_extents, op_name)
        return src_region, dst_region

    if src.kind == "region" and dst.kind == "load":
        assert src.extents is not None
        src_region = _clip_region_to_shape(src, src.mins, list(src.extents), op_name)
        dst_extents = _infer_load_extents_from_peer(dst, src_region.extents)
        dst_region = _clip_region_to_shape(dst, dst.mins, dst_extents, op_name)
        return src_region, dst_region

    src_extents = src.extents
    dst_extents = dst.extents
    if src.kind == "load":
        assert dst_extents is not None
        src_extents = _infer_load_extents_from_peer(src, dst_extents)
    if dst.kind == "load":
        assert src_extents is not None
        dst_extents = _infer_load_extents_from_peer(dst, src_extents)

    assert src_extents is not None and dst_extents is not None
    src_region = _clip_region_to_shape(src, src.mins, list(src_extents), op_name)
    dst_region = _clip_region_to_shape(dst, dst.mins, list(dst_extents), op_name)
    return src_region, dst_region


def _format_extents(extents: list[tir.PrimExpr]) -> str:
    return "[" + ", ".join(str(extent) for extent in extents) + "]"


def _suffix_axis_map(
    src: _NormalizedCommRegion,
    dst: _NormalizedCommRegion,
    op_name: str,
) -> list[tuple[int, int]]:
    src_rank = len(src.extents)
    dst_rank = len(dst.extents)
    matched_rank = min(src_rank, dst_rank)

    if src_rank > dst_rank:
        for dim, extent in enumerate(src.extents[: src_rank - dst_rank]):
            if not prim_expr_equal(extent, 1):
                raise ValueError(
                    f"{op_name} rank mismatch: src has non-1 extra leading dimension at dim {dim}, "
                    f"extent={extent}; src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
                )
        return [(src_rank - matched_rank + dim, dim) for dim in range(matched_rank)]

    if dst_rank > src_rank:
        for dim, extent in enumerate(dst.extents[: dst_rank - src_rank]):
            if not prim_expr_equal(extent, 1):
                raise ValueError(
                    f"{op_name} rank mismatch: dst has non-1 extra leading dimension at dim {dim}, "
                    f"extent={extent}; src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
                )
        return [(dim, dst_rank - matched_rank + dim) for dim in range(matched_rank)]

    return [(dim, dim) for dim in range(matched_rank)]


def _squeezed_axis_map(
    src: _NormalizedCommRegion,
    dst: _NormalizedCommRegion,
    op_name: str,
) -> list[tuple[int, int]]:
    if len(src.extents) == len(dst.extents):
        identity_compatible = True
        for src_extent, dst_extent in zip(src.extents, dst.extents):
            if _extent_equal(src_extent, dst_extent) or _extent_is_squeezable_one(src_extent):
                continue
            if _extent_is_squeezable_one(dst_extent) or _extent_gt(src_extent, dst_extent):
                identity_compatible = False
                break
        if identity_compatible:
            return [(dim, dim) for dim in range(len(src.extents))]

    src_axes = [(dim, extent) for dim, extent in enumerate(src.extents) if not _extent_is_squeezable_one(extent)]
    dst_axes = [(dim, extent) for dim, extent in enumerate(dst.extents) if not _extent_is_squeezable_one(extent)]
    if len(src_axes) != len(dst_axes):
        raise ValueError(
            f"{op_name} rank mismatch: mixed-region operation requires the same number of non-1 extents "
            f"after squeezing unit dimensions; src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
        )
    return [(src_dim, dst_dim) for (src_dim, _), (dst_dim, _) in zip(src_axes, dst_axes)]


def _validate_and_adjust_comm_regions(
    src: _NormalizedCommRegion,
    dst: _NormalizedCommRegion,
    op_name: str,
    *,
    require_exact_match: bool,
    allow_dynamic_exact_mismatch: bool = False,
) -> tuple[_NormalizedCommRegion, _NormalizedCommRegion]:
    axis_map = _suffix_axis_map(src, dst, op_name) if require_exact_match else _squeezed_axis_map(src, dst, op_name)
    dst_extents = list(dst.extents)

    for src_dim, dst_dim in axis_map:
        src_extent = src.extents[src_dim]
        dst_extent = dst.extents[dst_dim]
        if require_exact_match:
            if not _extent_equal(src_extent, dst_extent):
                if allow_dynamic_exact_mismatch and (_extent_is_dynamic(src_extent) or _extent_is_dynamic(dst_extent)):
                    continue
                raise ValueError(
                    f"{op_name} extent mismatch: exact match is required for Buffer-to-Buffer operation; "
                    f"src dim {src_dim} extent={src_extent}, dst dim {dst_dim} extent={dst_extent}; "
                    f"src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
                )
            continue

        if _extent_gt(src_extent, dst_extent):
            raise ValueError(
                f"{op_name} extent mismatch: src extent is larger than dst extent at matched axis; "
                f"src dim {src_dim} extent={src_extent}, dst dim {dst_dim} extent={dst_extent}; "
                f"src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
            )
        if not _extent_equal(src_extent, dst_extent):
            dst_extents[dst_dim] = src_extent

    return src, _NormalizedCommRegion(dst.spec, dst.mins, dst_extents)


def _normalize_one_to_one_regions(
    src: BufferLikeType,
    dst: BufferLikeType,
    op_name: str,
) -> tuple[_NormalizedCommRegion, _NormalizedCommRegion]:
    src_spec = _extract_comm_region_spec(src, op_name)
    dst_spec = _extract_comm_region_spec(dst, op_name)
    src_region, dst_region = _normalize_comm_regions(src_spec, dst_spec, op_name)
    return _validate_and_adjust_comm_regions(
        src_region,
        dst_region,
        op_name,
        require_exact_match=src_spec.kind == "buffer" and dst_spec.kind == "buffer",
    )


def _normalize_known_region(spec: _CommRegionSpec, op_name: str) -> _NormalizedCommRegion:
    assert spec.extents is not None
    return _clip_region_to_shape(spec, spec.mins, list(spec.extents), op_name)


def _allgather_recv_num(direction: str) -> int:
    mesh_shape = get_target_mesh_shape()
    if direction in ("horizontal", "h"):
        return mesh_shape["ncol"]
    if direction in ("vertical", "v"):
        return mesh_shape["nrow"]
    return mesh_shape["nrow"] * mesh_shape["ncol"]


def _normalize_allgather_axis(axis: int | None, send_rank: int) -> int:
    if axis is None:
        return -1

    assert isinstance(axis, int) and -send_rank <= axis < send_rank, (
        f"axis {axis} out of range for send buffer with {send_rank} dimensions."
    )
    normalized_axis = axis if axis >= 0 else axis + send_rank
    assert normalized_axis == 0 or normalized_axis == send_rank - 1, (
        f"Only axis=0 or axis=-1 (last dim) are currently supported, got axis={axis} "
        f"(normalized to {normalized_axis}) for {send_rank}-D send buffer."
    )
    return normalized_axis


def _allgather_gather_dim(axis: int) -> int:
    return 0 if axis < 0 else axis


def _divide_allgather_extent(
    extent: tir.PrimExpr,
    divisor: int,
    op_name: str,
    dim: int,
    *,
    require_divisible: bool,
) -> tir.PrimExpr:
    extent_int = _const_int(extent)
    if extent_int is not None:
        if require_divisible and extent_int % divisor != 0:
            raise ValueError(
                f"{op_name} recv extent at gather dim {dim} must be divisible by "
                f"the number of receiving cores ({divisor}), but got {extent}."
            )
        dtype = extent.dtype if hasattr(extent, "dtype") else "int32"
        return tir.IntImm(dtype, extent_int // divisor)
    return tir.floordiv(extent, divisor)


def _allgather_slot_region(
    recv: _NormalizedCommRegion,
    recv_num: int,
    axis: int,
    op_name: str,
    *,
    require_exact_match: bool,
) -> _NormalizedCommRegion:
    if axis < 0:
        return _NormalizedCommRegion(recv.spec, recv.mins[1:], recv.extents[1:])
    gather_dim = _allgather_gather_dim(axis)
    slot_extents = list(recv.extents)
    slot_extents[gather_dim] = _divide_allgather_extent(
        slot_extents[gather_dim],
        recv_num,
        op_name,
        gather_dim,
        require_divisible=require_exact_match,
    )
    return _NormalizedCommRegion(recv.spec, recv.mins, slot_extents)


def _allgather_recv_extents(slot_extents: list[tir.PrimExpr], recv_num: int, axis: int) -> list[tir.PrimExpr]:
    assert axis >= 0
    gather_dim = _allgather_gather_dim(axis)
    recv_extents = list(slot_extents)
    recv_extents[gather_dim] = recv_extents[gather_dim] * recv_num
    return recv_extents


def _normalize_allgather_leading_extent(
    recv: _NormalizedCommRegion,
    recv_num: int,
    op_name: str,
    *,
    require_exact_match: bool,
) -> _NormalizedCommRegion:
    extent = recv.extents[0]
    extent_int = _const_int(extent)
    if extent_int is None:
        return recv
    if extent_int < recv_num:
        raise ValueError(f"{op_name} recv leading extent is too small for {recv_num} receiving cores: got {extent}.")
    if require_exact_match and extent_int != recv_num:
        raise ValueError(
            f"{op_name} recv leading extent must equal the number of receiving cores ({recv_num}) "
            f"for Buffer-to-Buffer operation, but got {extent}."
        )
    if extent_int == recv_num:
        return recv
    recv_extents = list(recv.extents)
    recv_extents[0] = tir.IntImm(extent.dtype if hasattr(extent, "dtype") else "int32", recv_num)
    return _NormalizedCommRegion(recv.spec, recv.mins, recv_extents)


def _expected_allgather_recv_extents(send_extents: list[tir.PrimExpr], recv_num: int, axis: int) -> list[tir.PrimExpr]:
    if axis < 0:
        return [tir.IntImm("int32", recv_num), *send_extents]
    recv_extents = list(send_extents)
    recv_extents[axis] = recv_extents[axis] * recv_num
    return recv_extents


def _normalize_allgather_regions(
    send_buffer: BufferLikeType,
    recv_buffer: BufferLikeType,
    recv_num: int,
    axis: int,
    op_name: str,
) -> tuple[_NormalizedCommRegion, _NormalizedCommRegion]:
    # Apply one-to-one copy rules to send and one per-core slot in recv.
    send_spec = _extract_comm_region_spec(send_buffer, op_name)
    recv_spec = _extract_comm_region_spec(recv_buffer, op_name)
    send_rank = len(send_spec.mins)
    recv_rank = len(recv_spec.mins)

    expected_recv_rank = send_rank + 1 if axis < 0 else send_rank
    if recv_rank != expected_recv_rank:
        mode = "new-leading-axis" if axis < 0 else f"axis={axis}"
        raise ValueError(
            f"{op_name} rank mismatch for {mode}: send rank is {send_rank}, recv rank must be {expected_recv_rank}, but got {recv_rank}."
        )
    if send_spec.kind == "load" and recv_spec.kind == "load":
        raise ValueError(f"{op_name} cannot infer extents when both operands are BufferLoad values.")

    send_region = None if send_spec.kind == "load" else _normalize_known_region(send_spec, op_name)
    recv_region = None if recv_spec.kind == "load" else _normalize_known_region(recv_spec, op_name)

    if send_region is None:
        assert recv_region is not None
        recv_slot = _allgather_slot_region(
            recv_region,
            recv_num,
            axis,
            op_name,
            require_exact_match=False,
        )
        send_extents = _infer_load_extents_from_peer(send_spec, recv_slot.extents)
        send_region = _clip_region_to_shape(send_spec, send_spec.mins, send_extents, op_name)

    if recv_region is None:
        recv_extents = _expected_allgather_recv_extents(send_region.extents, recv_num, axis)
        recv_region = _clip_region_to_shape(recv_spec, recv_spec.mins, recv_extents, op_name)

    require_exact_match = send_spec.kind == "buffer" and recv_spec.kind == "buffer"
    if axis < 0:
        recv_region = _normalize_allgather_leading_extent(
            recv_region,
            recv_num,
            op_name,
            require_exact_match=require_exact_match,
        )
    recv_slot = _allgather_slot_region(
        recv_region,
        recv_num,
        axis,
        op_name,
        require_exact_match=require_exact_match,
    )
    send_region, recv_slot = _validate_and_adjust_comm_regions(
        send_region,
        recv_slot,
        op_name,
        require_exact_match=require_exact_match,
        allow_dynamic_exact_mismatch=True,
    )
    if require_exact_match:
        adjusted_recv_extents = recv_region.extents
    elif axis < 0:
        adjusted_recv_extents = [recv_region.extents[0], *recv_slot.extents]
    else:
        adjusted_recv_extents = _allgather_recv_extents(recv_slot.extents, recv_num, axis)
    recv_region = _NormalizedCommRegion(recv_spec, recv_region.mins, adjusted_recv_extents)
    return send_region, recv_region


def _allreduce_result_region(
    src: _NormalizedCommRegion,
    out_rank: int,
    dim: int,
    op_name: str,
) -> _NormalizedCommRegion:
    src_rank = len(src.extents)
    if out_rank == src_rank - 1:
        mins = src.mins[:dim] + src.mins[dim + 1 :]
        extents = src.extents[:dim] + src.extents[dim + 1 :]
    elif out_rank == src_rank:
        mins = list(src.mins)
        extents = list(src.extents)
        extents[dim] = _int_one()
    else:
        raise ValueError(
            f"{op_name} output rank must be input rank - 1 or input rank; input rank is {src_rank}, output rank is {out_rank}."
        )
    return _NormalizedCommRegion(src.spec, mins, extents)


def _infer_allreduce_input_extents(
    src_spec: _CommRegionSpec,
    out_region: _NormalizedCommRegion,
    dim: int,
    op_name: str,
) -> list[tir.PrimExpr]:
    src_rank = len(src_spec.mins)
    out_rank = len(out_region.extents)
    if out_rank == src_rank - 1:
        extents = list(out_region.extents)
        extents.insert(dim, src_spec.buffer.shape[dim])
        return extents
    if out_rank == src_rank:
        extents = list(out_region.extents)
        extents[dim] = src_spec.buffer.shape[dim]
        return extents
    raise ValueError(f"{op_name} output rank must be input rank - 1 or input rank; input rank is {src_rank}, output rank is {out_rank}.")


def _normalize_allreduce_regions(
    src: BufferLikeType,
    out: BufferLikeType,
    dim: int,
    op_name: str,
) -> tuple[_NormalizedCommRegion, _NormalizedCommRegion]:
    # Apply one-to-one copy rules after removing or retaining the reduced dim.
    src_spec = _extract_comm_region_spec(src, op_name)
    out_spec = _extract_comm_region_spec(out, op_name)
    src_rank = len(src_spec.mins)
    out_rank = len(out_spec.mins)

    if out_rank not in (src_rank - 1, src_rank):
        raise ValueError(
            f"{op_name} output rank must be input rank - 1 or input rank; input rank is {src_rank}, output rank is {out_rank}."
        )
    if src_spec.kind == "load" and out_spec.kind == "load":
        raise ValueError(f"{op_name} cannot infer extents when both operands are BufferLoad values.")

    src_region = None if src_spec.kind == "load" else _normalize_known_region(src_spec, op_name)
    out_region = None if out_spec.kind == "load" else _normalize_known_region(out_spec, op_name)

    if src_region is None:
        assert out_region is not None
        src_extents = _infer_allreduce_input_extents(src_spec, out_region, dim, op_name)
        src_region = _clip_region_to_shape(src_spec, src_spec.mins, src_extents, op_name)

    if out_region is None:
        result_region = _allreduce_result_region(src_region, out_rank, dim, op_name)
        out_region = _clip_region_to_shape(out_spec, out_spec.mins, result_region.extents, op_name)

    result_region = _allreduce_result_region(src_region, out_rank, dim, op_name)
    _, out_region = _validate_and_adjust_comm_regions(
        result_region,
        out_region,
        op_name,
        require_exact_match=src_spec.kind == "buffer" and out_spec.kind == "buffer",
        allow_dynamic_exact_mismatch=True,
    )
    return src_region, out_region


def _encode_normalized_region(region: _NormalizedCommRegion, access_type: str) -> tir.PrimExpr | tir.BufferRegion:
    ranges = [ir.Range.from_min_extent(min_value, extent) for min_value, extent in zip(region.mins, region.extents)]
    buffer_region = tir.BufferRegion(region.spec.buffer, ranges)
    if not any(_extent_is_dynamic(extent) for extent in region.extents):
        return buffer_region
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.region"),
        tir.BufferLoad(region.spec.buffer, region.mins),
        tir.IntImm("int32", ACCESS_MASK[access_type]),
        *region.extents,
    )


def _encode_full_buffer_region(buffer: tir.Buffer, access_type: str, op_name: str) -> tir.PrimExpr | tir.BufferRegion:
    spec = _extract_comm_region_spec(buffer, op_name)
    return _encode_normalized_region(_normalize_known_region(spec, op_name), access_type)


def _const_product(extents):
    result = 1
    for extent in extents:
        extent_int = _const_int(extent)
        if extent_int is None:
            return None
        result *= extent_int
    return result


def _check_size(size: int, extents, op_name: str):
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    elements = _const_product(extents)
    if size >= 0 and elements is not None:
        assert size <= elements, f"size {size} exceeds {op_name} buffer size {elements}."


def _prepare_normalized_comm_region(
    region: _NormalizedCommRegion,
    access_type: str,
) -> _PreparedCommRegion:
    return _PreparedCommRegion(
        buffer=region.spec.buffer,
        extents=list(region.extents),
        region=_encode_normalized_region(region, access_type),
    )


def _prepare_one_to_one_operands(
    src: BufferLikeType,
    dst: BufferLikeType,
    op_name: str,
) -> tuple[_PreparedCommRegion, _PreparedCommRegion]:
    if _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION:
        src_region, dst_region = _normalize_one_to_one_regions(src, dst, op_name)
        return (
            _prepare_normalized_comm_region(src_region, "r"),
            _prepare_normalized_comm_region(dst_region, "w"),
        )

    src_region = _prepare_comm_region_compact(src, "r")
    dst_region = _prepare_comm_region_compact(dst, "w")
    if not _compact_shape_compatible(src_region.extents, dst_region.extents):
        operation = op_name.rsplit(".", 1)[-1]
        raise ValueError(f"Source and destination buffer must have the same number of dimensions for {operation}.")
    return src_region, dst_region


def _prepare_allgather_operands(
    send_buffer: BufferLikeType,
    recv_buffer: BufferLikeType,
    recv_num: int,
    axis: int | None,
    op_name: str,
) -> tuple[_PreparedCommRegion, _PreparedCommRegion, int]:
    if _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION:
        send_spec = _extract_comm_region_spec(send_buffer, op_name)
        axis_arg = _normalize_allgather_axis(axis, len(send_spec.mins))
        send_region, recv_region = _normalize_allgather_regions(
            send_buffer,
            recv_buffer,
            recv_num,
            axis_arg,
            op_name,
        )
        return (
            _prepare_normalized_comm_region(send_region, "r"),
            _prepare_normalized_comm_region(recv_region, "w"),
            axis_arg,
        )

    send_region = _prepare_comm_region_compact(send_buffer, "r")
    recv_region = _prepare_comm_region_compact(recv_buffer, "w")
    axis_arg = _normalize_allgather_axis(axis, len(send_region.extents))
    if axis_arg < 0:
        expected_recv_shape = [recv_num, *send_region.extents]
    else:
        expected_recv_shape = list(send_region.extents)
        expected_recv_shape[axis_arg] = recv_num * send_region.extents[axis_arg]
    assert _compact_shape_equal(recv_region.extents, expected_recv_shape), (
        f"Receive buffer shape must be {expected_recv_shape} to hold gathered data from {recv_num} cores, but got {recv_region.extents}."
    )
    return send_region, recv_region, axis_arg


def _prepare_allreduce_operands(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int,
    op_name: str,
) -> tuple[_PreparedCommRegion, _PreparedCommRegion, int]:
    if _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION:
        buffer_spec = _extract_comm_region_spec(buffer, op_name)
        buffer_rank = len(buffer_spec.mins)
        assert isinstance(dim, int) and -1 <= dim < buffer_rank, f"dim {dim} out of bounds for buffer with {buffer_rank} dimensions."
        normalized_dim = buffer_rank - 1 if dim == -1 else dim
        buffer_region, out_region = _normalize_allreduce_regions(
            buffer,
            out,
            normalized_dim,
            op_name,
        )
        return (
            _prepare_normalized_comm_region(buffer_region, "r"),
            _prepare_normalized_comm_region(out_region, "w"),
            normalized_dim,
        )

    buffer_region = _prepare_comm_region_compact(buffer, "r")
    out_region = _prepare_comm_region_compact(out, "w")
    buffer_rank = len(buffer_region.extents)
    assert isinstance(dim, int) and -1 <= dim < buffer_rank, f"dim {dim} out of bounds for buffer with {buffer_rank} dimensions."
    normalized_dim = buffer_rank - 1 if dim == -1 else dim
    expected_shapes = [
        buffer_region.extents[:normalized_dim] + buffer_region.extents[normalized_dim + 1 :],
        buffer_region.extents[:normalized_dim] + [1] + buffer_region.extents[normalized_dim + 1 :],
    ]
    if not any(_compact_shape_equal(out_region.extents, shape) for shape in expected_shapes):
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer_region.extents}, dim is {normalized_dim}, "
            f"output shape is {out_region.extents}, expected shapes are {expected_shapes_str}"
        )
    return buffer_region, out_region, normalized_dim


def _prepare_allreduce_temporary(buffer: tir.Buffer, access_type: str) -> tir.PrimExpr | tir.BufferRegion:
    if _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION:
        return _encode_full_buffer_region(buffer, access_type, "T.comm.all_reduce")
    return _prepare_comm_region_compact(buffer, access_type).region


def broadcast(
    src: BufferLikeType,
    dst: BufferLikeType,
    src_core: CoreSpec,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
):
    """Broadcast data from a source buffer on a specific source core to a destination buffer
    on all cores in the specified direction by emitting the TIR intrinsic tl.tileop.comm_broadcast.
    Parameters
    ----------
    src : BufferLikeType
        Source buffer containing data to broadcast.
    dst : BufferLikeType
        Destination buffer to receive the broadcasted data.
    src_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear source core id, or (row, col) coordinates of the source core on
        the target mesh. Dynamic TIR expressions such as the block id returned
        by ``T.Kernel`` are allowed.
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of broadcast: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    size : int
        Number of elements to broadcast. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_broadcast`.
    Examples
    --------
    >>> broadcast(A, B, (1, 2), direction="horizontal")
    >>> broadcast(A, B, cid, direction="horizontal")
    """
    src_region, dst_region = _prepare_one_to_one_operands(src, dst, "T.comm.broadcast")
    src_dtype = src_region.buffer.dtype
    dst_dtype = dst_region.buffer.dtype
    assert src_dtype == dst_dtype, f"Source and destination buffer dtypes must match for broadcast. Got {src_dtype} vs {dst_dtype}."

    _check_size(size, src_region.extents, "source")

    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    src_core_id = core_to_id(src_core, "src_core")

    args = (
        src_region.region,
        dst_region.region,
        size,
        src_core_id,
        DIRECTION_MAP[direction.lower()],
    )
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_broadcast"), *args)


def put(
    src: BufferLikeType,
    dst: BufferLikeType,
    src_core: CoreSpec,
    dst_core: CoreSpec,
    size: int = -1,
):
    """Put data from a source buffer on a specific source core to a destination buffer on a specific destination core
    by emitting the TIR intrinsic tl.tileop.comm_put.
    Parameters
    ----------
    src : BufferLikeType
        Source buffer containing data to put.
    dst : BufferLikeType
        Destination buffer to receive the data.
    src_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear source core id, or (row, col) coordinates of the source core on
        the target mesh. Dynamic TIR expressions such as the block id returned
        by ``T.Kernel`` are allowed.
    dst_core : int | tir.PrimExpr | tuple[int | tir.PrimExpr, int | tir.PrimExpr]
        Linear destination core id, or (row, col) coordinates of the destination
        core on the target mesh. Dynamic TIR expressions are allowed.
    size : int
        Number of elements to put. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_put`.
    Examples
    --------
    >>> put(A, B, (1, 2), (2, 3))
    >>> put(A, B, cid, (cid + 1) % 16)
    """
    src_region, dst_region = _prepare_one_to_one_operands(src, dst, "T.comm.put")
    src_dtype = src_region.buffer.dtype
    dst_dtype = dst_region.buffer.dtype
    assert src_dtype == dst_dtype, f"Source and destination buffer dtypes must match for put. Got {src_dtype} vs {dst_dtype}."

    _check_size(size, src_region.extents, "source")

    src_core_id = core_to_id(src_core, "src_core")
    dst_core_id = core_to_id(dst_core, "dst_core")
    args = (src_region.region, dst_region.region, size, src_core_id, dst_core_id)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_put"), *args)


def all_gather(
    send_buffer: BufferLikeType,
    recv_buffer: BufferLikeType,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"] = "all",
    size: int = -1,
    axis: int | None = None,
    src_offset_byte: int = 0,
):
    """Perform an all-gather operation from a send buffer to a receive buffer
    by emitting the TIR intrinsic tl.tileop.comm_allgather.
    Parameters
    ----------
    send_buffer : BufferLikeType
        Buffer containing data to send.
    recv_buffer : BufferLikeType
        Buffer to receive gathered data.
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of all-gather: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    size : int
        Number of elements to send from each core. If -1, the entire send buffer is used.
    axis : int, optional
        Axis along which gathered data is concatenated. When ``axis`` is ``None``
        (default), a new leading axis is introduced and ``recv_buffer`` must have
        shape ``[K, *send_buffer.shape]`` where ``K`` is the number of contributing
        cores. When ``axis`` is an integer, gathered data is concatenated along
        that existing axis: ``recv_buffer.shape[axis] == K * send_buffer.shape[axis]``
        and all other dimensions match ``send_buffer``. Only ``axis=0`` and
        ``axis=-1`` (the last dim) are currently supported.
    src_offset_byte : int
        Byte offset added to the source pointer at codegen. Default 0. Set by the
        Sunmmio bf16 GEMM legalization pass to re-stage south-bound A data into a
        destination buffer's north bank. User code should leave this at 0.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allgather`.
    Examples
    --------
    >>> all_gather(A_local, C_local, direction="horizontal")
    >>> # send [d0, d1], 4-col mesh, axis=0 -> recv [4*d0, d1]
    >>> all_gather(A_local, R_local, direction="horizontal", axis=0)
    >>> # send [d0, d1], 4-col mesh, axis=-1 -> recv [d0, 4*d1]
    >>> all_gather(A_local, R_local, direction="horizontal", axis=-1)
    """
    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"

    direction = direction.lower()
    recv_num = _allgather_recv_num(direction)
    send_region, recv_region, axis_arg = _prepare_allgather_operands(
        send_buffer,
        recv_buffer,
        recv_num,
        axis,
        "T.comm.all_gather",
    )
    send_dtype = send_region.buffer.dtype
    recv_dtype = recv_region.buffer.dtype
    assert send_dtype == recv_dtype, f"Source and destination buffer dtypes must match for all_gather. Got {send_dtype} vs {recv_dtype}."

    # Sentinel -1 in the wire format means "no axis specified" (legacy
    # new-leading-axis semantics). User-facing axis is normalized to a
    # non-negative index before being forwarded.
    _check_size(size, send_region.extents, "send")

    assert isinstance(src_offset_byte, int) and src_offset_byte >= 0, "src_offset_byte must be a non-negative integer."

    cid = T.get_block_binding(0)

    args = (
        send_region.region,
        recv_region.region,
        DIRECTION_MAP[direction],
        size,
        axis_arg,
        cid,
    )
    ann = {ATTR_SRC_OFFSET_BYTE: src_offset_byte} if src_offset_byte != 0 else None
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allgather"), *args, annotations=ann)


def all_reduce(
    buffer: BufferLikeType,
    out: BufferLikeType,
    reduce_type: str,
    direction: Literal["horizontal", "h", "vertical", "v", "all", "a"],
    dim: int = -1,
    clear: bool = True,
):
    """Perform an all-reduce operation on a buffer and store the result in an output buffer
    by emitting the TIR intrinsic tl.tileop.comm_allreduce.
    Parameters
    ----------
    buffer : BufferLikeType
        Input buffer containing data to reduce.
    out : BufferLikeType
        Output buffer to store the reduced result.
    reduce_type : str
        Type of reduction operation (e.g., "sum", "max", etc.).
    direction : Literal["horizontal", "h", "vertical", "v", "all", "a"]
        Direction of all-reduce: "horizontal" (or "h") for row-wise, "vertical" (or "v") for column-wise,
        and "all" (or "a") for all cores.
    dim : int
        Dimension along which to perform the reduction. Default is -1 (last dimension).
    clear : bool
        Whether to clear the output buffer before reduction. Default is True.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allreduce`.
    Examples
    --------
    >>> all_reduce(A_local, E_local, "sum", "all", dim=-1, clear=False)
    """
    buffer_region, out_region, dim = _prepare_allreduce_operands(
        buffer,
        out,
        dim,
        "T.comm.all_reduce",
    )
    out_buffer = out_region.buffer
    out_dtype = out_buffer.dtype
    out_shape = out_region.extents

    reduce_type = reduce_type.lower()
    assert reduce_type in REDUCE_TYPE_LIST, f"Reduction op must be one of {REDUCE_TYPE_LIST}, but got {reduce_type}."

    assert direction.lower() in DIRECTION_MAP, f"Invalid direction string: {direction}"
    assert clear in [True, False], "clear must be a boolean value."

    mesh_shape = get_target_mesh_shape()

    # Create temporary buffers for row and column allgather results.  Keep the
    # temporaries in the output scope because the lowered Sunmmio path feeds
    # them back into ReduceOp and broadcast_.
    out_scope = out_buffer.scope()

    def alloc_tmp(shape):
        if out_scope.startswith("shared"):
            return T.alloc_shared(shape, out_dtype, scope=out_scope)
        return T.alloc_fragment(shape, out_dtype, scope=out_scope)

    direction_value = DIRECTION_MAP[direction.lower()]
    if direction_value == 0:
        row_allgather = alloc_tmp([mesh_shape["ncol"]] + list(out_shape))
        col_allgather = row_allgather
    elif direction_value == 1:
        col_allgather = alloc_tmp([mesh_shape["nrow"]] + list(out_shape))
        row_allgather = col_allgather
    else:
        row_allgather = alloc_tmp([mesh_shape["ncol"]] + list(out_shape))
        col_allgather = alloc_tmp([mesh_shape["nrow"]] + list(out_shape))

    row_allgather_region = _prepare_allreduce_temporary(row_allgather, "rw")
    col_allgather_region = _prepare_allreduce_temporary(col_allgather, "rw")
    cid = T.get_block_binding(0)

    args = (
        buffer_region.region,
        out_region.region,
        row_allgather_region,
        col_allgather_region,
        reduce_type,
        direction_value,
        dim,
        clear,
        cid,
    )

    # If not clearing, allocate an output copy buffer to hold intermediate results
    if not clear:
        out_copy = alloc_tmp(list(out_shape))
        out_copy_region = _prepare_allreduce_temporary(out_copy, "rw")
        args = (
            buffer_region.region,
            out_region.region,
            row_allgather_region,
            col_allgather_region,
            reduce_type,
            DIRECTION_MAP[direction.lower()],
            dim,
            clear,
            out_copy_region,
            cid,
        )

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allreduce"), *args)
