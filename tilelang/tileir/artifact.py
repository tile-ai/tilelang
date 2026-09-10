"""Serializable artifacts produced by the TileIR lowering pipeline."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
from typing import Any, Literal

from tilelang import tvm
from tvm import tirx


TILEIR_CACHE_FORMAT = "tilelang.tileir.artifact"
TILEIR_CACHE_FORMAT_VERSION = 3
TILEIR_CACHE_FILENAME = "kernel.tileir.json"


@dataclass(frozen=True)
class TileIRLaunchMetadata:
    """Native launch metadata for a compiled TileIR entry."""

    grid: tuple[Any, Any, Any] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    dynamic_smem_bytes: int = 0


@dataclass(frozen=True)
class TileIRTemporaryBuffer:
    """Global temporary buffer required between TileIR kernel launches."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TileIRArgumentRef:
    """Stable reference to one original parameter or program temporary."""

    kind: Literal["parameter", "temporary"]
    index: int

    def __post_init__(self) -> None:
        if self.kind not in {"parameter", "temporary"}:
            raise ValueError(f"TileIR argument reference kind must be `parameter` or `temporary`; got {self.kind!r}.")
        if type(self.index) is not int or self.index < 0:
            raise ValueError(f"TileIR argument reference index must be a non-negative integer; got {self.index!r}.")


@dataclass(frozen=True)
class TileIRArtifactCompatibility:
    """External ABI contract required to load a cached TileIR artifact."""

    target_arch: str
    cuda_tile_ir_version: str
    cuda_tile_runtime_version: str
    tileiras_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "target_arch",
            "cuda_tile_ir_version",
            "cuda_tile_runtime_version",
            "tileiras_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"TileIR artifact compatibility field `{field_name}` must be a non-empty string.")


@dataclass(frozen=True)
class TileIRLoweringResult:
    """Compiled TileIR payload ready for the native TileIR runtime boundary.

    ``argument_names`` contains every ordered entry argument, including
    scalars. Once populated at the runtime-adapter boundary,
    ``argument_scalar_flags`` distinguishes scalars from buffers and
    ``argument_refs`` maps each kernel argument to a stable program parameter
    or temporary index. Cache serialization requires both same-length ABI
    descriptors plus root-level external compatibility metadata.
    """

    kernel_name: str
    cubin: bytes
    tileir_source: str | None = None
    launch_metadata: TileIRLaunchMetadata = TileIRLaunchMetadata()
    argument_names: tuple[str, ...] = ()
    argument_scalar_flags: tuple[bool, ...] = ()
    argument_refs: tuple[TileIRArgumentRef, ...] = ()
    temporary_buffers: tuple[TileIRTemporaryBuffer, ...] = ()
    kernels: tuple[TileIRLoweringResult, ...] = ()
    compatibility: TileIRArtifactCompatibility | None = None
    # Nonzero adds one hidden uint8 tensor after the public entry arguments.
    scratch_bytes_per_block: int = 0

    def __post_init__(self) -> None:
        if type(self.scratch_bytes_per_block) is not int or self.scratch_bytes_per_block < 0 or self.scratch_bytes_per_block % 16:
            raise ValueError("TileIR scratch size must be a non-negative multiple of 16 bytes.")
        if self.argument_scalar_flags and len(self.argument_scalar_flags) != len(self.argument_names):
            raise ValueError(
                "TileIR argument metadata must contain one scalar flag per ordered argument: "
                f"got {len(self.argument_scalar_flags)} flags for {len(self.argument_names)} arguments."
            )
        if any(type(flag) is not bool for flag in self.argument_scalar_flags):
            raise TypeError("TileIR argument scalar flags must be booleans.")
        if self.argument_refs and len(self.argument_refs) != len(self.argument_names):
            raise ValueError(
                "TileIR argument metadata must contain one stable reference per ordered argument: "
                f"got {len(self.argument_refs)} references for {len(self.argument_names)} arguments."
            )


def _malformed(message: str) -> ValueError:
    return ValueError(f"Malformed TileIR cache artifact: {message}")


def _encode_launch_extent(extent: Any) -> dict[str, Any]:
    if type(extent) is int:
        return {"kind": "int", "value": extent}
    if isinstance(extent, tirx.PrimExpr):
        return {"kind": "tir", "value": tvm.ir.save_json(extent)}
    raise TypeError(f"Unsupported TileIR launch extent `{extent}` ({type(extent).__name__}).")


def _decode_launch_extent(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise _malformed("launch extent must be an object")
    kind = payload.get("kind")
    value = payload.get("value")
    if kind == "int" and type(value) is int:
        if value < 1:
            raise _malformed("static launch grid extent must be a positive integer")
        return value
    if kind == "tir" and isinstance(value, str):
        try:
            result = tvm.ir.load_json(value)
        except Exception as exc:
            raise _malformed("launch extent contains invalid TVM IR JSON") from exc
        if not isinstance(result, tirx.PrimExpr):
            raise _malformed("launch extent TVM IR is not a PrimExpr")
        return result
    raise _malformed(f"launch extent has unsupported kind `{kind}`")


def _encode_artifact(artifact: TileIRLoweringResult) -> dict[str, Any]:
    if len(artifact.argument_scalar_flags) != len(artifact.argument_names):
        raise ValueError(
            f"TileIR cache serialization requires explicit ordered scalar metadata for every argument in `{artifact.kernel_name}`."
        )
    if len(artifact.argument_refs) != len(artifact.argument_names):
        raise ValueError(f"TileIR cache serialization requires stable references for every argument in `{artifact.kernel_name}`.")
    return {
        "kernel_name": artifact.kernel_name,
        "cubin": base64.b64encode(artifact.cubin).decode("ascii"),
        "tileir_source": artifact.tileir_source,
        "launch_metadata": {
            "grid": [_encode_launch_extent(extent) for extent in artifact.launch_metadata.grid],
            "block": list(artifact.launch_metadata.block),
            "dynamic_smem_bytes": artifact.launch_metadata.dynamic_smem_bytes,
        },
        "argument_names": list(artifact.argument_names),
        "scratch_bytes_per_block": artifact.scratch_bytes_per_block,
        "argument_scalar_flags": list(artifact.argument_scalar_flags),
        "argument_refs": [{"kind": ref.kind, "index": ref.index} for ref in artifact.argument_refs],
        "temporary_buffers": [
            {"name": temporary.name, "shape": list(temporary.shape), "dtype": temporary.dtype} for temporary in artifact.temporary_buffers
        ],
        "kernels": [_encode_artifact(kernel) for kernel in artifact.kernels],
    }


def _require_list(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise _malformed(f"`{field}` must be a list")
    return value


def _decode_launch_metadata(payload: Any) -> TileIRLaunchMetadata:
    if not isinstance(payload, dict):
        raise _malformed("`launch_metadata` must be an object")
    grid = payload.get("grid")
    block = payload.get("block")
    dynamic_smem_bytes = payload.get("dynamic_smem_bytes")
    if not isinstance(grid, list) or len(grid) != 3:
        raise _malformed("launch grid must contain exactly three extents")
    if not isinstance(block, list) or len(block) != 3 or any(type(extent) is not int or extent < 1 for extent in block):
        raise _malformed("launch block must contain exactly three positive integers")
    if type(dynamic_smem_bytes) is not int or dynamic_smem_bytes < 0:
        raise _malformed("`dynamic_smem_bytes` must be a non-negative integer")
    return TileIRLaunchMetadata(
        grid=tuple(_decode_launch_extent(extent) for extent in grid),
        block=tuple(block),
        dynamic_smem_bytes=dynamic_smem_bytes,
    )


def _decode_temporary_buffer(payload: Any) -> TileIRTemporaryBuffer:
    if not isinstance(payload, dict):
        raise _malformed("temporary buffer must be an object")
    name = payload.get("name")
    shape = payload.get("shape")
    dtype = payload.get("dtype")
    if not isinstance(name, str) or not name:
        raise _malformed("temporary buffer `name` must be a non-empty string")
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise _malformed("temporary buffer `shape` must contain non-negative integers")
    if not isinstance(dtype, str) or not dtype:
        raise _malformed("temporary buffer `dtype` must be a non-empty string")
    return TileIRTemporaryBuffer(name=name, shape=tuple(shape), dtype=dtype)


def _decode_argument_ref(payload: Any) -> TileIRArgumentRef:
    if not isinstance(payload, dict):
        raise _malformed("argument reference must be an object")
    try:
        return TileIRArgumentRef(kind=payload.get("kind"), index=payload.get("index"))
    except (TypeError, ValueError) as exc:
        raise _malformed(f"invalid argument reference: {exc}") from exc


def _encode_compatibility(compatibility: TileIRArtifactCompatibility) -> dict[str, str]:
    return {
        "target_arch": compatibility.target_arch,
        "cuda_tile_ir_version": compatibility.cuda_tile_ir_version,
        "cuda_tile_runtime_version": compatibility.cuda_tile_runtime_version,
        "tileiras_version": compatibility.tileiras_version,
    }


def _decode_compatibility(payload: Any) -> TileIRArtifactCompatibility:
    if not isinstance(payload, dict):
        raise _malformed("`compatibility` must be an object")
    try:
        return TileIRArtifactCompatibility(
            target_arch=payload.get("target_arch"),
            cuda_tile_ir_version=payload.get("cuda_tile_ir_version"),
            cuda_tile_runtime_version=payload.get("cuda_tile_runtime_version"),
            tileiras_version=payload.get("tileiras_version"),
        )
    except (TypeError, ValueError) as exc:
        raise _malformed(f"invalid compatibility metadata: {exc}") from exc


def _decode_artifact(
    payload: Any,
    *,
    compatibility: TileIRArtifactCompatibility | None = None,
) -> TileIRLoweringResult:
    if not isinstance(payload, dict):
        raise _malformed("`artifact` must be an object")
    kernel_name = payload.get("kernel_name")
    cubin = payload.get("cubin")
    tileir_source = payload.get("tileir_source")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise _malformed("`kernel_name` must be a non-empty string")
    if not isinstance(cubin, str):
        raise _malformed("`cubin` must be a base64 string")
    try:
        cubin_bytes = base64.b64decode(cubin, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _malformed("`cubin` is not valid base64") from exc
    if tileir_source is not None and not isinstance(tileir_source, str):
        raise _malformed("`tileir_source` must be a string or null")

    argument_names = _require_list(payload, "argument_names")
    scalar_flags = _require_list(payload, "argument_scalar_flags")
    argument_refs = _require_list(payload, "argument_refs")
    if any(not isinstance(name, str) or not name for name in argument_names):
        raise _malformed("`argument_names` must contain non-empty strings")
    if any(type(flag) is not bool for flag in scalar_flags):
        raise _malformed("`argument_scalar_flags` must contain booleans")
    if len(argument_names) != len(scalar_flags):
        raise _malformed("argument names and scalar flags must have equal length")
    if len(argument_names) != len(argument_refs):
        raise _malformed("argument names and stable references must have equal length")

    return TileIRLoweringResult(
        kernel_name=kernel_name,
        cubin=cubin_bytes,
        tileir_source=tileir_source,
        launch_metadata=_decode_launch_metadata(payload.get("launch_metadata")),
        argument_names=tuple(argument_names),
        argument_scalar_flags=tuple(scalar_flags),
        argument_refs=tuple(_decode_argument_ref(ref) for ref in argument_refs),
        temporary_buffers=tuple(_decode_temporary_buffer(temporary) for temporary in _require_list(payload, "temporary_buffers")),
        kernels=tuple(_decode_artifact(kernel) for kernel in _require_list(payload, "kernels")),
        compatibility=compatibility,
        scratch_bytes_per_block=payload.get("scratch_bytes_per_block"),
    )


def serialize_tileir_artifact(artifact: TileIRLoweringResult) -> bytes:
    """Serialize a TileIR runtime artifact as versioned, data-only JSON."""

    if artifact.compatibility is None:
        raise ValueError("TileIR cache serialization requires external compatibility metadata.")

    envelope = {
        "format": TILEIR_CACHE_FORMAT,
        "version": TILEIR_CACHE_FORMAT_VERSION,
        "compatibility": _encode_compatibility(artifact.compatibility),
        "artifact": _encode_artifact(artifact),
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")


def deserialize_tileir_artifact(payload: bytes) -> TileIRLoweringResult:
    """Deserialize and validate a versioned TileIR cache artifact."""

    try:
        envelope = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _malformed("expected UTF-8 JSON") from exc
    if not isinstance(envelope, dict):
        raise _malformed("top-level value must be an object")
    if envelope.get("format") != TILEIR_CACHE_FORMAT:
        raise _malformed(f"unexpected format `{envelope.get('format')}`")
    version = envelope.get("version")
    if type(version) is not int:
        raise _malformed("`version` must be an integer")
    if version != TILEIR_CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported TileIR cache artifact version {version}; expected {TILEIR_CACHE_FORMAT_VERSION}.")
    compatibility = _decode_compatibility(envelope.get("compatibility"))
    return _decode_artifact(envelope.get("artifact"), compatibility=compatibility)
