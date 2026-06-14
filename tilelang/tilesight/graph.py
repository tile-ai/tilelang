"""Structured TileSight graph IR for TileLang kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class ExprInfo:
    """A TIR expression preserved as text plus an optional static value."""

    text: str
    value: int | float | str | bool | None = None

    @classmethod
    def from_value(cls, value: Any) -> "ExprInfo":
        return cls(text=str(value), value=value if isinstance(value, (int, float, str, bool)) else None)


@dataclass
class BufferInfo:
    id: str
    name: str
    scope: str
    dtype: str
    shape: list[ExprInfo]
    strides: list[ExprInfo] = field(default_factory=list)
    bytes_per_element: float | None = None
    static_bytes: float | None = None
    is_param: bool = False


@dataclass
class RegionInfo:
    buffer_id: str | None
    buffer_name: str | None
    scope: str | None
    dtype: str | None
    indices: list[ExprInfo] = field(default_factory=list)
    extents: list[ExprInfo] = field(default_factory=list)
    access: str | None = None
    static_bytes: float | None = None
    signature: str = ""


@dataclass
class LoopInfo:
    id: str
    var: str
    extent: ExprInfo
    kind: str
    annotations: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None

    @property
    def static_extent(self) -> int | None:
        return self.extent.value if isinstance(self.extent.value, int) else None

    @property
    def pipeline_stages(self) -> int:
        for key in ("num_stages", "tl_pipelined_num_stages"):
            value = self.annotations.get(key)
            if isinstance(value, int):
                return max(value, 1)
        return 1

    @property
    def is_pipelined(self) -> bool:
        return self.pipeline_stages > 1 or any(
            key in self.annotations for key in ("tl_pipeline_order", "tl_pipeline_stage", "tl_pipeline_group")
        )


@dataclass
class TileOpNode:
    id: str
    kind: str
    op_name: str
    loop_ids: list[str]
    regions: list[RegionInfo] = field(default_factory=list)
    input_buffers: list[str] = field(default_factory=list)
    output_buffers: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)
    static_bytes: float | None = None
    static_flops: float | None = None
    math_shape: dict[str, ExprInfo] = field(default_factory=dict)


@dataclass
class DataEdge:
    src_op: str
    dst_op: str
    buffer_id: str
    reason: str = "read_after_write"


@dataclass
class KernelNode:
    name: str
    buffers: dict[str, BufferInfo] = field(default_factory=dict)
    loops: dict[str, LoopInfo] = field(default_factory=dict)
    ops: list[TileOpNode] = field(default_factory=list)
    edges: list[DataEdge] = field(default_factory=list)
    grid: dict[str, ExprInfo] = field(default_factory=dict)
    threads: dict[str, ExprInfo] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def static_grid_size(self) -> int | None:
        return _static_product(self.grid.values())

    def static_thread_count(self) -> int | None:
        return _static_product(self.threads.values())

    def shared_footprint_bytes(self) -> float:
        return sum(
            buffer.static_bytes or 0.0
            for buffer in self.buffers.values()
            if buffer.scope.startswith("shared")
        )


@dataclass
class KernelGraph:
    kernels: list[KernelNode] = field(default_factory=list)
    source_stage: str = "after_pipeline_planning_before_software_pipeline"
    schema_version: int = 1

    def has_tile_ops(self) -> bool:
        return any(kernel.ops for kernel in self.kernels)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)


def _static_product(values) -> int | None:
    product = 1
    saw_value = False
    for value in values:
        scalar = value.value if isinstance(value, ExprInfo) else value
        if not isinstance(scalar, int):
            return None
        product *= scalar
        saw_value = True
    return product if saw_value else 1
