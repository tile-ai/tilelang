"""Refactored TileSight cost model.

The public entry is ``estimate_cost(graph, hardware)``. Internally the model is
split into three explicit layers:

1. single operator modeling
2. L2 cache simulation
3. pipeline modeling plus operator fusion
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any

from .arch import HardwareSpec
from .graph import KernelGraph, KernelNode, LoopInfo, TileOpNode


@dataclass
class ResourceEstimate:
    ddr_read_bytes: float = 0.0
    ddr_write_bytes: float = 0.0
    l2_read_bytes: float = 0.0
    l2_write_bytes: float = 0.0
    smem_bytes: float = 0.0
    register_bytes: float = 0.0
    compute_flops: float = 0.0
    tensor_core_flops: float = 0.0
    cuda_core_flops: float = 0.0
    sfu_flops: float = 0.0

    def scaled(self, factor: float) -> "ResourceEstimate":
        return ResourceEstimate(**{key: value * factor for key, value in asdict(self).items()})

    def add(self, other: "ResourceEstimate") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)

    @property
    def ddr_bytes(self) -> float:
        return self.ddr_read_bytes + self.ddr_write_bytes

    @property
    def l2_bytes(self) -> float:
        return self.l2_read_bytes + self.l2_write_bytes


@dataclass
class SingleOpEstimate:
    kernel: str
    op_id: str
    kind: str
    loop_ids: list[str]
    resources_per_call: ResourceEstimate
    loop_multiplier: int = 1
    pipeline_loop_id: str | None = None
    dtype: str | None = None
    bytes_per_element: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def resources_total_per_cta(self) -> ResourceEstimate:
        return self.resources_per_call.scaled(self.loop_multiplier)


@dataclass
class L2CacheSimulation:
    per_kernel_hit_rate: dict[str, float] = field(default_factory=dict)
    per_op_hit_rate: dict[str, float] = field(default_factory=dict)
    total_l2_read_bytes: float = 0.0
    total_ddr_read_bytes: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class PipelineEstimate:
    id: str
    loop_id: str | None
    stage_count: int
    iteration_count: int
    op_ids: list[str]
    per_cta_latency_seconds: float
    resources_per_cta: ResourceEstimate
    dtype: str | None = None
    bytes_per_element: float | None = None


@dataclass
class KernelEstimate:
    kernel: str
    grid_tiles: int
    threads_per_cta: int
    warps_per_cta: int
    tiles_per_sm: int
    waves: float
    latency_seconds: float
    l2_hit_rate: float
    pipelines: list[PipelineEstimate] = field(default_factory=list)
    fused_resources_per_cta: ResourceEstimate = field(default_factory=ResourceEstimate)


@dataclass
class CostModelResult:
    single_ops: list[SingleOpEstimate]
    l2_cache: L2CacheSimulation
    kernels: list[KernelEstimate]
    total_latency_seconds: float
    hardware: HardwareSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)


def estimate_cost(graph: KernelGraph, hardware: HardwareSpec) -> CostModelResult:
    single_ops = model_single_operators(graph, hardware)
    l2_cache = simulate_l2_cache(graph, single_ops, hardware)
    kernels = model_pipeline_and_fusion(graph, single_ops, l2_cache, hardware)
    total_latency = sum(kernel.latency_seconds for kernel in kernels)
    return CostModelResult(
        single_ops=single_ops,
        l2_cache=l2_cache,
        kernels=kernels,
        total_latency_seconds=total_latency,
        hardware=hardware,
    )


# ---------------------------------------------------------------------------
# 1. Single operator modeling
# ---------------------------------------------------------------------------


def model_single_operators(graph: KernelGraph, hardware: HardwareSpec) -> list[SingleOpEstimate]:
    estimates: list[SingleOpEstimate] = []
    for kernel in graph.kernels:
        pipeline_loops = {loop_id for loop_id, loop in kernel.loops.items() if loop.is_pipelined}
        for op in kernel.ops:
            pipeline_loop_id = next((loop_id for loop_id in reversed(op.loop_ids) if loop_id in pipeline_loops), None)
            loop_multiplier = _loop_multiplier(kernel, op, stop_at_loop=pipeline_loop_id)
            resources, dtype, bytes_per_element, notes = _model_op_resources(kernel, op)
            estimates.append(
                SingleOpEstimate(
                    kernel=kernel.name,
                    op_id=op.id,
                    kind=op.kind,
                    loop_ids=list(op.loop_ids),
                    resources_per_call=resources,
                    loop_multiplier=loop_multiplier,
                    pipeline_loop_id=pipeline_loop_id,
                    dtype=dtype,
                    bytes_per_element=bytes_per_element,
                    notes=notes,
                )
            )
    return estimates


def _model_op_resources(kernel: KernelNode, op: TileOpNode) -> tuple[ResourceEstimate, str | None, float | None, list[str]]:
    resources = ResourceEstimate()
    notes: list[str] = []
    dtype = None
    bytes_per_element = None

    for region in op.regions:
        if dtype is None:
            dtype = region.dtype
        if bytes_per_element is None and region.dtype:
            bytes_per_element = _buffer_dtype_bytes(kernel, region.buffer_id)
        amount = region.static_bytes or 0.0
        if amount == 0:
            continue
        scope = region.scope or "global"
        is_read = region.access in ("r", "rw", None)
        is_write = region.access in ("w", "rw")

        if scope == "global":
            if is_read:
                resources.l2_read_bytes += amount
            if is_write:
                resources.l2_write_bytes += amount
                resources.ddr_write_bytes += amount
        elif scope.startswith("shared"):
            if is_read:
                resources.smem_bytes += amount
            if is_write:
                resources.smem_bytes += amount
        elif scope.startswith("local.fragment"):
            if is_read or is_write:
                resources.register_bytes += amount
        else:
            if is_read or is_write:
                resources.register_bytes += amount

    if op.kind == "gemm":
        flops = op.static_flops or 0.0
        resources.compute_flops += flops
        resources.tensor_core_flops += flops
        if flops == 0:
            notes.append("non-static GEMM shape; compute flops left as 0")
    elif op.kind == "sfu":
        flops = _op_compute_flops(op, bytes_per_element)
        resources.compute_flops += flops
        resources.sfu_flops += flops
    elif op.kind in ("elementwise", "reduce", "scan", "atomic"):
        flops = _op_compute_flops(op, bytes_per_element)
        resources.compute_flops += flops
        resources.cuda_core_flops += flops

    return resources, dtype, bytes_per_element, notes


# ---------------------------------------------------------------------------
# 2. L2 cache simulation
# ---------------------------------------------------------------------------


def simulate_l2_cache(
    graph: KernelGraph,
    single_ops: list[SingleOpEstimate],
    hardware: HardwareSpec,
) -> L2CacheSimulation:
    by_op = {estimate.op_id: estimate for estimate in single_ops}
    result = L2CacheSimulation()

    for kernel in graph.kernels:
        grid_tiles = kernel.static_grid_size() or 1
        kernel_l2_reads = 0.0
        kernel_ddr_reads = 0.0
        for op in kernel.ops:
            estimate = by_op.get(op.id)
            if estimate is None:
                continue
            read_bytes = estimate.resources_total_per_cta.l2_read_bytes
            if read_bytes <= 0:
                result.per_op_hit_rate[op.id] = 0.0
                continue

            hit_rate = _estimate_op_l2_hit_rate(kernel, op, estimate, hardware)
            ddr_read = read_bytes * (1.0 - hit_rate)
            estimate.resources_per_call.ddr_read_bytes = estimate.resources_per_call.l2_read_bytes * (1.0 - hit_rate)
            result.per_op_hit_rate[op.id] = hit_rate
            kernel_l2_reads += read_bytes * grid_tiles
            kernel_ddr_reads += ddr_read * grid_tiles

        kernel_hit = 0.0 if kernel_l2_reads <= 0 else max(0.0, min(1.0, 1.0 - kernel_ddr_reads / kernel_l2_reads))
        result.per_kernel_hit_rate[kernel.name] = kernel_hit
        result.total_l2_read_bytes += kernel_l2_reads
        result.total_ddr_read_bytes += kernel_ddr_reads

    if result.total_l2_read_bytes == 0:
        result.notes.append("no static global reads were found")
    return result


def _estimate_op_l2_hit_rate(
    kernel: KernelNode,
    op: TileOpNode,
    estimate: SingleOpEstimate,
    hardware: HardwareSpec,
) -> float:
    # GEMM-like global->shared copies can be reused across one CTA grid axis.
    if op.kind == "copy":
        source = _global_read_region(op)
        if source is not None:
            reuse = _grid_reuse_factor(kernel, source.signature)
            if reuse > 1:
                raw_hit = 1.0 - 1.0 / reuse
                active_working_set = (source.static_bytes or 0.0) * min(kernel.static_grid_size() or 1, hardware.sm_count)
                capacity_factor = 1.0 if active_working_set <= 0 else min(1.0, hardware.l2_capacity / max(active_working_set, 1.0))
                return max(0.0, min(0.95, raw_hit * capacity_factor))

    # Fallback: symbolic duplicate regions within the same CTA are treated as L2 hits
    # if their footprint fits in cache. This is intentionally conservative.
    same_signature_reads = 0
    for other in kernel.ops:
        for region in other.regions:
            if region.scope == "global" and region.access in ("r", "rw", None) and region.signature:
                same_signature_reads += int(region.signature in {r.signature for r in op.regions})
    if same_signature_reads > 1:
        per_cta = estimate.resources_total_per_cta.l2_read_bytes
        if per_cta <= hardware.l2_capacity:
            return min(0.5, 1.0 - 1.0 / same_signature_reads)
    return 0.0


# ---------------------------------------------------------------------------
# 3. Pipeline modeling and operator fusion
# ---------------------------------------------------------------------------


def model_pipeline_and_fusion(
    graph: KernelGraph,
    single_ops: list[SingleOpEstimate],
    l2_cache: L2CacheSimulation,
    hardware: HardwareSpec,
) -> list[KernelEstimate]:
    by_kernel: dict[str, list[SingleOpEstimate]] = {}
    for estimate in single_ops:
        by_kernel.setdefault(estimate.kernel, []).append(estimate)

    kernel_results: list[KernelEstimate] = []
    for kernel in graph.kernels:
        estimates = by_kernel.get(kernel.name, [])
        pipelines = _build_pipeline_estimates(kernel, estimates, hardware)
        fused = ResourceEstimate()
        for pipeline in pipelines:
            fused.add(pipeline.resources_per_cta)

        grid_tiles = kernel.static_grid_size() or 1
        threads = kernel.static_thread_count() or 1
        warps = max(math.ceil(threads / 32), 1)
        tiles_per_sm = _compute_occupancy(kernel, fused, warps, hardware)
        tiles_per_wave = max(hardware.sm_count * tiles_per_sm, 1)
        waves = grid_tiles / tiles_per_wave
        latency = _wave_adjusted_latency(pipelines, grid_tiles, tiles_per_sm, hardware)

        kernel_results.append(
            KernelEstimate(
                kernel=kernel.name,
                grid_tiles=grid_tiles,
                threads_per_cta=threads,
                warps_per_cta=warps,
                tiles_per_sm=tiles_per_sm,
                waves=waves,
                latency_seconds=latency,
                l2_hit_rate=l2_cache.per_kernel_hit_rate.get(kernel.name, 0.0),
                pipelines=pipelines,
                fused_resources_per_cta=fused,
            )
        )
    return kernel_results


def _build_pipeline_estimates(
    kernel: KernelNode,
    estimates: list[SingleOpEstimate],
    hardware: HardwareSpec,
) -> list[PipelineEstimate]:
    groups: dict[str | None, list[SingleOpEstimate]] = {}
    for estimate in estimates:
        groups.setdefault(estimate.pipeline_loop_id, []).append(estimate)

    pipelines: list[PipelineEstimate] = []
    for group_id, group in groups.items():
        resources = ResourceEstimate()
        for estimate in group:
            resources.add(estimate.resources_total_per_cta)

        loop: LoopInfo | None = kernel.loops.get(group_id) if group_id else None
        stage_count = loop.pipeline_stages if loop else 1
        iteration_count = loop.static_extent if loop and loop.static_extent else 1
        dtype = next((estimate.dtype for estimate in group if estimate.dtype), None)
        bytes_per_element = next((estimate.bytes_per_element for estimate in group if estimate.bytes_per_element), None)
        latency = _pipeline_latency(resources, stage_count, iteration_count, group, hardware, active_sms=hardware.sm_count)
        pipelines.append(
            PipelineEstimate(
                id=f"{kernel.name}:{group_id or 'body'}",
                loop_id=group_id,
                stage_count=stage_count,
                iteration_count=iteration_count,
                op_ids=[estimate.op_id for estimate in group],
                per_cta_latency_seconds=latency,
                resources_per_cta=resources,
                dtype=dtype,
                bytes_per_element=bytes_per_element,
            )
        )
    return pipelines


def _pipeline_latency(
    resources: ResourceEstimate,
    stage_count: int,
    iteration_count: int,
    estimates: list[SingleOpEstimate],
    hardware: HardwareSpec,
    active_sms: int,
    dtype_override: str | None = None,
    bytes_per_element_override: float | None = None,
) -> float:
    bytes_per_element = bytes_per_element_override or next((estimate.bytes_per_element for estimate in estimates if estimate.bytes_per_element), 2.0)
    dtype = dtype_override or next((estimate.dtype for estimate in estimates if estimate.dtype), None)
    active_sms = max(min(active_sms, hardware.sm_count), 1)
    mem_time = max(
        # DDR/L2 are chip-wide shared resources; a CTA observes contention from
        # the number of active SMs in the current wave.
        _safe_div(resources.ddr_bytes * active_sms, hardware.ddr_bandwidth * hardware.ddr_max_util),
        _safe_div(resources.l2_bytes * active_sms, hardware.l2_bandwidth * hardware.l2_max_util),
        # SMEM and compute throughput are quoted for the whole chip, so convert
        # to a per-SM rate for a single CTA resident on one SM.
        _safe_div(resources.smem_bytes * hardware.sm_count, hardware.smem_bandwidth * hardware.smem_max_util),
    )
    tensor_time = _safe_div(
        resources.tensor_core_flops * hardware.sm_count,
        hardware.tensor_flops_for_dtype(dtype, bytes_per_element) * hardware.compute_max_util,
    )
    cuda_time = _safe_div(
        resources.cuda_core_flops * hardware.sm_count,
        hardware.cuda_flops_for_dtype(dtype, bytes_per_element) * hardware.compute_max_util,
    )
    sfu_time = _safe_div(
        resources.sfu_flops * hardware.sm_count,
        hardware.sfu_flops * hardware.compute_max_util,
    )
    compute_time = max(tensor_time, cuda_time, sfu_time)

    if stage_count <= 1:
        return mem_time + compute_time

    # The resources entering this function are already multiplied by the loop
    # iteration count, so divide back to model prologue/steady/epilogue.
    per_iter_mem = mem_time / max(iteration_count, 1)
    per_iter_compute = compute_time / max(iteration_count, 1)
    depth = max(stage_count - 1, 1)
    steady_iters = max(iteration_count - depth, 0)
    return depth * per_iter_mem + steady_iters * max(per_iter_mem, per_iter_compute) + depth * per_iter_compute


def _wave_adjusted_latency(
    pipelines: list[PipelineEstimate],
    grid_tiles: int,
    tiles_per_sm: int,
    hardware: HardwareSpec,
) -> float:
    sm_count = max(hardware.sm_count, 1)
    tiles_per_sm = max(tiles_per_sm, 1)
    tiles_per_wave = sm_count * tiles_per_sm
    full_waves = grid_tiles // tiles_per_wave
    tail_tiles = grid_tiles % tiles_per_wave

    def wave_latency(active_sms: int, ctas_per_sm: int) -> float:
        active_sms = max(min(active_sms, sm_count), 1)
        ctas_per_sm = max(ctas_per_sm, 1)
        per_cta = 0.0
        for pipeline in pipelines:
            per_cta += _pipeline_latency(
                pipeline.resources_per_cta,
                pipeline.stage_count,
                pipeline.iteration_count,
                [],
                hardware,
                active_sms=active_sms,
                dtype_override=pipeline.dtype,
                bytes_per_element_override=pipeline.bytes_per_element,
            )
        return per_cta * ctas_per_sm

    total = full_waves * wave_latency(sm_count, tiles_per_sm)
    if tail_tiles > 0:
        if tail_tiles <= sm_count:
            total += wave_latency(tail_tiles, 1)
        else:
            total += wave_latency(sm_count, min(math.ceil(tail_tiles / sm_count), tiles_per_sm))
    return total


def _compute_occupancy(kernel: KernelNode, resources: ResourceEstimate, warps: int, hardware: HardwareSpec) -> int:
    smem_footprint = kernel.shared_footprint_bytes()
    smem_limit = int(hardware.configurable_smem_capacity / smem_footprint) if smem_footprint > 0 else hardware.max_blocks_per_sm
    reg_footprint = max(resources.register_bytes, 1.0)
    reg_per_block = reg_footprint * max(warps, 1)
    reg_limit = int(hardware.register_capacity_per_sm / reg_per_block) if reg_per_block > 0 else hardware.max_blocks_per_sm
    return max(1, min(smem_limit, reg_limit, hardware.max_blocks_per_sm))


def _loop_multiplier(kernel: KernelNode, op: TileOpNode, stop_at_loop: str | None) -> int:
    multiplier = 1
    for loop_id in op.loop_ids:
        if loop_id == stop_at_loop:
            loop = kernel.loops.get(loop_id)
            if loop and loop.static_extent:
                multiplier *= loop.static_extent
            break
        loop = kernel.loops.get(loop_id)
        if loop is None:
            continue
        if loop.static_extent is None:
            continue
        multiplier *= loop.static_extent
    return multiplier


def _buffer_dtype_bytes(kernel: KernelNode, buffer_id: str | None) -> float | None:
    if not buffer_id:
        return None
    buffer = kernel.buffers.get(buffer_id)
    return buffer.bytes_per_element if buffer else None


def _op_compute_flops(op: TileOpNode, bytes_per_element: float | None) -> float:
    if op.static_flops is not None:
        return float(op.static_flops)
    return float(op.static_bytes or 0.0) / max(bytes_per_element or 1.0, 1.0)


def _global_read_region(op: TileOpNode):
    for region in op.regions:
        if region.scope == "global" and region.access in ("r", "rw", None):
            return region
    return None


def _grid_reuse_factor(kernel: KernelNode, signature: str) -> int:
    grid_x = kernel.grid.get("blockIdx.x")
    grid_y = kernel.grid.get("blockIdx.y")
    x = grid_x.value if grid_x and isinstance(grid_x.value, int) else 1
    y = grid_y.value if grid_y and isinstance(grid_y.value, int) else 1
    uses_x = "bx" in signature or "blockIdx.x" in signature
    uses_y = "by" in signature or "blockIdx.y" in signature
    if uses_y and not uses_x:
        return max(x, 1)
    if uses_x and not uses_y:
        return max(y, 1)
    return 1


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0
