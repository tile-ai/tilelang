"""
Pipeline Planning Pass for TileLang.

This module implements the pipeline planning pass that analyzes loop bodies,
detects global copy patterns, and generates software pipeline annotations
for efficient memory copy pipelining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from tilelang import tvm as tvm
from tvm import arith, ir, tir
from tvm.ir import Range
from tvm.tir import (
    AttrStmt,
    Block,
    BlockRealize,
    Buffer,
    BufferLoad,
    BufferRegion,
    BufferStore,
    Call,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    LetStmt,
    PrimExpr,
    PrimFunc,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.analysis import get_block_read_write_region
from tvm.tir.stmt_functor import ir_transform, post_order_visit
from tvm.tir.transform import prim_func_pass

# Get builtin operations using Op.get
_OP_CALL_EXTERN = tir.op.Op.get("tir.call_extern")
_OP_TVM_ACCESS_PTR = tir.op.Op.get("tir.tvm_access_ptr")
_OP_ADDRESS_OF = tir.op.Op.get("tir.address_of")
_OP_IF_THEN_ELSE = tir.op.Op.get("tir.if_then_else")


def _get_buffer_scope(buf: Buffer) -> str:
    """Get the storage scope of a buffer."""
    ptr_type = buf.data.type_annotation
    if hasattr(ptr_type, "storage_scope"):
        return ptr_type.storage_scope
    return ""


def _full_buffer_region(buf: Buffer) -> BufferRegion:
    """Create a BufferRegion covering the entire buffer."""
    region = [Range.from_min_extent(0, s) for s in buf.shape]
    return BufferRegion(buf, region)


def _may_conflict(region1: List[Range], region2: List[Range]) -> bool:
    """
    Check whether two regions have intersections.

    Parameters
    ----------
    region1 : List[Range]
        The first region.
    region2 : List[Range]
        The second region.

    Returns
    -------
    bool
        Whether region1 and region2 have intersections.
    """
    if len(region1) != len(region2):
        return True

    analyzer = arith.Analyzer()

    for dim1, dim2 in zip(region1, region2):
        # Check if the ranges are disjoint
        # Range is [min, min+extent)
        # Two ranges [a, a+e1) and [b, b+e2) are disjoint if:
        # a + e1 <= b OR b + e2 <= a
        end1 = dim1.min + dim1.extent
        end2 = dim2.min + dim2.extent

        # If we can prove they don't overlap, return False
        if analyzer.can_prove(end1 <= dim2.min) or analyzer.can_prove(end2 <= dim1.min):
            return False

    return True


class DependencyType(Enum):
    """Type of data dependency between pipeline stages."""

    RAW = "read_after_write"  # True dependency: B reads what A writes
    WAR = "write_after_read"  # Anti-dependency: B writes what A reads
    WAW = "write_after_write"  # Output dependency: B writes what A also writes


@dataclass
class StageDependency:
    """
    Represents a dependency edge in the pipeline DAG.

    Attributes
    ----------
    from_stage : int
        The original_stmt_index of the source stage (producer)
    to_stage : int
        The original_stmt_index of the dependent stage (consumer)
    dep_type : DependencyType
        The type of dependency (RAW, WAR, or WAW)
    buffer_name : str
        Name of the buffer causing this dependency
    """

    from_stage: int
    to_stage: int
    dep_type: DependencyType
    buffer_name: str

    def __repr__(self) -> str:
        return f"S{self.from_stage}->{self.to_stage}({self.dep_type.name}:{self.buffer_name})"


class AsyncDependencyChainBuilder:
    """
    Build the dependency chain between async operations and their
    corresponding buffers & synchronizations.

    Example:
        If we encounter the following pattern:

        tcgen5mma_gemm_ts(..., mbar, ...)
        mbarrier_wait_parity(mbar)

        The builder will link the mbarrier to the buffers used in the TCGEN5MMA
    """

    def __init__(self, buffer_data_to_buffer: Dict[Var, Buffer]):
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.mbar_to_buffer_reads: Dict[Buffer, List[BufferRegion]] = {}
        self.mbar_to_buffer_writes: Dict[Buffer, List[BufferRegion]] = {}

    def __call__(self, stmt: Stmt) -> None:
        def visit(node):
            if isinstance(node, Call):
                self._visit_call(node)

        post_order_visit(stmt, visit)

    def _get_buf_from_access_ptr_call(self, expr: PrimExpr) -> Optional[Buffer]:
        """Extract buffer from a tvm_access_ptr call."""
        if not isinstance(expr, Call):
            return None
        if not expr.op.same_as(_OP_TVM_ACCESS_PTR):
            return None

        if len(expr.args) < 2:
            return None

        var = expr.args[1]
        if not isinstance(var, Var):
            return None

        return self.buffer_data_to_buffer.get(var)

    def _visit_call(self, op: Call) -> None:
        args = op.args
        if op.op.same_as(_OP_IF_THEN_ELSE):
            # Recursively visit then and else expressions
            if len(args) >= 3: # condition, then, else are defined in the if_then_else node
                post_order_visit(args[1], lambda n: self._visit_call(n) if isinstance(n, Call) else None)
                post_order_visit(args[2], lambda n: self._visit_call(n) if isinstance(n, Call) else None)

class BufferRegionCollector:
    """
    Detect if a statement follows the global memory copy pattern:
        1. Contains exactly one buffer store operation
        2. Source buffer must be in global memory scope
        3. Destination buffer must be in local or shared memory scope
    """

    def __init__(
        self,
        buffer_data_to_buffer: Dict[Var, Buffer],
        chain_builder: AsyncDependencyChainBuilder,
    ):
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.chain_builder = chain_builder
        self.reads: List[BufferRegion] = []
        self.writes: List[BufferRegion] = []
        self.is_global_copy_pattern = False
        self._is_global_read = False
        self._within_condition_expr = False

    def __call__(self, stmt: Stmt) -> None:
        self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, BufferStore):
            self._visit_buffer_store(stmt)
        elif isinstance(stmt, IfThenElse):
            self._visit_if_then_else(stmt)
        elif isinstance(stmt, SeqStmt):
            for s in stmt.seq:
                self._visit_stmt(s)
        elif isinstance(stmt, For):
            self._visit_stmt(stmt.body)
        elif isinstance(stmt, Block):
            self._visit_stmt(stmt.body)
        elif isinstance(stmt, BlockRealize):
            self._visit_stmt(stmt.block.body)
        elif isinstance(stmt, LetStmt):
            self._visit_stmt(stmt.body)
        elif isinstance(stmt, tir.Evaluate):
            # Handle Evaluate statements containing Call expressions
            self._visit_expr(stmt.value)
        elif hasattr(stmt, "body"):
            self._visit_stmt(stmt.body)
        # Visit expressions in Evaluate statements (fallback)
        elif hasattr(stmt, "value"):
            self._visit_expr(stmt.value)

    def _visit_buffer_store(self, op: BufferStore) -> None:
        store_buffer = op.buffer
        indices = op.indices

        # Convert indices to region
        region = [Range.from_min_extent(idx, 1) for idx in indices]
        store_region = BufferRegion(store_buffer, region)
        self.writes.append(store_region)

        self._is_global_read = False
        self._visit_expr(op.value)

        if self._is_global_read:
            scope = _get_buffer_scope(store_buffer)
            if scope in ("shared", "shared.dyn"):
                self.is_global_copy_pattern = True

        self._is_global_read = False

    def _visit_expr(self, expr: PrimExpr) -> None:
        def visit(node):
            if isinstance(node, BufferLoad):
                self._visit_buffer_load(node)
            elif isinstance(node, Call):
                self._visit_call(node)

        post_order_visit(expr, visit)

    def _visit_buffer_load(self, op: BufferLoad) -> None:
        load_buffer = op.buffer
        indices = op.indices

        # Convert indices to region
        region = [Range.from_min_extent(idx, 1) for idx in indices]
        load_region = BufferRegion(load_buffer, region)
        self.reads.append(load_region)

        scope = _get_buffer_scope(load_buffer)
        if scope == "global" and not self._within_condition_expr:
            self._is_global_read = True

    def _visit_call(self, op: Call) -> None:
        args = op.args
        op_name = str(op.op)

        if op.op.same_as(_OP_ADDRESS_OF):
            buffer_region = None
            if len(args) > 0:
                if isinstance(args[0], BufferLoad):
                    buffer_region = _full_buffer_region(args[0].buffer)
                elif isinstance(args[0], Var):
                    buf = self.buffer_data_to_buffer.get(args[0])
                    if buf is not None:
                        buffer_region = _full_buffer_region(buf)

            if buffer_region is not None:
                self.reads.append(buffer_region)

        elif op.op.same_as(_OP_TVM_ACCESS_PTR):
            if len(args) >= 2 and isinstance(args[1], Var):
                buf = self.buffer_data_to_buffer.get(args[1])
                if buf is not None:
                    buffer_region = _full_buffer_region(buf)
                    self.reads.append(buffer_region)

        elif op.op.same_as(_OP_IF_THEN_ELSE):
            # Skip condition expr for global read detection
            self._within_condition_expr = True
            if len(args) >= 1:
                self._visit_expr(args[0])
            self._within_condition_expr = False
            for i in range(1, len(args)):
                self._visit_expr(args[i])

        elif "tl.mbarrier_wait_parity" in op_name:
            if len(args) > 0 and isinstance(args[0], BufferLoad):
                mbar_buf = args[0].buffer
                buffer_reads = self.chain_builder.mbar_to_buffer_reads.get(mbar_buf, [])
                buffer_writes = self.chain_builder.mbar_to_buffer_writes.get(
                    mbar_buf, []
                )
                self.reads.extend(buffer_reads)
                self.writes.extend(buffer_writes)

        elif "tl.tileop.region" in op_name:
            # tl.region(buffer_load, access_type, *extents)
            # access_type: 1=read, 2=write, 3=rw
            if len(args) >= 2:
                buffer_load = args[0]
                access_type = args[1]
                if isinstance(buffer_load, BufferLoad):
                    region = _full_buffer_region(buffer_load.buffer)
                    access_val = access_type.value if isinstance(access_type, IntImm) else int(access_type)
                    if access_val & 1:  # read
                        self.reads.append(region)
                        scope = _get_buffer_scope(buffer_load.buffer)
                        if scope == "global" and not self._within_condition_expr:
                            self._is_global_read = True
                    if access_val & 2:  # write
                        self.writes.append(region)
                        scope = _get_buffer_scope(buffer_load.buffer)
                        if scope in ("shared", "shared.dyn"):
                            self.is_global_copy_pattern = self._is_global_read

        elif "tl.tileop.copy" in op_name:
            # tl.copy(src_region, dst_region, ...)
            # Recursively visit to extract regions
            for arg in args[:2]:  # First two args are src and dst regions
                if isinstance(arg, Call):
                    self._visit_call(arg)

        elif "tl.gemm" in op_name or "tl.tileop.gemm" in op_name:
            # tl.gemm(A_region, B_region, C_region, ...)
            # A, B are read; C is read-write (accumulate)
            for arg in args[:3]:  # First three args are A, B, C regions
                if isinstance(arg, Call):
                    self._visit_call(arg)

    def _visit_if_then_else(self, op: IfThenElse) -> None:
        self._within_condition_expr = True
        self._visit_expr(op.condition)
        self._within_condition_expr = False

        self._visit_stmt(op.then_case)
        if op.else_case is not None:
            self._within_condition_expr = True
            self._visit_stmt(op.else_case)
            self._within_condition_expr = False


@dataclass
class PipelineStageInfo:
    """
    Information about a pipeline stage.

    Attributes
    ----------
    reads : List[BufferRegion]
        Array of buffer regions read by this stage
    writes : List[BufferRegion]
        Array of buffer regions written by this stage
    original_stmt_index : int
        Original position of this stage in the pipeline before reordering
    order : int
        Current position of this stage in the pipeline after reordering (-1 if not yet assigned)
    stage : int
        Pipeline stage number this operation belongs to (-1 if not yet assigned)
    copy_stage : bool
        Whether this stage is a memory copy operation
    producer_for_copy : bool
        Whether this stage produces data for a copy stage
    last_use_stmt_index : int
        Index of the last statement (in original order) that uses the results of this stage
    predecessors : Set[int]
        Set of stage indices that this stage depends on (incoming edges in DAG)
    successors : Set[int]
        Set of stage indices that depend on this stage (outgoing edges in DAG)
    dependencies : List[StageDependency]
        Detailed dependency edges from predecessors to this stage
    """

    reads: List[BufferRegion] = field(default_factory=list)
    writes: List[BufferRegion] = field(default_factory=list)
    original_stmt_index: int = 0
    order: int = -1
    stage: int = -1
    copy_stage: bool = False
    producer_for_copy: bool = False
    last_use_stmt_index: int = -1
    # DAG fields
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    dependencies: List[StageDependency] = field(default_factory=list)

    def is_first_stage(self) -> bool:
        return self.copy_stage or self.producer_for_copy

    def is_copy_stage(self) -> bool:
        return self.copy_stage

    def is_producer_for_copy(self) -> bool:
        return self.producer_for_copy

    def is_last_use_stmt_index_valid(self) -> bool:
        return self.last_use_stmt_index != -1

    def get_label(self) -> str:
        """Get a human-readable label for this stage."""
        label = f"S{self.original_stmt_index}"
        if self.copy_stage:
            label += " [copy]"
        elif self.producer_for_copy:
            label += " [prod]"
        return label


class CopyStageDependencyReadsManager:
    """Helper class to manage copy stage dependency reads."""

    def __init__(self):
        self.regions: List[BufferRegion] = []

    def add_unique(self, region: BufferRegion) -> bool:
        """Add a region if not already present (by buffer identity)."""
        for copy_read in self.regions:
            if region.buffer.same_as(copy_read.buffer):
                return False
        self.regions.append(region)
        return True

    def contains(self, region: BufferRegion) -> bool:
        """Check if a region is present (by buffer identity)."""
        for copy_read in self.regions:
            if region.buffer.same_as(copy_read.buffer):
                return True
        return False

    def size(self) -> int:
        return len(self.regions)


def build_dependency_dag(stage_infos: List[PipelineStageInfo]) -> None:
    """
    Build the dependency DAG by analyzing reads/writes between stages.

    This function populates the predecessors, successors, and dependencies
    fields of each PipelineStageInfo based on data dependencies:
    - RAW (Read After Write): stage j reads what stage i writes
    - WAR (Write After Read): stage j writes what stage i reads
    - WAW (Write After Write): stage j writes what stage i also writes

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects to analyze. The DAG fields
        will be populated in-place.
    """
    n = len(stage_infos)

    for i in range(n):
        for j in range(i + 1, n):
            stage_i = stage_infos[i]
            stage_j = stage_infos[j]

            # Check RAW: j reads what i writes
            for write_i in stage_i.writes:
                for read_j in stage_j.reads:
                    if write_i.buffer.same_as(read_j.buffer):
                        if _may_conflict(list(write_i.region), list(read_j.region)):
                            stage_i.successors.add(j)
                            stage_j.predecessors.add(i)
                            stage_j.dependencies.append(
                                StageDependency(
                                    i, j, DependencyType.RAW, write_i.buffer.name
                                )
                            )

            # Check WAW: j writes what i also writes
            for write_i in stage_i.writes:
                for write_j in stage_j.writes:
                    if write_i.buffer.same_as(write_j.buffer):
                        if _may_conflict(list(write_i.region), list(write_j.region)):
                            # Only add if not already connected
                            if j not in stage_i.successors:
                                stage_i.successors.add(j)
                                stage_j.predecessors.add(i)
                            stage_j.dependencies.append(
                                StageDependency(
                                    i, j, DependencyType.WAW, write_i.buffer.name
                                )
                            )

            # Check WAR: j writes what i reads
            for read_i in stage_i.reads:
                for write_j in stage_j.writes:
                    if read_i.buffer.same_as(write_j.buffer):
                        if _may_conflict(list(read_i.region), list(write_j.region)):
                            # Only add if not already connected
                            if j not in stage_i.successors:
                                stage_i.successors.add(j)
                                stage_j.predecessors.add(i)
                            stage_j.dependencies.append(
                                StageDependency(
                                    i, j, DependencyType.WAR, read_i.buffer.name
                                )
                            )


def dag_to_dot(stage_infos: List[PipelineStageInfo], title: str = "Pipeline DAG") -> str:
    """
    Generate DOT format string for the dependency DAG.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.
    title : str
        Title for the graph.

    Returns
    -------
    str
        DOT format string that can be rendered by graphviz.

    Example
    -------
    >>> print(dag_to_dot(stage_infos))
    digraph PipelineDAG {
      rankdir=TB;
      label="Pipeline DAG";
      0 [label="S0 [copy]" style=filled fillcolor=lightblue];
      1 [label="S1 [copy]" style=filled fillcolor=lightblue];
      2 [label="S2"];
      0 -> 2 [label="RAW"];
      1 -> 2 [label="RAW"];
    }
    """
    lines = [
        "digraph PipelineDAG {",
        "  rankdir=TB;",
        f'  label="{title}";',
        "  labelloc=t;",
        "  node [shape=box];",
    ]

    # Add nodes
    for info in stage_infos:
        idx = info.original_stmt_index
        label = info.get_label()

        # Style based on stage type
        if info.copy_stage:
            style = 'style=filled fillcolor=lightblue'
        elif info.producer_for_copy:
            style = 'style=filled fillcolor=lightyellow'
        else:
            style = 'style=filled fillcolor=white'

        lines.append(f'  {idx} [label="{label}" {style}];')

    # Add edges with dependency type labels
    added_edges: Set[Tuple[int, int]] = set()
    for info in stage_infos:
        for dep in info.dependencies:
            edge_key = (dep.from_stage, dep.to_stage)
            if edge_key not in added_edges:
                # Collect all dependency types for this edge
                dep_types = [
                    d.dep_type.name
                    for d in info.dependencies
                    if d.from_stage == dep.from_stage and d.to_stage == dep.to_stage
                ]
                label = ",".join(sorted(set(dep_types)))
                lines.append(f'  {dep.from_stage} -> {dep.to_stage} [label="{label}"];')
                added_edges.add(edge_key)

    lines.append("}")
    return "\n".join(lines)


def dag_to_ascii(stage_infos: List[PipelineStageInfo], style: str = "vertical") -> str:
    """
    Generate ASCII representation of the dependency DAG.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.
    style : str
        "vertical" for top-down graph, "list" for simple list format.

    Returns
    -------
    str
        ASCII art representation of the DAG.
    """
    if style == "list":
        return _dag_to_ascii_list(stage_infos)
    return _dag_to_ascii_vertical(stage_infos)


def _dag_to_ascii_list(stage_infos: List[PipelineStageInfo]) -> str:
    """Simple list-style ASCII output."""
    lines = ["Pipeline Dependency DAG", "=" * 40]

    for info in stage_infos:
        label = info.get_label()
        pred_str = ""
        if info.predecessors:
            pred_list = sorted(info.predecessors)
            pred_str = f"  <- [S{', S'.join(map(str, pred_list))}]"
        lines.append(f"  {label}{pred_str}")

        for succ_idx in sorted(info.successors):
            succ_info = stage_infos[succ_idx]
            deps_to_succ = [
                d for d in succ_info.dependencies
                if d.from_stage == info.original_stmt_index
            ]
            if deps_to_succ:
                # Deduplicate dependencies
                seen = set()
                unique_deps = []
                for d in deps_to_succ:
                    key = (d.dep_type.name, d.buffer_name)
                    if key not in seen:
                        seen.add(key)
                        unique_deps.append(f"{d.dep_type.name}:{d.buffer_name}")
                dep_details = ", ".join(unique_deps)
                lines.append(f"    └─> S{succ_idx} ({dep_details})")
            else:
                lines.append(f"    └─> S{succ_idx}")

    return "\n".join(lines)


def _dag_to_ascii_vertical(stage_infos: List[PipelineStageInfo]) -> str:
    """
    Generate a vertical top-down ASCII DAG visualization.

    Example output:
    ┌──────────────────────────────────────┐
    │       Pipeline Dependency DAG        │
    └──────────────────────────────────────┘

        ┌──────────┐          ┌──────────┐
        │ S0 copy  │          │ S1 copy  │
        └────┬─────┘          └────┬─────┘
             │   A_shared          │   B_shared
             └─────────┬───────────┘
                       ▼
                 ┌──────────┐
                 │    S2    │
                 └──────────┘
    """
    if not stage_infos:
        return "Empty DAG"

    # Group stages by their "level" (topological order based on dependencies)
    levels: Dict[int, List[int]] = {}
    stage_level: Dict[int, int] = {}

    # Compute levels using BFS from sources
    for info in stage_infos:
        idx = info.original_stmt_index
        if not info.predecessors:
            stage_level[idx] = 0
        else:
            max_pred_level = max(stage_level.get(p, 0) for p in info.predecessors)
            stage_level[idx] = max_pred_level + 1

    for idx, level in stage_level.items():
        if level not in levels:
            levels[level] = []
        levels[level].append(idx)

    # Sort stages within each level
    for level in levels:
        levels[level].sort()

    # Build the visualization
    lines = []
    box_width = 12

    # Title
    title = "Pipeline Dependency DAG"
    title_width = max(50, len(stage_infos) * (box_width + 6))
    lines.append("┌" + "─" * title_width + "┐")
    lines.append("│" + title.center(title_width) + "│")
    lines.append("└" + "─" * title_width + "┘")
    lines.append("")

    max_level = max(levels.keys()) if levels else 0
    total_width = title_width + 2

    # Store positions for each stage for drawing connections
    stage_positions: Dict[int, int] = {}

    for level in range(max_level + 1):
        stage_indices = levels.get(level, [])
        num_stages = len(stage_indices)

        if num_stages == 0:
            continue

        # Calculate spacing
        stage_spacing = total_width // (num_stages + 1)

        # Calculate positions for this level
        positions = []
        for i in range(num_stages):
            pos = stage_spacing * (i + 1)
            positions.append(pos)
            stage_positions[stage_indices[i]] = pos

        # Build box lines
        box_top = [" "] * total_width
        box_mid = [" "] * total_width
        box_bot = [" "] * total_width

        for i, stage_idx in enumerate(stage_indices):
            info = stage_infos[stage_idx]
            label = f"S{stage_idx}"
            if info.copy_stage:
                label += " copy"
            elif info.producer_for_copy:
                label += " prod"

            label = label[:box_width - 2].center(box_width - 2)
            pos = positions[i]
            start = pos - box_width // 2

            # Draw box
            box_top[start] = "┌"
            for j in range(1, box_width - 1):
                box_top[start + j] = "─"
            box_top[start + box_width - 1] = "┐"

            box_mid[start] = "│"
            for j, c in enumerate(label):
                box_mid[start + 1 + j] = c
            box_mid[start + box_width - 1] = "│"

            box_bot[start] = "└"
            for j in range(1, box_width - 1):
                box_bot[start + j] = "─"
            box_bot[start + box_width - 1] = "┘"

        lines.append("".join(box_top))
        lines.append("".join(box_mid))
        lines.append("".join(box_bot))

        # Draw arrows to next level if not last level
        if level < max_level:
            next_stages = levels.get(level + 1, [])
            if next_stages:
                # Collect connections: from current level to next level
                connections = []
                for stage_idx in stage_indices:
                    info = stage_infos[stage_idx]
                    for succ_idx in info.successors:
                        if succ_idx in next_stages:
                            # Get dependency info
                            succ_info = stage_infos[succ_idx]
                            deps = [d for d in succ_info.dependencies if d.from_stage == stage_idx]
                            # Deduplicate
                            seen = set()
                            dep_names = []
                            for d in deps:
                                if d.buffer_name not in seen:
                                    seen.add(d.buffer_name)
                                    dep_names.append(d.buffer_name)
                            connections.append((stage_idx, succ_idx, dep_names))

                # Draw vertical lines with labels
                line1 = [" "] * total_width
                for stage_idx in stage_indices:
                    info = stage_infos[stage_idx]
                    if any(s in next_stages for s in info.successors):
                        pos = stage_positions[stage_idx]
                        line1[pos] = "│"

                lines.append("".join(line1))

                # Draw dependency labels next to lines
                label_line = [" "] * total_width
                for stage_idx, succ_idx, dep_names in connections:
                    if dep_names:
                        pos = stage_positions[stage_idx]
                        label = dep_names[0][:10]  # Truncate long names
                        start = pos + 2
                        for j, c in enumerate(label):
                            if start + j < total_width:
                                label_line[start + j] = c

                lines.append("".join(label_line))

                # Draw converging lines
                next_spacing = total_width // (len(next_stages) + 1)
                next_positions = [next_spacing * (i + 1) for i in range(len(next_stages))]

                # Draw horizontal merge line
                merge_line = [" "] * total_width
                for succ_idx in next_stages:
                    target_pos = next_positions[next_stages.index(succ_idx)]
                    # Find all sources for this target
                    source_positions = []
                    for stage_idx in stage_indices:
                        info = stage_infos[stage_idx]
                        if succ_idx in info.successors:
                            source_positions.append(stage_positions[stage_idx])

                    if source_positions:
                        min_pos = min(source_positions)
                        max_pos = max(source_positions)

                        # Draw horizontal line from sources to target
                        for p in range(min_pos, max_pos + 1):
                            if merge_line[p] == " ":
                                merge_line[p] = "─"

                        # Draw corners
                        for sp in source_positions:
                            if sp == min_pos:
                                merge_line[sp] = "└"
                            elif sp == max_pos:
                                merge_line[sp] = "┘"
                            else:
                                merge_line[sp] = "┴"

                        # Draw down arrow point
                        center = (min_pos + max_pos) // 2
                        merge_line[center] = "┬"

                lines.append("".join(merge_line))

                # Draw arrow to target
                arrow_line = [" "] * total_width
                for i, succ_idx in enumerate(next_stages):
                    # Find center of sources
                    source_positions = []
                    for stage_idx in stage_indices:
                        info = stage_infos[stage_idx]
                        if succ_idx in info.successors:
                            source_positions.append(stage_positions[stage_idx])
                    if source_positions:
                        center = (min(source_positions) + max(source_positions)) // 2
                        arrow_line[center] = "▼"

                lines.append("".join(arrow_line))
                lines.append("")

    return "\n".join(lines)


def dag_to_mermaid(stage_infos: List[PipelineStageInfo]) -> str:
    """
    Generate Mermaid format string for the dependency DAG.

    Mermaid is supported by GitHub markdown and many documentation tools.

    Parameters
    ----------
    stage_infos : List[PipelineStageInfo]
        List of pipeline stage info objects with DAG fields populated.

    Returns
    -------
    str
        Mermaid format string.

    Example
    -------
    >>> print(dag_to_mermaid(stage_infos))
    ```mermaid
    graph TD
        S0[S0 copy]:::copy --> S2
        S1[S1 copy]:::copy --> S2
        classDef copy fill:#add8e6
    ```
    """
    lines = ["```mermaid", "graph TD"]

    # Define nodes
    for info in stage_infos:
        idx = info.original_stmt_index
        label = f"S{idx}"
        if info.copy_stage:
            label += " copy"
            lines.append(f"    S{idx}[{label}]:::copy")
        elif info.producer_for_copy:
            label += " prod"
            lines.append(f"    S{idx}[{label}]:::prod")
        else:
            lines.append(f"    S{idx}[{label}]")

    # Define edges
    added_edges: Set[Tuple[int, int]] = set()
    for info in stage_infos:
        for succ_idx in info.successors:
            edge_key = (info.original_stmt_index, succ_idx)
            if edge_key not in added_edges:
                lines.append(f"    S{info.original_stmt_index} --> S{succ_idx}")
                added_edges.add(edge_key)

    # Add styles
    lines.append("    classDef copy fill:#add8e6")
    lines.append("    classDef prod fill:#fffacd")
    lines.append("```")

    return "\n".join(lines)


def _target_has_async_copy(target: ir.Target) -> bool:
    """Check if the target supports async copy."""
    if target is None:
        return False

    kind = str(target.kind)
    if kind != "cuda":
        return False

    # Check for SM80+ (Ampere and later)
    arch = target.attrs.get("arch", "")
    if arch:
        # Extract SM version from arch string like "sm_80" or "sm_100a"
        if arch.startswith("sm_"):
            try:
                # Remove any trailing letters (like 'a' in 'sm_100a')
                version_str = arch[3:]
                numeric_part = ""
                for c in version_str:
                    if c.isdigit():
                        numeric_part += c
                    else:
                        break
                if numeric_part:
                    sm_version = int(numeric_part)
                    return sm_version >= 80
            except ValueError:
                pass

    return False


class PipelinePlanner:
    """
    Pipeline planner that transforms loop bodies with pipeline annotations.
    """

    def __init__(self, use_async_copy: bool = True, verbose: bool = False):
        self.buffer_data_to_buffer: Dict[Var, Buffer] = {}
        self.target: Optional[ir.Target] = None
        self.use_async_copy = use_async_copy
        self.verbose = verbose

    def substitute(self, f: PrimFunc) -> Stmt:
        """Apply pipeline planning transformation to a PrimFunc."""
        # Collect buffer_data -> buffer mapping from buffer_map
        for _, buffer in f.buffer_map.items():
            self.buffer_data_to_buffer[buffer.data] = buffer

        # Get target attribute
        target = f.attrs.get("target", None)
        if target is None:
            raise ValueError("Pipeline_Planning: Require the target attribute")
        self.target = target

        return self._visit_stmt(f.body)

    def _visit_stmt(self, stmt: Stmt) -> Stmt:
        """Recursively visit and transform statements."""
        if isinstance(stmt, For):
            return self._visit_for(stmt)
        elif isinstance(stmt, Block):
            return self._visit_block(stmt)
        elif isinstance(stmt, BlockRealize):
            new_block = self._visit_stmt(stmt.block)
            if new_block.same_as(stmt.block):
                return stmt
            return BlockRealize(stmt.iter_values, stmt.predicate, new_block)
        elif isinstance(stmt, SeqStmt):
            new_seq = [self._visit_stmt(s) for s in stmt.seq]
            return SeqStmt(new_seq)
        elif isinstance(stmt, IfThenElse):
            new_then = self._visit_stmt(stmt.then_case)
            new_else = self._visit_stmt(stmt.else_case) if stmt.else_case else None
            return IfThenElse(stmt.condition, new_then, new_else)
        elif isinstance(stmt, LetStmt):
            new_body = self._visit_stmt(stmt.body)
            return LetStmt(stmt.var, stmt.value, new_body)
        elif isinstance(stmt, AttrStmt):
            new_body = self._visit_stmt(stmt.body)
            if new_body.same_as(stmt.body):
                return stmt
            return AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        else:
            return stmt

    def _visit_block(self, op: Block) -> Block:
        """Visit a Block node."""
        # Register allocated buffers
        for buffer in op.alloc_buffers:
            self.buffer_data_to_buffer[buffer.data] = buffer

        new_body = self._visit_stmt(op.body)

        # Unregister allocated buffers
        for buffer in op.alloc_buffers:
            if buffer.data in self.buffer_data_to_buffer:
                del self.buffer_data_to_buffer[buffer.data]

        if new_body.same_as(op.body):
            return op

        return Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            new_body,
            op.init,
            op.alloc_buffers,
            op.match_buffers,
            op.annotations,
        )

    def _visit_for(self, loop: For) -> Stmt:
        """Visit a For loop and potentially add pipeline annotations."""
        annotations = dict(loop.annotations)

        order_anno = annotations.get("tl_pipeline_order")
        stage_anno = annotations.get("tl_pipeline_stage")
        num_stages_anno = annotations.get("num_stages")

        # If order and stage annotations already exist
        if order_anno is not None and stage_anno is not None:
            # Check if WS+TMA is enabled (contains -1)
            ws_tma_enabled = False
            order_array = list(order_anno)
            stage_array = list(stage_anno)

            for val in order_array:
                if isinstance(val, (int, IntImm)):
                    v = val.value if isinstance(val, IntImm) else val
                    if v == -1:
                        ws_tma_enabled = True
                        break

            if not ws_tma_enabled:
                for val in stage_array:
                    if isinstance(val, (int, IntImm)):
                        v = val.value if isinstance(val, IntImm) else val
                        if v == -1:
                            ws_tma_enabled = True
                            break

            if ws_tma_enabled:
                # Recursively visit body
                new_body = self._visit_stmt(loop.body)
                if new_body.same_as(loop.body):
                    return loop
                return For(
                    loop.loop_var,
                    loop.min,
                    loop.extent,
                    loop.kind,
                    new_body,
                    loop.thread_binding,
                    loop.annotations,
                )

            # Replace tl_pipeline_* with software_pipeline_*
            new_annotations = {}
            for key, value in annotations.items():
                if key == "tl_pipeline_order":
                    new_annotations["software_pipeline_order"] = value
                elif key == "tl_pipeline_stage":
                    new_annotations["software_pipeline_stage"] = value
                else:
                    new_annotations[key] = value

            if _target_has_async_copy(self.target) and self.use_async_copy:
                new_annotations["software_pipeline_async_stages"] = [0]

            return For(
                loop.loop_var,
                loop.min,
                loop.extent,
                loop.kind,
                loop.body,
                loop.thread_binding,
                new_annotations,
            )

        # If no num_stages annotation, just visit recursively
        if num_stages_anno is None:
            new_body = self._visit_stmt(loop.body)
            if new_body.same_as(loop.body):
                return loop
            return For(
                loop.loop_var,
                loop.min,
                loop.extent,
                loop.kind,
                new_body,
                loop.thread_binding,
                loop.annotations,
            )

        # Extract num_stages value
        num_stages = (
            num_stages_anno.value
            if isinstance(num_stages_anno, IntImm)
            else int(num_stages_anno)
        )

        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")

        if loop.kind != ForKind.SERIAL:
            raise ValueError("Pipeline loop must be serial")

        # Find the pipeline body
        pipeline_body_root: Optional[Stmt] = None
        if isinstance(loop.body, BlockRealize):
            block = loop.body.block
            for buffer in block.alloc_buffers:
                self.buffer_data_to_buffer[buffer.data] = buffer
            pipeline_body_root = block.body
        else:
            pipeline_body_root = loop.body

        # Navigate through IfThenElse and LetStmt to find SeqStmt
        pipeline_body_seq: Optional[SeqStmt] = None
        current = pipeline_body_root
        while True:
            if isinstance(current, SeqStmt):
                pipeline_body_seq = current
                break
            elif isinstance(current, IfThenElse):
                if current.else_case is not None:
                    raise ValueError(
                        "Pipeline_Planning: Can't handle IfThenElse with else branch"
                    )
                current = current.then_case
            elif isinstance(current, LetStmt):
                current = current.body
            else:
                raise ValueError(
                    f"Pipeline_Planning: Can't handle body type {type(current)}"
                )

        if pipeline_body_seq is None:
            raise ValueError("Pipeline_Planning: Could not find SeqStmt in loop body")

        # Build async dependency chain
        chain_builder = AsyncDependencyChainBuilder(self.buffer_data_to_buffer)
        chain_builder(pipeline_body_root)

        # Create pipeline stage info for each statement
        pipeline_stage_infos: List[PipelineStageInfo] = []
        for i, stmt in enumerate(pipeline_body_seq.seq):
            pinfo = self._make_pipeline_stage_info(stmt, i, chain_builder)
            pipeline_stage_infos.append(pinfo)

        # Build dependency DAG and optionally print it
        build_dependency_dag(pipeline_stage_infos)
        if self.verbose:
            print("\n" + "=" * 60)
            print("Pipeline Planning: Stage Analysis")
            print("=" * 60)
            # First show detailed list format
            print(dag_to_ascii(pipeline_stage_infos, style="list"))
            print()
            # Then show visual vertical format
            print(dag_to_ascii(pipeline_stage_infos, style="vertical"))
            print()

        # Mark copy stage dependencies
        copy_stage_dependency_reads_mgr = CopyStageDependencyReadsManager()

        # Step 1: Collect copy reads
        for pinfo in pipeline_stage_infos:
            if pinfo.is_copy_stage():
                for read in pinfo.reads:
                    copy_stage_dependency_reads_mgr.add_unique(read)

        # Step 2: Find producers for copy stages
        max_iterations = len(pipeline_stage_infos) * 4 + 16
        iter_count = 0

        for pinfo in pipeline_stage_infos:
            if not pinfo.is_copy_stage():
                continue

            original_copy_stmt_index = pinfo.original_stmt_index
            updated = True

            while updated:
                updated = False
                for pinfo_inner in pipeline_stage_infos:
                    if pinfo_inner.is_copy_stage():
                        continue
                    if pinfo_inner.original_stmt_index >= original_copy_stmt_index:
                        break

                    should_prepare = False
                    for write in pinfo_inner.writes:
                        if copy_stage_dependency_reads_mgr.contains(write):
                            should_prepare = True
                            break

                    if should_prepare and not pinfo_inner.is_producer_for_copy():
                        pinfo_inner.producer_for_copy = True
                        updated = True

                    if should_prepare:
                        for read in pinfo_inner.reads:
                            before = copy_stage_dependency_reads_mgr.size()
                            copy_stage_dependency_reads_mgr.add_unique(read)
                            if copy_stage_dependency_reads_mgr.size() > before:
                                updated = True

                iter_count += 1
                if iter_count > max_iterations:
                    raise RuntimeError(
                        f"Pipeline planning: Exceeded maximum iterations ({max_iterations})"
                    )

        # Analyze use-def chain to determine last_use_stmt_index
        for pinfo in pipeline_stage_infos:
            if not pinfo.is_first_stage():
                continue

            for i in range(
                pinfo.original_stmt_index + 1, len(pipeline_body_seq.seq)
            ):
                for read in pipeline_stage_infos[i].reads:
                    for write in pinfo.writes:
                        if write.buffer.same_as(read.buffer) and _may_conflict(
                            list(write.region), list(read.region)
                        ):
                            pinfo.last_use_stmt_index = max(
                                pinfo.last_use_stmt_index, i
                            )

                # Check for write-after-write conflicts
                if pinfo.is_copy_stage():
                    for write in pipeline_stage_infos[i].writes:
                        for pinfo_write in pinfo.writes:
                            if pinfo_write.buffer.same_as(
                                write.buffer
                            ) and _may_conflict(
                                list(pinfo_write.region), list(write.region)
                            ):
                                raise ValueError(
                                    f"Pipeline planning error: Multiple writes to overlapping buffer regions. "
                                    f"Stage {pinfo.original_stmt_index} and stage {i} "
                                    f"are both writing to buffer '{write.buffer.name}'"
                                )

        # Make stages and orders
        order_idx = 0

        # Stage 1: Create pipeline stages and assign order
        for pinfo in pipeline_stage_infos:
            # Skip elements that must be in first stage
            if pinfo.is_first_stage() and pinfo.is_last_use_stmt_index_valid():
                continue

            # Main logic stage assignment
            pinfo.order = order_idx
            order_idx += 1
            pinfo.stage = num_stages

            # Schedule copy stages that have this stage as their last consumer
            for pinfo_1 in pipeline_stage_infos:
                if (
                    pinfo_1.is_first_stage()
                    and pinfo_1.last_use_stmt_index == pinfo.original_stmt_index
                ):
                    pinfo_1.order = order_idx
                    order_idx += 1
                    pinfo_1.stage = 0

        if order_idx != len(pipeline_stage_infos):
            raise ValueError(
                f"The number of stages should be equal to the number of pipeline stages. "
                f"Got {order_idx} stages and {len(pipeline_stage_infos)} pipeline stages."
            )

        # Step 2: If all copy stages are at the end, move them to the beginning
        copy_stage_cnt = 0
        copy_order_min = len(pipeline_stage_infos)
        non_copy_order_max = 0

        for pinfo in pipeline_stage_infos:
            if pinfo.is_first_stage():
                copy_stage_cnt += 1
                copy_order_min = min(copy_order_min, pinfo.order)
            else:
                non_copy_order_max = max(non_copy_order_max, pinfo.order)

        copy_stage_at_end = copy_stage_cnt if copy_order_min > non_copy_order_max else -1

        if copy_stage_at_end > 0 and num_stages >= 2:
            for pinfo in pipeline_stage_infos:
                pinfo.order = (pinfo.order + copy_stage_at_end) % len(
                    pipeline_stage_infos
                )
                if not pinfo.is_copy_stage() and not pinfo.is_producer_for_copy():
                    pinfo.stage -= 1

        # Build final annotations
        new_annotations = {}
        for key, value in annotations.items():
            if key != "num_stages":
                new_annotations[key] = value

        # Use IntImm for annotation values to match C++ behavior
        orders = [IntImm("int32", pinfo.order) for pinfo in pipeline_stage_infos]
        stages = [IntImm("int32", pinfo.stage) for pinfo in pipeline_stage_infos]

        if self.verbose:
            print("Pipeline Planning: Final Stage Assignments")
            print("-" * 40)
            for pinfo in pipeline_stage_infos:
                reads_str = ", ".join(r.buffer.name for r in pinfo.reads)
                writes_str = ", ".join(w.buffer.name for w in pinfo.writes)
                print(f"  S{pinfo.original_stmt_index}: order={pinfo.order}, stage={pinfo.stage}")
                print(f"      reads: [{reads_str}]")
                print(f"      writes: [{writes_str}]")
                print(f"      copy_stage={pinfo.copy_stage}, producer_for_copy={pinfo.producer_for_copy}")
            print(f"\nFinal order: {[o.value for o in orders]}")
            print(f"Final stage: {[s.value for s in stages]}")
            print("=" * 60 + "\n")

        new_annotations["software_pipeline_stage"] = stages
        new_annotations["software_pipeline_order"] = orders

        if _target_has_async_copy(self.target) and self.use_async_copy:
            new_annotations["software_pipeline_async_stages"] = [IntImm("int32", 0)]

        return For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            new_annotations,
        )

    def _make_pipeline_stage_info(
        self, stmt: Stmt, idx: int, chain_builder: AsyncDependencyChainBuilder
    ) -> PipelineStageInfo:
        """Create PipelineStageInfo for a statement."""
        collector = BufferRegionCollector(self.buffer_data_to_buffer, chain_builder)
        collector(stmt)

        pinfo = PipelineStageInfo()
        pinfo.reads = collector.reads
        pinfo.writes = collector.writes
        pinfo.original_stmt_index = idx
        pinfo.copy_stage = collector.is_global_copy_pattern

        return pinfo


def PipelinePlanning(use_async_copy: bool = True, verbose: bool = False):
    """
    Create a pipeline planning pass.

    This pass analyzes loop bodies with `num_stages` annotations and generates
    appropriate `software_pipeline_stage` and `software_pipeline_order` annotations
    for efficient memory copy pipelining.

    Parameters
    ----------
    use_async_copy : bool
        Whether to enable async copy for supported targets.
    verbose : bool
        Whether to print the dependency DAG and stage assignments for debugging.

    Returns
    -------
    pass : tvm.transform.Pass
        The pipeline planning pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        planner = PipelinePlanner(use_async_copy, verbose)
        new_body = planner.substitute(func)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.PipelinePlanning")
