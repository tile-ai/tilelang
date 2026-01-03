"""
Pipeline Planning Pass for TileLang.

This module implements the pipeline planning pass that analyzes loop bodies,
detects global copy patterns, and generates software pipeline annotations
for efficient memory copy pipelining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    PyStmtExprMutator,
    PyStmtExprVisitor,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.stmt_functor import post_order_visit
from tvm.tir.transform import prim_func_pass

from tilelang.utils import is_global, is_shared

# Get builtin operations using Op.get
_OP_TVM_ACCESS_PTR = tir.op.Op.get("tir.tvm_access_ptr")
_OP_ADDRESS_OF = tir.op.Op.get("tir.address_of")
_OP_IF_THEN_ELSE = tir.op.Op.get("tir.if_then_else")
_OP_TILEOP_REGION = tir.op.Op.get("tl.tileop.region")


def _full_buffer_region(buf: Buffer) -> BufferRegion:
    """Create a BufferRegion covering the entire buffer."""
    region = [Range.from_min_extent(0, s) for s in buf.shape]
    return BufferRegion(buf, region)


def _may_conflict(region1: list[Range], region2: list[Range]) -> bool:
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

    def __init__(self, buffer_data_to_buffer: dict[Var, Buffer]):
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.mbar_to_buffer_reads: dict[Buffer, list[BufferRegion]] = {}
        self.mbar_to_buffer_writes: dict[Buffer, list[BufferRegion]] = {}

    def __call__(self, stmt: Stmt) -> None:
        def visit(node):
            if isinstance(node, Call):
                self._visit_call(node)

        post_order_visit(stmt, visit)

    def _get_buf_from_access_ptr_call(self, expr: PrimExpr) -> Buffer | None:
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
        if op.op.same_as(_OP_IF_THEN_ELSE) and len(args) >= 3:
            # Recursively visit then and else expressions
            # condition, then, else are defined in the if_then_else node
            post_order_visit(args[1], lambda n: self._visit_call(n) if isinstance(n, Call) else None)
            post_order_visit(args[2], lambda n: self._visit_call(n) if isinstance(n, Call) else None)


@tir.functor.visitor
class BufferRegionCollector(PyStmtExprVisitor):
    """
    Detect if a statement follows the global memory copy pattern:
        1. Contains exactly one buffer store operation
        2. Source buffer must be in global memory scope
        3. Destination buffer must be in local or shared memory scope
    """

    def __init__(
        self,
        buffer_data_to_buffer: dict[Var, Buffer],
        chain_builder: AsyncDependencyChainBuilder,
    ):
        super().__init__()
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.chain_builder = chain_builder
        self.reads: list[BufferRegion] = []
        self.writes: list[BufferRegion] = []
        self.is_global_copy_pattern = False
        self._is_global_read = False
        self._within_condition_expr = False

    def visit_buffer_store_(self, op: BufferStore) -> None:
        store_buffer = op.buffer
        indices = op.indices

        # Convert indices to region
        region = [Range.from_min_extent(idx, 1) for idx in indices]
        store_region = BufferRegion(store_buffer, region)
        self.writes.append(store_region)

        self._is_global_read = False
        self.visit_expr(op.value)

        if self._is_global_read and is_shared(store_buffer):
            self.is_global_copy_pattern = True

        self._is_global_read = False

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        load_buffer = op.buffer
        indices = op.indices

        # Convert indices to region
        region = [Range.from_min_extent(idx, 1) for idx in indices]
        load_region = BufferRegion(load_buffer, region)
        self.reads.append(load_region)

        if is_global(load_buffer) and not self._within_condition_expr:
            self._is_global_read = True

    def visit_call_(self, op: Call) -> None:
        args = op.args

        if op.op.same_as(_OP_ADDRESS_OF):
            # T.address_of(buffer_load) - extract buffer from address_of
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
            # tvm_access_ptr(dtype, data, offset, extent, access_mask)
            if len(args) >= 2 and isinstance(args[1], Var):
                buf = self.buffer_data_to_buffer.get(args[1])
                if buf is not None:
                    buffer_region = _full_buffer_region(buf)
                    self.reads.append(buffer_region)

        elif op.op.same_as(_OP_TILEOP_REGION):
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
                        if is_global(buffer_load.buffer) and not self._within_condition_expr:
                            self._is_global_read = True
                    if access_val & 2:  # write
                        self.writes.append(region)
                        if is_shared(buffer_load.buffer):
                            self.is_global_copy_pattern = self._is_global_read

        else:
            # For all other calls, recursively visit arguments
            for arg in args:
                self.visit_expr(arg)

    def visit_if_then_else_(self, op: IfThenElse) -> None:
        self._within_condition_expr = True
        self.visit_expr(op.condition)
        self._within_condition_expr = False

        self.visit_stmt(op.then_case)
        if op.else_case is not None:
            self._within_condition_expr = True
            self.visit_stmt(op.else_case)
            self._within_condition_expr = False


@dataclass
class PipelineStageInfo:
    """
    Information about a pipeline stage.

    Attributes
    ----------
    reads : list[BufferRegion]
        Array of buffer regions read by this stage
    writes : list[BufferRegion]
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
    predecessors : set[int]
        Set of stage indices that this stage depends on (incoming edges in DAG)
    successors : set[int]
        Set of stage indices that depend on this stage (outgoing edges in DAG)
    dependencies : list[StageDependency]
        Detailed dependency edges from predecessors to this stage
    """

    reads: list[BufferRegion] = field(default_factory=list)
    writes: list[BufferRegion] = field(default_factory=list)
    original_stmt_index: int = 0
    order: int = -1
    stage: int = -1
    copy_stage: bool = False
    producer_for_copy: bool = False
    last_use_stmt_index: int = -1
    # DAG fields
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)
    dependencies: list[StageDependency] = field(default_factory=list)

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
        self.regions: list[BufferRegion] = []

    def add_unique(self, region: BufferRegion) -> bool:
        """Add a region if not already present (by buffer identity)."""
        for copy_read in self.regions:
            if region.buffer.same_as(copy_read.buffer):
                return False
        self.regions.append(region)
        return True

    def contains(self, region: BufferRegion) -> bool:
        """Check if a region is present (by buffer identity)."""
        return any(region.buffer.same_as(copy_read.buffer) for copy_read in self.regions)

    def size(self) -> int:
        return len(self.regions)


def build_dependency_dag(stage_infos: list[PipelineStageInfo]) -> None:
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
                    if write_i.buffer.same_as(read_j.buffer) and _may_conflict(list(write_i.region), list(read_j.region)):
                        stage_i.successors.add(j)
                        stage_j.predecessors.add(i)
                        stage_j.dependencies.append(StageDependency(i, j, DependencyType.RAW, write_i.buffer.name))

            # Check WAW: j writes what i also writes
            for write_i in stage_i.writes:
                for write_j in stage_j.writes:
                    if write_i.buffer.same_as(write_j.buffer) and _may_conflict(list(write_i.region), list(write_j.region)):
                        # Only add if not already connected
                        if j not in stage_i.successors:
                            stage_i.successors.add(j)
                            stage_j.predecessors.add(i)
                        stage_j.dependencies.append(StageDependency(i, j, DependencyType.WAW, write_i.buffer.name))

            # Check WAR: j writes what i reads
            for read_i in stage_i.reads:
                for write_j in stage_j.writes:
                    if read_i.buffer.same_as(write_j.buffer) and _may_conflict(list(read_i.region), list(write_j.region)):
                        # Only add if not already connected
                        if j not in stage_i.successors:
                            stage_i.successors.add(j)
                            stage_j.predecessors.add(i)
                        stage_j.dependencies.append(StageDependency(i, j, DependencyType.WAR, read_i.buffer.name))


# Import visualization functions from separate module
from .dag_visualization import dag_to_dot, dag_to_ascii, dag_to_mermaid

# Pipeline DAG printing mask constants
PRINT_DAG_LIST = 1  # ASCII list format
PRINT_DAG_VERTICAL = 2  # ASCII vertical format
PRINT_DAG_DOT = 4  # DOT format (Graphviz)
PRINT_DAG_MERMAID = 8  # Mermaid format


def _get_pipeline_dag_print_mask() -> int:
    """Get the pipeline DAG printing mask from PassContext config.

    Returns
    -------
    int
        Bitmask controlling which DAG formats to print:
          0: No printing (default)
          1: ASCII list format
          2: ASCII vertical format
          4: DOT format (Graphviz)
          8: Mermaid format
        Values can be combined, e.g., 3 = list + vertical.
    """
    ctx = tvm.transform.PassContext.current()
    return ctx.config.get("tl.print_pipeline_dag", 0)


@tir.functor.mutator
class PipelinePlanner(PyStmtExprMutator):
    """
    Pipeline planner that transforms loop bodies with pipeline annotations.
    """

    def __init__(self, buffer_data_to_buffer: dict[Var, Buffer], target: ir.Target):
        super().__init__()
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.target = target

    def visit_block_(self, op: Block) -> Block:
        """Visit a Block node."""
        # Register allocated buffers
        for buffer in op.alloc_buffers:
            self.buffer_data_to_buffer[buffer.data] = buffer

        new_body = self.visit_stmt(op.body)

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

    def visit_for_(self, loop: For) -> Stmt:
        """Visit a For loop and potentially add pipeline annotations."""
        annotations = dict(loop.annotations)

        num_stages_anno = annotations.get("num_stages")
        order_anno = annotations.get("tl_pipeline_order")
        stage_anno = annotations.get("tl_pipeline_stage")

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
                new_body = self.visit_stmt(loop.body)
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
            new_body = self.visit_stmt(loop.body)
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
        num_stages = num_stages_anno.value if isinstance(num_stages_anno, IntImm) else int(num_stages_anno)

        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")

        if loop.kind != ForKind.SERIAL:
            raise ValueError("Pipeline loop must be serial")

        # Find the pipeline body
        pipeline_body_root: Stmt | None = None
        if isinstance(loop.body, BlockRealize):
            block = loop.body.block
            for buffer in block.alloc_buffers:
                self.buffer_data_to_buffer[buffer.data] = buffer
            pipeline_body_root = block.body
        else:
            pipeline_body_root = loop.body

        # Navigate through IfThenElse and LetStmt to find SeqStmt
        pipeline_body_seq: SeqStmt | None = None
        current = pipeline_body_root
        while True:
            if isinstance(current, SeqStmt):
                pipeline_body_seq = current
                break
            elif isinstance(current, IfThenElse):
                if current.else_case is not None:
                    raise ValueError("Pipeline_Planning: Can't handle IfThenElse with else branch")
                current = current.then_case
            elif isinstance(current, LetStmt):
                current = current.body
            else:
                raise ValueError(f"Pipeline_Planning: Can't handle body type {type(current)}")

        if pipeline_body_seq is None:
            raise ValueError("Pipeline_Planning: Could not find SeqStmt in loop body")

        # Build async dependency chain
        chain_builder = AsyncDependencyChainBuilder(self.buffer_data_to_buffer)
        chain_builder(pipeline_body_root)

        # Create pipeline stage info for each statement
        pipeline_stage_infos: list[PipelineStageInfo] = []
        for i, stmt in enumerate(pipeline_body_seq.seq):
            pinfo = self._make_pipeline_stage_info(stmt, i, chain_builder)
            pipeline_stage_infos.append(pinfo)

        # Build dependency DAG and optionally print it
        build_dependency_dag(pipeline_stage_infos)
        print_mask = _get_pipeline_dag_print_mask()
        if print_mask:
            print("\n" + "=" * 60)
            print("Pipeline Planning: Stage Analysis")
            print("=" * 60)
            if print_mask & PRINT_DAG_LIST:
                print(dag_to_ascii(pipeline_stage_infos, style="list"))
                print()
            if print_mask & PRINT_DAG_VERTICAL:
                print(dag_to_ascii(pipeline_stage_infos, style="vertical"))
                print()
            if print_mask & PRINT_DAG_DOT:
                print(dag_to_dot(pipeline_stage_infos))
                print()
            if print_mask & PRINT_DAG_MERMAID:
                print(dag_to_mermaid(pipeline_stage_infos))
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
                    raise RuntimeError(f"Pipeline planning: Exceeded maximum iterations ({max_iterations})")

        # Analyze use-def chain to determine last_use_stmt_index
        for pinfo in pipeline_stage_infos:
            if not pinfo.is_first_stage():
                continue

            for i in range(pinfo.original_stmt_index + 1, len(pipeline_body_seq.seq)):
                for read in pipeline_stage_infos[i].reads:
                    for write in pinfo.writes:
                        if write.buffer.same_as(read.buffer) and _may_conflict(list(write.region), list(read.region)):
                            pinfo.last_use_stmt_index = max(pinfo.last_use_stmt_index, i)

                # Check for write-after-write conflicts
                if pinfo.is_copy_stage():
                    for write in pipeline_stage_infos[i].writes:
                        for pinfo_write in pinfo.writes:
                            if pinfo_write.buffer.same_as(write.buffer) and _may_conflict(list(pinfo_write.region), list(write.region)):
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
                if pinfo_1.is_first_stage() and pinfo_1.last_use_stmt_index == pinfo.original_stmt_index:
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
                pinfo.order = (pinfo.order + copy_stage_at_end) % len(pipeline_stage_infos)
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

        if print_mask:
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

        return For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            new_annotations,
        )

    def _make_pipeline_stage_info(self, stmt: Stmt, idx: int, chain_builder: AsyncDependencyChainBuilder) -> PipelineStageInfo:
        """Create PipelineStageInfo for a statement."""
        collector = BufferRegionCollector(self.buffer_data_to_buffer, chain_builder)
        collector.visit_stmt(stmt)

        pinfo = PipelineStageInfo()
        pinfo.reads = collector.reads
        pinfo.writes = collector.writes
        pinfo.original_stmt_index = idx
        pinfo.copy_stage = collector.is_global_copy_pattern

        return pinfo


def PipelinePlanning():
    """
    Create a pipeline planning pass.

    This pass analyzes loop bodies with `num_stages` annotations and generates
    appropriate `software_pipeline_stage` and `software_pipeline_order` annotations
    for efficient memory copy pipelining.

    Parameters
    ----------

    Notes
    -----
    To print the pipeline DAG and stage assignments for debugging, set the
    PassContext config option "tl.print_pipeline_dag" to a bitmask value:

        # Print ASCII list format only
        with tvm.transform.PassContext(config={"tl.print_pipeline_dag": 1}):
            mod = PipelinePlanning()(mod)

        # Print both list and vertical ASCII formats
        with tvm.transform.PassContext(config={"tl.print_pipeline_dag": 3}):
            mod = PipelinePlanning()(mod)

        # Print all formats (list + vertical + DOT + Mermaid)
        with tvm.transform.PassContext(config={"tl.print_pipeline_dag": 15}):
            mod = PipelinePlanning()(mod)

    Mask values:
        0: No printing (default)
        1: ASCII list format
        2: ASCII vertical format
        4: DOT format (Graphviz)
        8: Mermaid format

    Returns
    -------
    pass : tvm.transform.Pass
        The pipeline planning pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        # Collect buffer_data -> buffer mapping from buffer_map
        buffer_data_to_buffer: dict[Var, Buffer] = {}
        for _, buffer in func.buffer_map.items():
            buffer_data_to_buffer[buffer.data] = buffer

        # Get target attribute
        target = func.attrs.get("target", None)
        if target is None:
            raise ValueError("PipelinePlanning: Require the target attribute")

        planner = PipelinePlanner(buffer_data_to_buffer, target)
        new_body = planner.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.PipelinePlanning")
