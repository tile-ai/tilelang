"""Analysis on TIR blocks, loops and functions."""

from __future__ import annotations
from typing import Literal

from tvm_tirx import ir, tirx
from tvm_ffi import dtype as DataType
from tvm_tirx.target import Target
from ._native_analysis import NativeSchedule, StageRef, LoopRef, StageView, TensorView, post_order_visit


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: ir.Var
    _dom: ir.Expr
    loop_rv: LoopRef

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: ir.Var,
        dom: ir.Expr,
        loop_rv: LoopRef,
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv

    @property
    def dom(self) -> int | ir.Expr:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, ir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom})'

    def __repr__(self) -> str:
        return str(self)


class BlockInfo:
    """Information about a TIR block."""

    name: str
    iters: list[IterInfo]
    block_rv: StageRef
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        iters: list[IterInfo],
        block_rv: StageRef,
        reduction_block: bool = False,
    ):
        """Construct a BlockInfo object."""
        self.name = name
        self.block_rv = block_rv
        self.iters = iters
        self._reduction_block = reduction_block

    def dom(self) -> list[int | ir.Expr]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the block is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self, sch: NativeSchedule) -> bool:
        """Whether the block is elementwise, i.e. trivial mapping between read/write region"""

        def _check_unit_var_range(dom: ir.Range, var: ir.Var) -> bool:
            return dom.begin.same_as(var) and dom.extent == 1

        if not self.is_injective():
            return False
        block = sch.get(self.block_rv)
        if len(block.reads) != 1 or len(block.writes) != 1:
            return False
        r_region = block.reads[0].region
        w_region = block.writes[0].region
        if len(r_region) != len(w_region):
            return False
        for var, r_dom, w_dom in zip(block.axes, r_region, w_region):
            if not _check_unit_var_range(r_dom, var.var) or not _check_unit_var_range(w_dom, var.var):
                return False
        return True

    def is_reduction(self) -> bool:
        """Whether the block is a reduction workload."""
        # TODO(@junrushao): distinguish GEMV and reduction
        return self._reduction_block

    def is_gemv(self) -> bool:
        """Whether the block is a GEMV workload."""
        raise NotImplementedError

    def is_gemm(self) -> bool:
        """Whether the block is a GEMM workload."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f'BlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


def normalize_prim_func(sch: NativeSchedule) -> list[BlockInfo] | None:
    """Normalize native stage axes, preserving the original failure boundary."""
    blocks = sch.normalize()
    if blocks is None:
        return None
    return [
        BlockInfo(
            name=sch.get(block).name_hint,
            iters=[IterInfo(axis.kind, axis.var, axis.domain, loop) for axis, loop in zip(sch.get(block).axes, sch.get_loops(block))],
            block_rv=block,
            reduction_block=sch.get(block).init is not None,
        )
        for block in blocks
    ]


def find_var_from_func(func, var: str):
    matches = set()
    for buffer in (param for param in func.params if tirx.is_tensor_var(param)):
        for dim in buffer.ty.shape:
            matches.update(candidate for candidate in collect_vars_used_in_prim_expr(dim) if candidate.name == var)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous Carver shape variable name {var!r}; use distinct names for distinct identities")
    return next(iter(matches)) if matches else None


def check_func_with_dynamic(func):
    for buffer in (param for param in func.params if tirx.is_tensor_var(param)):
        for i in buffer.ty.shape:
            if collect_vars_used_in_prim_expr(i):
                return True
    return False


def _assert_gpu_target(target: Target):
    if "gpu" not in target.keys:
        raise ValueError(f"Expect a GPU target, but got {target}")


def get_max_threads_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


def get_max_shared_memory_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_shared_memory_per_block = target.attrs.get("max_shared_memory_per_block", None)
    if max_shared_memory_per_block is None:
        raise ValueError(f"Cannot find `max_shared_memory_per_block` in {target}, please specify it manually")
    return int(max_shared_memory_per_block)


def get_root_block(sch: NativeSchedule, func_name: str = "main") -> StageRef:
    if func_name != "main":
        raise ValueError(f"Unknown Carver function {func_name!r}")
    return sch.root


def collect_block_iter_vars_used_in_access_region(block: StageView, region: list[ir.Range]) -> set[ir.Var]:
    """Collect the block iter variables used in the access region of a buffer region."""
    tir_vars = set()
    for expr in region:
        if expr.extent != 1:
            continue
        tir_vars |= collect_vars_used_in_prim_expr(expr.begin)
    tir_vars &= set(iter_var.var for iter_var in block.axes)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: ir.Expr) -> set[ir.Var]:
    """Collect the variables used in the PrimExpr."""
    tir_vars = set()

    def _collect_tir_var(expr):
        if isinstance(expr, ir.Var):
            tir_vars.add(expr)

    post_order_visit(expr, _collect_tir_var)
    return tir_vars


def detect_dominant_read(block: StageView) -> ir.Expr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(block, buffer_region.region)
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    (result,) = dominant_read.tensor.offset_of([e.begin for e in dominant_read.region])
    return result


def is_broadcast_epilogue(
    sch: NativeSchedule,
    block: StageRef,
    epilogue: StageRef,
) -> bool:
    """Check if the epilogue block is a broadcast pattern"""
    write_buffers = {r.tensor for r in sch.get(block).writes}
    epilogue_iters = {
        i.var: i for i in sch.get(epilogue).axes if not (isinstance(i.domain.extent, ir.IntImm) and int(i.domain.extent) == 1)
    }
    for buffer_region in sch.get(epilogue).reads:
        if buffer_region.tensor not in write_buffers:
            continue
        tir_vars = collect_block_iter_vars_used_in_access_region(sch.get(epilogue), buffer_region.region)
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False


def get_reduction_blocks(sch: NativeSchedule, blocks: list[StageRef]) -> list[StageRef]:
    # Get the main computation block
    def is_reduction(block: StageRef) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.kind for iter_var in block_stmt.axes}
        return iter_types == {"R", "S"}

    def is_spatial(block: StageRef) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.kind for iter_var in block_stmt.axes}
        return iter_types == {"S"}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) == 0:
        return None
    return reduction_blocks


def get_coalesced_veclen(block_stmt: StageView, target_bits: int = 128) -> int:
    # gpu memory prefer 128 bits coalesced access (e.g. four banks)
    # 128 bits
    buffers: list[TensorView] = []
    for read in block_stmt.reads:
        buffers.append(read.tensor)
    for write in block_stmt.writes:
        buffers.append(write.tensor)
    # pick the dtype with the largest bits
    max_dtype_bits: int = 0
    for buffer in buffers:
        max_dtype_bits = max(max_dtype_bits, DataType(buffer.dtype).bits)
    return target_bits // max_dtype_bits
