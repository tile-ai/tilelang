# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from tvm_tirx import ir, tirx
from tvm_tirx.ir import Range, Expr, Var
from tvm_tirx.tirx import IndexMap
from ._native_analysis import NativeSchedule, StageRef, StageView, TensorView, TensorAccess, pre_order_visit
from tvm_tirx.tirx.analysis import undefined_vars
from .analysis import (
    collect_block_iter_vars_used_in_access_region,
    get_root_block,
    get_reduction_blocks,
)
from tvm_tirx.target import Target
from .arch import get_arch, is_tensorcore_supported_precision
from .arch.cuda import check_sm_version
from .arch.rdna import _get_rdna_tuning_config
from tilelang.rocm.target import target_get_mcpu, target_is_rdna
import logging

logger = logging.getLogger(__name__)


def collect_vars_from_expr(prim_expr):
    vars = []

    def callback(node):
        if isinstance(node, Var):
            vars.append(node)
        return True

    pre_order_visit(prim_expr, callback)

    return vars


def _is_one(x: Expr) -> bool:
    return isinstance(x, ir.IntImm) and x.value == 1


def _collect_producers(sch: NativeSchedule, block: StageRef, visited=None):
    visited = {block.uid} if visited is None else visited
    result = []
    for producer in sch.get_producers(block):
        if producer.uid in visited:
            continue
        visited.add(producer.uid)
        result.append(producer)
        result.extend(_collect_producers(sch, producer, visited))
    return result


def _collect_consumers(sch: NativeSchedule, block: StageRef, visited=None):
    visited = {block.uid} if visited is None else visited
    result = []
    for consumer in sch.get_consumers(block):
        if consumer.uid in visited:
            continue
        visited.add(consumer.uid)
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer, visited))
    return result


def auto_inline_producers(
    sch: NativeSchedule,
    block: StageRef,
    skip_blocks: list[StageRef] | None = None,
):
    skip_ids = {item.uid for item in (skip_blocks or [])}
    with sch.transaction():
        while True:
            for producer in _collect_producers(sch, block):
                if producer.uid in skip_ids:
                    continue
                try:
                    sch.compute_inline(producer)
                except Exception:
                    continue
                break
            else:
                return


def auto_inline_consumers(
    sch: NativeSchedule,
    block: StageRef,
):
    with sch.transaction():
        while True:
            changed = False
            for operation in (sch.compute_inline, sch.reverse_compute_inline):
                for consumer in _collect_consumers(sch, block):
                    try:
                        operation(consumer)
                    except Exception:
                        continue
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                return


def auto_inline_consumer_chain(
    sch: NativeSchedule,
    block: StageRef,
):
    with sch.transaction():
        auto_inline_consumers(sch, block)
        remaining_consumers = sch.get_consumers(block)
        if remaining_consumers:
            for consumer in remaining_consumers:
                for producer in sch.get_producers(consumer):
                    if producer.uid != block.uid:
                        sch.compute_inline(producer)
            auto_inline_consumers(sch, block)


# used to match the similar region with dequantize op.
def find_first_similar_region(regions: list[TensorAccess], buffer: TensorView):
    for region in regions:
        if len(region.tensor.shape) == len(buffer.shape):
            return region
    return None


# used to match the similar buffer with dequantize op.
def find_first_similar_buffer(regions: list[TensorAccess], buffer: TensorView):
    for region in regions:
        if len(region.tensor.shape) == len(buffer.shape):
            return region.tensor
    return None


# find the block that required to be reindex and scope.
def find_last_producer_from_buffer(sch, main_block, buffer: TensorView) -> StageRef | None:
    # block that most near to the arguments
    block = main_block
    buffer = buffer

    while True:
        last_buffer = buffer
        producers = sch.get_producers(block)

        if len(producers) == 0:
            # do not have any producer means it is the first block
            break

        for producer in producers:
            for write in sch.get(producer).writes:
                if write.tensor == buffer:
                    block = producer
                    buffer = find_first_similar_buffer(sch.get(producer).reads, last_buffer)
        if buffer == last_buffer:
            break
    return block


def find_arg_idx_from_buffer_chain(sch: NativeSchedule, main_block: StageRef, buffer: TensorView) -> int:
    """traverse to find the arg index from the buffer"""
    producers = sch.get_producers(main_block)

    # a head buffer has no producer blocks
    def find_args_index(sch: NativeSchedule, buffer: TensorView):
        for i, param in enumerate(sch.mod["main"].params):
            if param.same_as(buffer.tensor):
                return i
        return None

    is_head_buffer = len(producers) == 0
    if is_head_buffer:
        return find_args_index(sch, buffer)
    for block in sch.get_producers(main_block):
        if len(sch.get(block).reads) != 1 or len(sch.get(block).writes) != 1:
            continue
        for write in sch.get(block).writes:
            if write.tensor == buffer:
                return find_arg_idx_from_buffer_chain(sch, block, sch.get(block).reads[0].tensor)

    # if no buffer producer block found, it means the buffer is an input buffer
    return find_args_index(sch, buffer)


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: Expr


def make_iter_fusion_index_map(
    traits: list[IterTrait],
    kind_order: list[IterKind],
) -> tirx.IndexMap:
    fused_iters: dict[IterKind, Expr] = {}
    input_iters: list[ir.Var] = []
    for i, trait in enumerate(traits):
        v_i = ir.Var(f"i{i}", ir.PrimType(trait.extent.ty.dtype))
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: list[ir.Expr] = [fused_iters.get(kind, ir.IntImm(traits[0].extent.ty.dtype, 0)) for kind in kind_order]

    return tirx.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: StageView) -> tuple[list[IterTrait]] | None:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : StageView
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: list[Range]) -> set[Var]:
        axes: set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.begin)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: dict[Var, IterTrait] = {}
    for iter_var in block.axes:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.domain.extent):
            if iter_var.kind == "R":
                # for simplified case (e.g. 1x1 conv kernel)
                kind = IterKind.kIter_K
            else:
                kind = IterKind.kIter_T
        elif iter_var.kind == "S":
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.kind == "R":
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.domain.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.axes if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.axes if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.axes if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.axes]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: StageView, layout: list[str] | None = None) -> tuple[tirx.IndexMap, ...] | None:
    """Get index maps for the block

    Parameters
    ----------
    block : StageView
        The block to be analyzed

    layout : List[str]
        the target layout index map to be used.
        'n' for [i, k] layout
        't' for [k, j] layout
        'a' for auto inference based on whether the last axis is reduction.

    Returns
    -------
    index_maps : Optional[Tuple[tirx.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    if layout is None:
        layout = ["n", "t", "n"]
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    def get_ordered_axes(region: list[Range]) -> set[Var]:
        axes: list[Var] = []
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes.append(r.begin)
        return axes

    def is_common_reduce(var: Var) -> bool:
        return any(iter_var.var == var and iter_var.kind == "R" for iter_var in block.axes)

    def has_common_reduce(var: Var) -> bool:
        vars = collect_vars_from_expr(var)
        return any(is_common_reduce(v) for v in vars)

    def check_last_trait(region: list[Range]):
        axes = get_ordered_axes(region)
        return has_common_reduce(axes[-1])

    def infer_layout(layout: str, region: list[Range], kind: str = "A"):
        """
        Infer the layout based on the region and the kind of buffer
        kind: "A", "B", "C"
        """
        primary_iter, secondary_iter, reduction_iter = {
            "A": (IterKind.kIter_I, IterKind.kIter_K, IterKind.kIter_K),
            "B": (IterKind.kIter_K, IterKind.kIter_J, IterKind.kIter_K),
            "C": (IterKind.kIter_I, IterKind.kIter_J, None),
        }[kind]

        spatial_iter = {
            "A": IterKind.kIter_I,
            "B": IterKind.kIter_J,
            "C": None,
        }[kind]

        if layout == "n":
            return [IterKind.kIter_S, primary_iter, secondary_iter]
        elif layout == "t":
            return [IterKind.kIter_S, secondary_iter, primary_iter]
        elif layout == "a":
            # auto inference layout
            # for buffer with reduction axis, we put it as the last axis
            # otherwise, we put it as the first axis
            if kind == "C":
                return [IterKind.kIter_S, primary_iter, secondary_iter]
            else:
                return (
                    [IterKind.kIter_S, spatial_iter, reduction_iter]
                    if check_last_trait(region)
                    else [IterKind.kIter_S, reduction_iter, spatial_iter]
                )
        else:
            raise ValueError(f"Unknown layout {layout}")

    A_index_map = make_iter_fusion_index_map(A_traits, infer_layout(layout[0], block.reads[0].region, kind="A"))
    B_index_map = make_iter_fusion_index_map(B_traits, infer_layout(layout[1], block.reads[1].region, kind="B"))
    C_index_map = make_iter_fusion_index_map(C_traits, infer_layout(layout[2], block.writes[0].region, kind="C"))

    matmul_index_map = make_iter_fusion_index_map(
        block_traits,
        [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K],
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_in_out_dtypes(block: StageView) -> tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].tensor.dtype
    out_dtype = block.writes[0].tensor.dtype
    return (in_dtype, out_dtype)


def get_dequantize_block(sch, blocks) -> StageRef | None:
    # check at least two input and one output
    # at lease one input has uint dtype, and the output dtype is float
    def is_dequantize(block: StageRef) -> bool:
        block_stmt = sch.get(block)
        if len(block_stmt.reads) < 2:
            return False
        has_uint_input = any("uint" in str(region.tensor.dtype) for region in block_stmt.reads)
        if not has_uint_input:
            return False
        return not (len(block_stmt.writes) != 1 or "float" not in str(block_stmt.writes[0].tensor.dtype))

    dequantize_blocks = [block for block in blocks if is_dequantize(block)]
    return dequantize_blocks[0] if len(dequantize_blocks) == 1 else None


def is_identity_or_transpose_block(block_stmt: StageView) -> bool:
    iter_types = {iter_var.kind for iter_var in block_stmt.axes}
    if iter_types != {"S"}:
        return False, False
    if not isinstance(block_stmt.body, ir.TensorStore):
        return False, False
    if not isinstance(block_stmt.body.value, ir.TensorLoad):
        return False, False

    def get_access_vars(region: list[Range]) -> list[Var]:
        axes: list[Var] = []
        for r in region:
            if not _is_one(r.extent):
                return None
            axes.extend(undefined_vars(r.begin))
        # remove trivial axis
        trivial_vars = set(iter_var.var for iter_var in block_stmt.axes if _is_one(iter_var.domain.extent))
        axes = [axis for axis in axes if axis not in trivial_vars]
        # remove duplicate axis
        axes = [var for i, var in enumerate(axes) if i == 0 or var != axes[i - 1]]
        return axes

    lhs_access_vars = get_access_vars(block_stmt.reads[0].region)[-2:]
    rhs_access_vars = get_access_vars(block_stmt.writes[0].region)[-2:]
    is_identity = list(lhs_access_vars) == list(rhs_access_vars)
    is_transpose = list(lhs_access_vars) != list(rhs_access_vars) and set(lhs_access_vars) == set(rhs_access_vars)
    return is_identity, is_transpose


def is_identity_block(block_stmt: StageView) -> bool:
    return is_identity_or_transpose_block(block_stmt)[0]


def is_transpose_block(block_stmt: StageView) -> bool:
    return is_identity_or_transpose_block(block_stmt)[1]


def inline_transpose_block(sch: NativeSchedule, blocks: list[StageRef]):
    result_blocks = []
    with sch.transaction():
        for block in blocks:
            if not is_transpose_block(sch.get(block)):
                result_blocks.append(block)
                continue
            try:
                sch.compute_inline(block)
            except Exception:
                try:
                    sch.reverse_compute_inline(block)
                except Exception:
                    result_blocks.append(block)
    return [sch.refresh(block) for block in result_blocks]


def normalize_to_matmul(sch: NativeSchedule, main_block: StageRef, layout: list[str] | None = None) -> NativeSchedule | None:
    if layout is None:
        layout = ["n", "t", "n"]
    block_stmt = sch.get(main_block)

    # let layout be 'a' to auto inference the layout
    index_maps = get_index_map(block_stmt, layout=layout)
    if index_maps is None:
        logger.debug("Cannot find the appropriate index map for tensorcore")
        return None

    matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

    # The three cache rewrites and loop rebind form one transaction: an
    # unsupported map must leave the original Function and handles intact.
    with sch.transaction():
        block = sch.reindex(main_block, ("read", 0), skip_simplify=True)
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1), skip_simplify=True)
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0), skip_simplify=True)
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)
        sch.function = sch.function.with_attr("dlight.tensorcore_prenormlized", True)
    return sch


def get_tensorized_func_and_tags(
    func: tirx.Function,
    target: Target,
    layout: list[str] | None = None,
    skip_normalize: bool = False,
    allow_gemv: bool = False,
) -> tuple[tirx.Function, dict[str, list[int] | int]]:
    """
    transform function to matmul if necessary (e.g. transform conv2d with im2col)
    """
    if layout is None:
        layout = ["a", "a", "a"]
    # step1. detect whether the function can utilize tensorcore
    sch = NativeSchedule(func)
    root_block = get_root_block(sch)
    blocks = sch.get_child_blocks(root_block)
    reduction_blocks = get_reduction_blocks(sch, blocks)
    if not reduction_blocks or len(reduction_blocks) != 1:
        return func, None

    def _can_be_tensorized(sch: NativeSchedule, block: StageRef) -> bool:
        block_stmt = sch.get(block)
        conditions = []
        conditions.append(len(block_stmt.reads) == 2)
        conditions.append(len(block_stmt.writes) == 1)
        conditions.append(len(collect_block_iter_vars_used_in_access_region(block_stmt, block_stmt.writes[0].region)) > 0)
        return all(conditions)

    # step2. transform function to tensorcore matmul (e.g. conv2d with im2col)
    def is_cuda_tensorcore_target(target: Target) -> bool:
        return target.kind.name == "cuda" and check_sm_version(target.attrs.get("arch", "")) >= 70

    def is_rdna_wmma_target(target: Target) -> bool:
        return target.kind.name == "hip" and target_is_rdna(target)

    def analysis_tensorcore_tags(sch: NativeSchedule, block: StageRef, target: Target) -> bool | dict:
        tags: dict[str, list[int] | int] = {}
        block_stmt = sch.get(block)

        # Tensorized schedules are supported by NVIDIA Tensor Cores and by RDNA
        # WMMA on HIP/gfx11+ targets.  Other targets fall back to the default
        # Roller policy.
        if not is_cuda_tensorcore_target(target) and not is_rdna_wmma_target(target):
            return False
        # analysis tensorcore axis
        # todo(lei): maybe we can remove this in the future
        (write_buffer_region,) = block_stmt.writes
        out_axis = len(write_buffer_region.tensor.shape)
        tags["tensorcore_config"] = [out_axis - 2, out_axis - 1]

        # analysis pipeline stage
        # todo(lei): maybe we can integrate this into policy in the future
        tags["pipeline_stage"] = 1
        if target.kind.name == "cuda" and check_sm_version(target.attrs.get("arch", "")) in {80, 90}:
            # enable pipeline stage only for sm_80 devices
            tags["pipeline_stage"] = 2
        elif is_rdna_wmma_target(target):
            tags["pipeline_stage"] = _get_rdna_tuning_config(target_get_mcpu(target)).pipeline_stage

        # analysis async copy
        # todo(lei): maybe we can integrate this into policy in the future
        tags["use_async_copy"] = False
        if target.kind.name == "cuda" and tags["pipeline_stage"] == 2 and check_sm_version(target.attrs.get("arch", "")) in {80, 90}:
            # async copy only works in software pipeline.
            tags["use_async_copy"] = True

        # analysis intrin information
        def get_ordered_axes(region: list[Range]) -> set[Var]:
            axes: list[Var] = []
            for r in region:
                if not _is_one(r.extent):
                    raise ValueError("Expect elemwise block access")
                axes.append(r.begin)
            return axes

        def is_common_reduce(var: Var) -> bool:
            return any(iter_var.var == var and iter_var.kind == "R" for iter_var in block_stmt.axes)

        def has_common_reduce(var: Var) -> bool:
            vars = collect_vars_from_expr(var)
            return any(is_common_reduce(v) for v in vars)

        def check_last_trait(region: list[Range]):
            axes = get_ordered_axes(region)
            return has_common_reduce(axes[-1])

        intrin_info: dict = {}
        in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
        intrin_info["in_dtype"] = in_dtype
        intrin_info["out_dtype"] = out_dtype

        if target.kind.name == "cuda" and 70 <= check_sm_version(target.attrs.get("arch", "")) < 80 and out_dtype == "int32":
            # INT32 Accum TensorCore only supports SM Version > 32.
            return False

        # if the last dimension is reduce axis, the B is transposed
        intrin_info["trans_b"] = check_last_trait(block_stmt.reads[1].region)
        if func.attrs is not None and "input_transform_kind" in func.attrs:
            intrin_info["input_transform_kind"] = func.attrs["input_transform_kind"]
        if func.attrs is not None and "weight_transform_kind" in func.attrs:
            intrin_info["weight_transform_kind"] = func.attrs["weight_transform_kind"]
        tags["intrin_info"] = intrin_info
        # Analysis Block Reduction Optimization
        # Currently, we only support block reduction depth 2 for small M
        # When the func is a dequantize like ops, we should consider the M
        require_block_reduce = False
        # And we only support float16 for now
        if hasattr(func.attrs, "dequantize_info") and in_dtype in ["bfloat16", "float16"]:
            for arg in func.params:
                if not tirx.is_tensor_var(arg):
                    continue
                inp_shape = arg.ty.shape
                M = inp_shape[0]
                if isinstance(M, ir.IntImm) and M <= 128:
                    require_block_reduce = True
                break
        if target.kind.name == "cuda" and require_block_reduce and check_sm_version(target.attrs.get("arch", "")) == 80:
            tags["block_reduction_depth"] = 2
        return tags

    (main_block,) = reduction_blocks
    if not _can_be_tensorized(sch, main_block):
        return func, None

    block_stmt = sch.get(main_block)
    if is_cuda_tensorcore_target(target) or is_rdna_wmma_target(target):
        in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
        if is_cuda_tensorcore_target(target):
            if not is_tensorcore_supported_precision(in_dtype, out_dtype, arch=get_arch(target)):
                logger.debug(f"The input and output dtype ({in_dtype}, {out_dtype})is not supported by tensorcore")
                return func, None
        elif is_rdna_wmma_target(target) and (str(in_dtype), str(out_dtype)) not in {("float16", "float32")}:
            logger.debug(f"The input and output dtype ({in_dtype}, {out_dtype})is not supported by RDNA WMMA")
            return func, None

        # reindex and transform functions
        # Normalize tensor functions to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # or C[S, I, J] += A[S, I, K] * B[S, K, J]
        # skip normalize when we want to detect tags only.
        if not skip_normalize:
            sch = normalize_to_matmul(sch, main_block, layout)
            if sch is None:
                return func, None

        main_block = sch.refresh(main_block)
        block_stmt = sch.get(main_block)

        # 16 for 16 bits tensor core while 32 for 8bits tensorcore.
        minimal_tensorize_spatial_threshold = 16
        minimal_tensorize_reduce_threshold = 16 if in_dtype in ["bfloat16", "float16"] else 32
        # the batch dimension is not taken into consideration.
        for item_var in block_stmt.axes[1:]:
            extent = item_var.domain.extent
            iter_type = item_var.kind

            if iter_type == "S":
                minimal_tensorize_threshold = minimal_tensorize_spatial_threshold
            elif iter_type == "R":
                minimal_tensorize_threshold = minimal_tensorize_reduce_threshold
            else:
                raise ValueError(f"Unknown IterVar type {iter_type}")

            if isinstance(extent, ir.IntImm) and extent.value < minimal_tensorize_threshold:
                return func, None
        tags = analysis_tensorcore_tags(sch, main_block, target)
        return sch.mod["main"], tags

    return func, None


def get_propagate_map(trans: bool = True, dtype="float16", matrix_name="A", index_dtype="int32"):
    from tilelang.cuda.intrinsics.layout.mma_layout import (  # pylint: disable=import-outside-toplevel
        ldmatrix_32x8_to_shared_16x16_layout,
        ldmatrix_trans_32x8_to_shared_16x16_layout,
        ldmatrix_32x16_to_shared_16x32_layout_a,
        ldmatrix_32x16_to_shared_16x32_layout_b,
    )

    assert dtype in [
        "bfloat16",
        "float16",
        "int8",
        "float8_e4m3",
        "float8_e5m2",
    ], "Only support bfloat16, float16, int8, float8_e4m3, float8_e5m2"
    # TODO(lei): actually should analyze based on bits instead of dtype
    if dtype in ["bfloat16", "float16"]:
        ldmatrix_layout = ldmatrix_32x8_to_shared_16x16_layout
        ldmatrix_layout_trans = ldmatrix_trans_32x8_to_shared_16x16_layout
    elif dtype in ["int8", "float8_e4m3", "float8_e5m2"]:
        # int8 mma only support 32x16 to 16x32 layout
        if matrix_name == "A" and trans is False:
            ldmatrix_layout = ldmatrix_32x16_to_shared_16x32_layout_a
        elif matrix_name == "B" and trans is True:
            ldmatrix_layout = ldmatrix_32x16_to_shared_16x32_layout_b
        else:
            raise ValueError("Unknown matrix name ", matrix_name)

    # IntraWarp memory layout was occurred by ldmatrix, we should lift the ld_matrix out
    def ldmatrix_permutation_16x16_32x8_16x16(kernel_i, kernel_j):
        thread_id = kernel_i * 2 + kernel_j // 8
        local_id = kernel_j % 8
        return ldmatrix_layout(thread_id, local_id)

    def ldmatrix_trans_permutation_16x16_32x8_16x16(kernel_i, kernel_j):
        thread_id = kernel_i * 2 + kernel_j // 8
        local_id = kernel_j % 8
        return ldmatrix_layout_trans(thread_id, local_id)

    def ldmatrix_permutation_16x32_32x16_32x16(kernel_i, kernel_j):
        thread_id = kernel_i * 2 + kernel_j // 16
        local_id = kernel_j % 16
        return ldmatrix_layout(thread_id, local_id)

    if dtype in ["bfloat16", "float16"]:
        ldmatrix_index_map = ldmatrix_trans_permutation_16x16_32x8_16x16 if trans else ldmatrix_permutation_16x16_32x8_16x16
    else:
        ldmatrix_index_map = ldmatrix_permutation_16x32_32x16_32x16

    ldmatrix_index_map = IndexMap.from_func(ldmatrix_index_map, index_dtype=index_dtype)
    # TODO(lei): index_dtype should be analyzed from the schedule
    row, col = [16, 16] if dtype in ["bfloat16", "float16"] else [16, 32]
    inversed_index_map = ldmatrix_index_map.inverse([row, col])
    return ldmatrix_index_map, inversed_index_map


# This function is used to get the index map for the stage3 of the
# Ladder weight propagation, which can be used to avoid the ldmatrix
# Instructions.
def get_ladder_stage3_map(dtype="float16", index_dtype="int32"):
    def shared_32x8_to_mma_32x8_layout(i, j):
        thread_id = (i % 8) * 4 + (j // 2)
        local_id = (i // 8) * 2 + (j % 2)
        return thread_id, local_id

    def shared_32x16_to_mma_32x16_layout(i, j):
        thread_id = (i % 8) * 4 + (j // 4)
        local_id = (i // 8) * 4 + (j % 4)
        return thread_id, local_id

    assert dtype in [
        "bfloat16",
        "float16",
        "int8",
        "float8_e4m3",
        "float8_e5m2",
    ], "Only support float16, int8, float8_e4m3, float8_e5m2"
    if dtype in ["bfloat16", "float16"]:
        stage3_layout = shared_32x8_to_mma_32x8_layout
    elif dtype in ["int8", "float8_e4m3", "float8_e5m2"]:
        stage3_layout = shared_32x16_to_mma_32x16_layout
    else:
        raise ValueError("Unknown dtype ", dtype)

    # IntraWarp memory layout was occurred by ldmatrix, we should lift the ld_matrix out
    def ladder_stage3_permutation_16x16_32x8_32x8_16x16(kernel_i, kernel_j):
        thread_id = kernel_i * 2 + kernel_j // 8
        local_id = kernel_j % 8
        new_thread_id, new_local_id = stage3_layout(thread_id, local_id)
        new_kernel_i = (new_thread_id * 8 + new_local_id) // 16
        new_kernel_j = (new_thread_id * 8 + new_local_id) % 16
        return new_kernel_i, new_kernel_j

    def ladder_stage3_permutation_16x32_32x16_32x16_16x32(kernel_i, kernel_j):
        thread_id = kernel_i * 2 + kernel_j // 16
        local_id = kernel_j % 16
        new_thread_id, new_local_id = stage3_layout(thread_id, local_id)
        new_kernel_i = (new_thread_id * 16 + new_local_id) // 32
        new_kernel_j = (new_thread_id * 16 + new_local_id) % 32
        return new_kernel_i, new_kernel_j

    if dtype in ["bfloat16", "float16"]:
        stage3_index_map = ladder_stage3_permutation_16x16_32x8_32x8_16x16
    else:
        stage3_index_map = ladder_stage3_permutation_16x32_32x16_32x16_16x32

    stage3_index_map = IndexMap.from_func(stage3_index_map, index_dtype=index_dtype)
    # TODO(lei): index_dtype should be analyzed from the schedule
    row, col = [16, 16] if dtype in ["bfloat16", "float16"] else [16, 32]
    inversed_index_map = stage3_index_map.inverse([row, col])
    return stage3_index_map, inversed_index_map


def layout_propagate_chain(
    sch: NativeSchedule,
    start_block: StageRef,
    start_buffer: TensorView,
    end_block: StageRef,
    index_map: IndexMap,
):
    # some layout transformation may only apply to the last n dimensions
    # propagate the layout transformation to the chain of blocks
    block = start_block
    buffer = start_buffer
    index_map = index_map
    while True:
        last_buffer = buffer
        producers = sch.get_producers(block)
        if len(producers) == 0:
            break
        for producer in producers:
            if len(sch.get(producer).writes) != 1:
                return index_map
            if producer.uid == end_block.uid:
                return index_map
            (write,) = sch.get(producer).writes

            read = find_first_similar_region(sch.get(producer).reads, last_buffer)
            if write.tensor == buffer:
                block = producer
                buffer = read.tensor
                write_indices = [r.begin for r in write.region]
                read_indices = [r.begin for r in read.region]
                # reverse index map from [vi // x] -> [vi * x] to match the inconsistent layout
                tmp_index_map = IndexMap(write_indices, read_indices, None)
                tmp_index_map = tmp_index_map.non_surjective_inverse(write.tensor.shape)[0]

                # if dequantize like ops are used, the scaling factor should be considered
                # to be applied to the final indices
                scaling_factor = 1
                for i, j in zip(write.tensor.shape, read.tensor.shape):
                    scaling_factor *= i // j
                final_indices = list(index_map.map_indices(tmp_index_map.map_indices(write_indices)))
                final_indices[-1] = final_indices[-1] // scaling_factor
                index_map = IndexMap(
                    write_indices,
                    final_indices,
                    None,
                )
        if buffer == last_buffer:
            break
    return index_map
