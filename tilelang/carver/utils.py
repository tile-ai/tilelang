from __future__ import annotations
from tvm import tir, IRModule
from tvm.tir import PrimFunc
from .arch import TileDevice
from .roller.policy import TensorCorePolicy, DefaultPolicy
from .roller.hint import Hint
from .roller.node import OutputNode
from .matmul_analysis import get_tensorized_func_and_tags
import logging

logger = logging.getLogger(__name__)


def get_rasterization_code(pannel_width: int = 8) -> str:
    return f"""
        const int MAX_BLOCK_N = {pannel_width};
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
    """


def get_roller_hints_from_func(
    func_or_module: tir.PrimFunc | IRModule, arch: TileDevice, topk: int = 10, tensorcore_only: bool = False, allow_gemv: bool = False
) -> list[Hint] | None:
    func = None
    if isinstance(func_or_module, tir.PrimFunc):
        func = func_or_module
    elif isinstance(func_or_module, IRModule):
        func = retrieve_func_from_module(func_or_module)
    else:
        raise ValueError("Not supported type: ", type(func_or_module))

    assert func is not None, "The function should not be None"

    roller_hints = None
    if tensorcore_only:
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
            roller_hints = policy.emit_config(topk)
        else:
            roller_hints = None
    else:
        policy = DefaultPolicy.from_prim_func(func=func, arch=arch)
        tensorized_func = None
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy.from_prim_func(func=tensorized_func, arch=arch, tags=tags)
        roller_hints = policy.emit_config(topk)
    return roller_hints


def get_roller_hints_from_output_nodes(
    output_nodes: list[OutputNode], arch: TileDevice, topk: int = 10, extra_tags: list[str] | None = None
) -> list[Hint] | None:
    assert isinstance(output_nodes, list), "The input should be a list of functions."

    lints = []
    try:
        policy = TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=None)
        lints = policy.emit_config(topk)
    except Exception as e_msg:
        logger.debug(f"Generate hints from output nodes failed: {e_msg}", "fallback to default policy")

    if len(lints) == 0:
        policy = DefaultPolicy.from_output_nodes(output_nodes, arch=arch, tags=None)
        lints = policy.emit_config(topk)
    return lints


def retrieve_func_from_module(ir_module: IRModule) -> PrimFunc:
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, "The optimized module should only have one global variable for default schedule."
    func = list(ir_module.functions.values())[0]
    return func
