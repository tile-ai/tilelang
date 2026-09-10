"""TileIR pass infrastructure — passes package."""

from .base import Pass, PassContext, run_pipeline, walk_block
from .dataflow import (
    DataPredicate,
    DataflowResult,
    dataflow_analysis,
    dataflow_pass,
    ALIAS_EMPTY,
    ALIAS_UNIVERSE,
)
from .token_order import TokenPlan, token_order_pass
from .loop_carry import loop_carry_pass
from .gemm_orientation import gemm_orientation_pass

__all__ = [
    "loop_carry_pass",
    "gemm_orientation_pass",
    "Pass",
    "PassContext",
    "run_pipeline",
    "walk_block",
    # dataflow analysis
    "DataPredicate",
    "DataflowResult",
    "dataflow_analysis",
    "dataflow_pass",
    "ALIAS_EMPTY",
    "ALIAS_UNIVERSE",
    # token ordering
    "TokenPlan",
    "token_order_pass",
]
