from tvm_tirx.ir import Node, Scriptable
import tvm_ffi
from tvm_tirx.target import Target
from tilelang import _ffi_api
from tilelang._compat.location import (
    make_span as make_span,
    set_stmt_span as set_stmt_span,
    get_stmt_span as get_stmt_span,
    set_buffer_span as set_buffer_span,
    get_buffer_span as get_buffer_span,
    set_prim_func_span as set_prim_func_span,
    get_prim_func_span as get_prim_func_span,
    stamp_stmt_spans as stamp_stmt_spans,
    span_to_location as span_to_location,
    span_coverage as span_coverage,
)


@tvm_ffi.register_object("tl.Fill")
class Fill(Node, Scriptable): ...


@tvm_ffi.register_object("tl.AtomicAdd")
class AtomicAdd(Node, Scriptable): ...


@tvm_ffi.register_object("tl.Copy")
class Copy(Node, Scriptable): ...


@tvm_ffi.register_object("tl.Im2Col")
class Im2ColOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.GemmWarpPolicy")
class GemmWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


@tvm_ffi.register_object("tl.GemmSPWarpPolicy")
class GemmSPWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmSPWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


@tvm_ffi.register_object("tl.FinalizeReducerOp")
class FinalizeReducerOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ParallelOp")
class ParallelOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ReduceOp")
class ReduceOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.CumSumOp")
class CumSumOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.CumMaxOp")
class CumMaxOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ReduceType")
class ReduceType(Node, Scriptable): ...


@tvm_ffi.register_object("tl.Tcgen05Meta")
class Tcgen05Meta(Node, Scriptable):
    """One tcgen05.ld/st data-movement shape: the per-warp single-issue
    CUTLASS TV atom plus the wrapper's chaining limit."""

    intrinsics_name: str
    max_chunks: int


@tvm_ffi.register_object("tl.Tcgen05CopyPlan")
class Tcgen05CopyPlan(Node, Scriptable):
    """The tiled copy of one TMEM tile built by tl.ExpandTcgen05Layout."""

    num_chunks_each_wg: int
    num_issues: int
    vals_per_issue: int
