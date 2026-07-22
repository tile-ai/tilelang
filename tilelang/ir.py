from tilelang import tvm as tvm
from tvm.ir.base import Node, SourceName, Span
from tvm.runtime import Scriptable
import tvm_ffi
from tvm.target import Target
from tilelang import _ffi_api


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


@tvm_ffi.register_object("tl.RegionOp")
class RegionOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ReduceType")
class ReduceType(Node, Scriptable): ...


# ---------------------------------------------------------------------------
# Source span helpers
#
# tirx Stmt/Buffer/PrimFunc nodes carry a mutable `span` field that is
# reflected read-only to Python and never participates in structural
# equality/hashing. The functions below write spans during script parsing
# (see tilelang/language/eager/builder.py) and read them back from
# diagnostics, the LSP analyzer, and visualization tools.
# ---------------------------------------------------------------------------


_span_ffi_cache: dict = {}


def _span_ffi(name: str):
    # `tl.ir.*` names contain a dot and are therefore skipped by
    # `init_ffi_api`; fetch them from the global registry directly. Cache the
    # lookup: the stamping hot path calls this per IR node.
    f = _span_ffi_cache.get(name)
    if f is None:
        f = tvm_ffi.get_global_func(f"tl.ir.{name}")
        _span_ffi_cache[name] = f
    return f


def make_span(file: str, line: int) -> Span:
    """Create a span covering a whole source line."""
    return Span(SourceName(file), line, line, 1, 1 << 20)


def set_stmt_span(stmt, span: Span) -> None:
    _span_ffi("SetStmtSpan")(stmt, span)


def get_stmt_span(stmt) -> Span | None:
    span = _span_ffi("GetStmtSpan")(stmt)
    return span if span is not None and span.source_name is not None else None


def set_buffer_span(buffer, span: Span) -> None:
    _span_ffi("SetBufferSpan")(buffer, span)


def get_buffer_span(buffer) -> Span | None:
    span = _span_ffi("GetBufferSpan")(buffer)
    return span if span is not None and span.source_name is not None else None


def set_prim_func_span(func, span: Span) -> None:
    _span_ffi("SetPrimFuncSpan")(func, span)


def get_prim_func_span(func) -> Span | None:
    span = _span_ffi("GetPrimFuncSpan")(func)
    return span if span is not None and span.source_name is not None else None


def span_to_location(span: Span | None) -> tuple[str, int] | None:
    """Convert a span to a (file, line) tuple, or None when undefined."""
    if span is None or span.source_name is None:
        return None
    return (span.source_name.name, span.line)


def span_coverage(func) -> dict:
    """Statistics on span injection coverage for a PrimFunc.

    Returns {"stmts": [with_span, total], "buffers": [with_span, total]},
    where buffers cover both function parameters (buffer_map) and block-
    allocated buffers (SBlock.alloc_buffers, i.e. T.alloc_* sites).
    New nodes synthesized by passes legitimately have no span; this metric
    is meant for the freshly parsed IR (span injection validation).
    """
    from tvm.tirx.stmt_functor import post_order_visit

    stmts = [0, 0]
    seen_buffers = []

    def count_buffer(buffer):
        if any(buffer.same_as(b) for b in seen_buffers):
            return
        seen_buffers.append(buffer)
        buffers[1] += 1
        if get_buffer_span(buffer) is not None:
            buffers[0] += 1

    buffers = [0, 0]

    def _visit(node):
        if isinstance(node, tvm.tirx.Stmt):
            stmts[1] += 1
            if get_stmt_span(node) is not None:
                stmts[0] += 1
        if type(node).__name__ == "SBlockRealize":
            for buf in node.block.alloc_buffers:
                count_buffer(buf)

    post_order_visit(func.body, _visit)

    for _, buffer in func.buffer_map.items():
        count_buffer(buffer)
    return {"stmts": stmts, "buffers": buffers}
