"""Native TileLang reflected nodes and source-location helpers."""

from tvm_tirx import ir as _ir, tirx as _tirx
from tvm_tirx.ir import Node, SourceName, Location as Span, SourceLoc, UnknownLoc, CallSiteLoc, Scriptable
import tvm_ffi
from tvm_tirx.target import Target
from tilelang import _ffi_api


class Fill(Node, Scriptable): ...


class AtomicAdd(Node, Scriptable): ...


class Copy(Node, Scriptable): ...


class Im2ColOp(Node, Scriptable): ...


class GemmWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


class GemmSPWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmSPWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


class FinalizeReducerOp(Node, Scriptable): ...


class ParallelOp(Node, Scriptable): ...


class ReduceOp(Node, Scriptable): ...


class CumSumOp(Node, Scriptable): ...


class CumMaxOp(Node, Scriptable): ...


class ReduceType(Node, Scriptable): ...


# Public span-helper spellings remain available over native Location metadata.
# Native setters replace the field exactly; UnknownLoc is the cleared state.
# Registration of unrelated backend node types is delayed until requested.


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
    return SourceLoc(SourceName(file), line, 1, line, 1 << 20)


def set_stmt_span(stmt, span: Span) -> None:
    if stmt is not None:
        _span_ffi("SetStmtSpan")(stmt, UnknownLoc() if span is None else span)


def get_stmt_span(stmt) -> Span | None:
    if stmt is None:
        return None
    span = _span_ffi("GetStmtSpan")(stmt)
    return span if span is not None and not isinstance(span, UnknownLoc) else None


def set_buffer_span(buffer, span: Span) -> None:
    if buffer is not None:
        _span_ffi("SetBufferSpan")(buffer, UnknownLoc() if span is None else span)


def get_buffer_span(buffer) -> Span | None:
    if buffer is None:
        return None
    span = _span_ffi("GetBufferSpan")(buffer)
    return span if span is not None and not isinstance(span, UnknownLoc) else None


def set_prim_func_span(func, span: Span) -> None:
    if func is not None:
        _span_ffi("SetPrimFuncSpan")(func, UnknownLoc() if span is None else span)


def stamp_stmt_spans(stmt, span: Span | None) -> None:
    """Fill in `span` on every Stmt in the subtree whose span is undefined.

    Nodes that already carry a span are left untouched. Intended for passes
    that rebuild a user statement into a fresh subtree: stamping the result
    with the original statement's span keeps source-line information alive
    through lowering.
    """
    if stmt is None or span is None:
        return
    _span_ffi("StampSubtreeSpans")(stmt, span)


def get_prim_func_span(func) -> Span | None:
    if func is None:
        return None
    span = _span_ffi("GetPrimFuncSpan")(func)
    return span if span is not None and not isinstance(span, UnknownLoc) else None


def span_to_location(span: Span | None) -> tuple[str, int] | None:
    """Return the most specific known source of a native Location."""
    if isinstance(span, SourceLoc):
        return (span.source_name.name, span.start_line)
    if isinstance(span, CallSiteLoc):
        return span_to_location(span.callee) or span_to_location(span.caller)
    return None


def span_coverage(func) -> dict:
    """Count located statements and tensor definitions in a native Function.

    Tensor variables may be parameters, allocations or views. A single
    identity is counted once even when it has many uses or metadata references.
    Scalar and pointer variables are not tensor definitions.
    """
    stmts = [0, 0]
    buffers = [0, 0]
    seen = set()

    def count_tensor(tensor):
        if tensor in seen or not _tirx.is_tensor_var(tensor):
            return
        seen.add(tensor)
        buffers[1] += 1
        buffers[0] += not isinstance(tensor.loc, UnknownLoc)

    def visit(node):
        stmts[1] += 1
        stmts[0] += not isinstance(node.loc, UnknownLoc)
        if isinstance(node, _ir.Bind):
            count_tensor(node.var)

    if not func.attrs.get("declaration", False):
        tvm_ffi.structural_walk(func.body, (_ir.Stmt, visit), order="pre")
    for parameter in func.params:
        count_tensor(parameter)
    return {"stmts": stmts, "buffers": buffers}


class Tcgen05Meta(Node, Scriptable):
    """One tcgen05.ld/st data-movement shape: the per-warp single-issue
    CUTLASS TV atom plus the wrapper's chaining limit."""

    intrinsics_name: str
    max_chunks: int


class Tcgen05CopyPlan(Node, Scriptable):
    """The tiled copy of one TMEM tile built by tl.ExpandTcgen05Layout."""

    num_chunks_each_wg: int
    num_issues: int
    vals_per_issue: int


# Loading the location helpers must not force registration of every backend's
# optional reflected class. A requested class is still checked against the
# real compiler registry; no replacement type key or Python object is created.
_NATIVE_TYPES = {
    "Fill": ("tl.Fill", Fill),
    "AtomicAdd": ("tl.AtomicAdd", AtomicAdd),
    "Copy": ("tl.Copy", Copy),
    "Im2ColOp": ("tl.Im2Col", Im2ColOp),
    "GemmWarpPolicy": ("tl.GemmWarpPolicy", GemmWarpPolicy),
    "GemmSPWarpPolicy": ("tl.GemmSPWarpPolicy", GemmSPWarpPolicy),
    "FinalizeReducerOp": ("tl.FinalizeReducerOp", FinalizeReducerOp),
    "ParallelOp": ("tl.ParallelOp", ParallelOp),
    "ReduceOp": ("tl.ReduceOp", ReduceOp),
    "CumSumOp": ("tl.CumSumOp", CumSumOp),
    "CumMaxOp": ("tl.CumMaxOp", CumMaxOp),
    "ReduceType": ("tl.ReduceType", ReduceType),
    "Tcgen05Meta": ("tl.Tcgen05Meta", Tcgen05Meta),
    "Tcgen05CopyPlan": ("tl.Tcgen05CopyPlan", Tcgen05CopyPlan),
}
for _name in _NATIVE_TYPES:
    del globals()[_name]


def __getattr__(name):
    if name not in _NATIVE_TYPES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    type_key, cls = _NATIVE_TYPES[name]
    value = tvm_ffi.register_object(type_key)(cls)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _NATIVE_TYPES.keys())
