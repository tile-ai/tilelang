"""Shared helpers for the layout-inference verification harness.

Runs a native Function through the exact pass prefix LayoutInference sees in the
real pipeline (BindTarget -> MaterializeKernelLaunch -> LayoutInference)
and extracts the search result from the IR annotations the pass leaves
behind:

  - lexical ``tl.annotation`` layout_map: TensorVar -> Layout/Fragment
  - ``tl.loop.annotations`` region: the loop nest's Fragment

Layouts are snapshotted as STRUCTURED dicts (shapes, replicate extent,
thread extent, forward maps as expression strings), not repr strings:
golden diffs then point at the exact field that moved, and case checks
assert on fields instead of substring-matching a print format.
"""

from __future__ import annotations

import tilelang as tl
import tilelang.language as T  # noqa: F401  (cases import via common)
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.layout import Fragment
from tilelang.analysis._visitor import single_loop, tensor_scope, tilelang_metadata
import tvm_ffi
from tvm_tirx import ir, sym

# The two selection policies behind `tl.layout_cost_model`, by name.
COST_MODELS = ("register-count", "io-aware")


def _run_passes(prim_func, cost_model: str, target=None):
    """The exact pass prefix LayoutInference sees in the real pipeline."""
    if target is None:
        target = tvm.target.Target(determine_target("auto"))
    mod = tvm.IRModule({"main": prim_func})
    with tvm.target.Target(target):
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        with tvm.transform.PassContext(config={"tl.layout_cost_model": cost_model}):
            mod = tl.transform.LayoutInference()(mod)
    return mod["main"]


def run_layout_inference(prim_func, cost_model: str, target=None):
    """Run LayoutInference on `prim_func` and return the inferred layouts.

    Returns ``{"buffers": {name: layout_dict}, "loops": {key: layout_dict}}``
    where a loop key names the parallel nest by its loop vars and extents,
    e.g. ``"i,j[128x128]"``, and layout_dict is `layout_to_dict` output.
    """
    return extract_layouts(_run_passes(prim_func, cost_model, target))


def run_layout_inference_objects(prim_func, cost_model: str, target=None):
    """Like run_layout_inference, but returns the LIVE objects:
    ``{"buffers": {name: (TensorVar, Layout)}}`` — for consumers that need the
    layout expressions themselves (e.g. the CuTe experiment), not the
    structured snapshot."""
    func = _run_passes(prim_func, cost_model, target)
    buffers: dict[str, tuple] = {}

    def visit(node):
        for tensor, layout in _layout_map(node).items():
            buffers[tensor.name] = (tensor, layout)

    tvm_ffi.structural_walk(func.body, ((ir.Call, ir.RegionStmt), visit), order="post")
    return {"buffers": buffers}


def lower_and_extract_vector_widths(prim_func, target=None) -> dict[str, int]:
    """Fully lower under the IO-AWARE pass config and report, per global
    buffer, the widest vectorized access (in lanes) the final device TIR
    performs.

    This is the ground truth that anchors the cost model's vector-width
    beliefs: the model scores a layout assuming the vectorizer will emit a
    given width, and this function reads back what the vectorizer actually
    emitted for the layout that won.
    """
    if target is None:
        target = tvm.target.Target(determine_target("auto"))
    with tvm.target.Target(target), tvm.transform.PassContext(config={"tl.layout_cost_model": "io-aware"}):
        artifact = tl.lower(prim_func, target=target, enable_device_compile=False)
    widths: dict[str, int] = {}

    def visit(node):
        buf, lanes = None, 0
        if isinstance(node, ir.TensorLoad):
            buf, lanes = node.source, node.ty.dtype.lanes
        elif isinstance(node, ir.TensorStore):
            buf, lanes = node.dest, node.value.ty.dtype.lanes
        if buf is not None and tensor_scope(buf) == "global":
            widths[buf.name] = max(widths.get(buf.name, 1), lanes)

    for _, func in artifact.device_mod.functions.items():
        tvm_ffi.structural_walk(func.body, ((ir.TensorLoad, ir.TensorStore), visit), order="post")
    return widths


def _shape_list(arr) -> list:
    """Shape as ints where constant, strings where symbolic."""
    out = []
    for x in arr:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            out.append(str(x))
    return out


def layout_to_dict(layout) -> dict:
    """Structured, field-comparable snapshot of a Layout or Fragment."""
    info = {
        "kind": type(layout).__name__,
        "input_shape": _shape_list(layout.get_input_shape()),
        "output_shape": _shape_list(layout.get_output_shape()),
        "forward_index": [str(e) for e in layout.get_forward_index()],
    }
    if isinstance(layout, Fragment):
        info["replicate"] = int(layout.replicate_size)
        info["threads"] = int(layout.get_thread_size())
        info["forward_thread"] = str(layout.forward_thread)
        thread_range = layout.thread_range
        if thread_range is not None:
            info["thread_range"] = [int(thread_range.begin), int(thread_range.extent)]
    return info


def _loop_key(for_node) -> str:
    """Name a parallel nest by its loop vars and extents: ``i,j[128x128]``."""
    names, extents = [], []
    cur = for_node
    analyzer = sym.Analyzer()
    while cur is not None and cur.kind == ir.ForKind.PARALLEL:
        names.append(cur.var.name)
        step = analyzer.simplify(cur.step)
        if analyzer.can_prove(step > 0):
            extent = tvm.tirx.max(tvm.tirx.ceildiv(cur.end - cur.begin, step), 0)
        elif analyzer.can_prove(step < 0):
            extent = tvm.tirx.max(tvm.tirx.ceildiv(cur.begin - cur.end, -step), 0)
        else:
            raise ValueError("Cannot form a layout loop key without a proven nonzero step")
        extents.append(str(analyzer.simplify(extent)))
        cur = single_loop(cur.body)
    return f"{','.join(names)}[{'x'.join(extents)}]"


def _layout_map(node):
    """Read the explicit layout payload at its lexical publication site."""
    metadata = tilelang_metadata(node)
    if metadata is None:
        return {}
    if metadata.get("key") == "layout_map":
        return metadata["value"]
    return metadata.get("layout_map", {})


def extract_layouts(func) -> dict:
    buffers: dict[str, dict] = {}
    loops: dict[str, dict] = {}

    def visit(node):
        for tensor, layout in _layout_map(node).items():
            buffers[tensor.name] = layout_to_dict(layout)
        metadata = tilelang_metadata(node)
        loop = None
        loop_layout = None
        if isinstance(node, ir.RegionStmt) and metadata is not None and metadata.get("key") == "tl.loop.annotations":
            loop = single_loop(node.body)
            loop_layout = metadata["value"].get("parallel_loop_layout")
        elif isinstance(node, ir.For):
            loop = node
            loop_layout = node.annotations.get("parallel_loop_layout")
        if loop_layout is not None:
            if loop is None:
                raise ValueError("parallel_loop_layout metadata must enclose exactly one loop")
            key = _loop_key(loop)
            # Disambiguate identical nests (post-order visit is deterministic).
            if key in loops:
                suffix = 2
                while f"{key}#{suffix}" in loops:
                    suffix += 1
                key = f"{key}#{suffix}"
            loops[key] = layout_to_dict(loop_layout)

    tvm_ffi.structural_walk(func.body, ((ir.Call, ir.RegionStmt, ir.For), visit), order="post")
    return {"buffers": buffers, "loops": loops}
