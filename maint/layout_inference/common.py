"""Shared helpers for the layout-inference verification harness.

Runs a PrimFunc through the exact pass prefix LayoutInference sees in the
real pipeline (BindTarget -> MaterializeKernelLaunch -> LayoutInference)
and extracts the search result from the IR annotations the pass leaves
behind:

  - SBlock annotation ``layout_map``:  Buffer -> Layout/Fragment
  - For annotation ``parallel_loop_layout``: the loop nest's Fragment

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
from tvm.tirx.stmt_functor import post_order_visit

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
    ``{"buffers": {name: (Buffer, Layout)}}`` — for consumers that need the
    layout expressions themselves (e.g. the CuTe experiment), not the
    structured snapshot."""
    func = _run_passes(prim_func, cost_model, target)
    buffers: dict[str, tuple] = {}

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and "layout_map" in node.annotations:
            for buf, layout in node.annotations["layout_map"].items():
                buffers[buf.name] = (buf, layout)

    post_order_visit(func.body, visit)
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
        if isinstance(node, tvm.tirx.BufferLoad):
            buf, lanes = node.buffer, node.dtype.lanes
        elif isinstance(node, tvm.tirx.BufferStore):
            buf, lanes = node.buffer, node.value.dtype.lanes
        if buf is not None and buf.scope() == "global":
            widths[buf.name] = max(widths.get(buf.name, 1), lanes)

    for _, func in artifact.device_mod.functions.items():
        post_order_visit(func.body, visit)
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
            info["thread_range"] = [int(thread_range.min), int(thread_range.extent)]
    return info


def _loop_key(for_node) -> str:
    """Name a parallel nest by its loop vars and extents: ``i,j[128x128]``."""
    names, extents = [], []
    cur = for_node
    while cur is not None and cur.kind == tvm.tirx.ForKind.PARALLEL:
        names.append(cur.loop_var.name)
        extents.append(str(cur.extent))
        cur = cur.body if isinstance(cur.body, tvm.tirx.For) else None
    return f"{','.join(names)}[{'x'.join(extents)}]"


def extract_layouts(func) -> dict:
    buffers: dict[str, dict] = {}
    loops: dict[str, dict] = {}

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and "layout_map" in node.annotations:
            for buf, layout in node.annotations["layout_map"].items():
                buffers[buf.name] = layout_to_dict(layout)
        elif isinstance(node, tvm.tirx.For) and "parallel_loop_layout" in node.annotations:
            key = _loop_key(node)
            # Disambiguate identical nests (post-order visit is deterministic).
            if key in loops:
                suffix = 2
                while f"{key}#{suffix}" in loops:
                    suffix += 1
                key = f"{key}#{suffix}"
            loops[key] = layout_to_dict(node.annotations["parallel_loop_layout"])

    post_order_visit(func.body, visit)
    return {"buffers": buffers, "loops": loops}
