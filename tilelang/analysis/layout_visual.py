from __future__ import annotations

from collections.abc import Sequence
import warnings

import tilelang.language as T
from tilelang.layout import PartialFragment
import tvm_ffi
from tvm_tirx import tirx

from ._visitor import NativeStmtVisitor, tilelang_metadata


def print_fragment_format(layout: T.Fragment) -> None:
    """
    Format fragment layout information into a human-readable string.

    Parameters
    ----------
    layout : T.Fragment
        The fragment layout to format

    Returns
    -------
    str
        Formatted string showing shape, thread mapping, and index mapping
    """
    if isinstance(layout, T.Fragment):
        input_shape = layout.get_input_shape()
        output_shape = layout.get_output_shape()
        # A reducer partial's replicas are addends awaiting the finalize
        # collective, not equal copies — label them to avoid misreading.
        rep_suffix = " (partials)" if isinstance(layout, PartialFragment) else ""
        lines = [
            f"  Shape: {input_shape} -> {output_shape}",
            f"  Thread: {layout.thread}",
            f"  Index:  {layout.get_forward_index()}",
            f"  Replicate:  {layout.replicate_size}{rep_suffix}",
        ]
        print("\n".join(lines))
    else:
        raise ValueError(f"Expected T.Fragment, but got {type(layout).__name__}")


class _LayoutVisualVisitor(NativeStmtVisitor):
    """
    User-friendly pass which visualizes fragment layouts inferred during compilation.

    In TileLang, Fragment layouts describe:
    - How logical indices (e.g., [i, j]) map to thread IDs
    - How logical indices map to register file locations within each thread
    - The shape transformation from input dimensions to output dimensions

    This pass generates two types of output:
    1. Textual output: A human-readable description printed to console
    2. Visual diagrams: Color-coded plots saved to files (PDF, PNG, SVG formats)

    Configuration:
    The pass is controlled by the TL_ENABLE_LAYOUT_VISUALIZATION configuration option.
    The configuration accepts string values:

    - Empty string or not set: Pass does nothing (default, disabled)
    - "png": Generate PNG format only (recommended for quick inspection)
    - "pdf": Generate PDF format only (recommended for documentation)
    - "svg": Generate SVG format only (recommended for web/vector graphics)
    - "all": Generate all formats (PDF, PNG, SVG)
    - "png,svg": Generate multiple formats (comma-separated)
    """

    def __init__(self, formats: str | Sequence[str] = ""):
        super().__init__()
        if formats is None:
            parsed: list[str] = []
        elif isinstance(formats, str):
            formats_str = formats.strip()
            if formats_str == "":
                parsed = []
            elif formats_str == "all":
                parsed = ["pdf", "png", "svg"]
            else:
                parsed = [f.strip() for f in formats_str.split(",") if f.strip()]
        else:
            parsed = [str(f).strip() for f in formats if str(f).strip()]
        self.formats_list = [f for f in parsed if f != "txt"]
        self.printed = {}

    def _print_metadata(self, metadata):
        if metadata is None:
            return
        if metadata.get("key") == "layout_map":
            layout_map = metadata["value"]
        else:
            layout_map = metadata.get("layout_map", {})
        for tensor, layout in layout_map.items():
            if not isinstance(layout, T.Fragment):
                continue
            previous = self.printed.get(tensor)
            if previous is not None and tvm_ffi.structural_equal(previous, layout):
                continue
            self.printed[tensor] = layout
            name = getattr(tensor, "name", str(tensor))
            print(f"{name} inferred layout:")
            print_fragment_format(layout)
            if self.formats_list and len(layout.get_input_shape()) != 2:
                warnings.warn(
                    f"Skip plotting {name} layout: input_shape={layout.get_input_shape()} is not 2D.",
                    stacklevel=2,
                )
                continue
            for fmt in self.formats_list:
                from tilelang.tools.plot_layout import plot_layout

                plot_layout(layout, name=f"{name}_layout", formats=fmt)

    def visit_call(self, node, visitor):
        self._print_metadata(tilelang_metadata(node))
        super().visit_call(node, visitor)

    def visit_region(self, node, visitor):
        self._print_metadata(tilelang_metadata(node))
        super().visit_region(node, visitor)


def LayoutVisual(formats: str | Sequence[str] = ""):
    """Print/plot the native lexical layout_map published by layout inference.

    TensorVar keys stay bound at the publication site, including annotations
    following local allocation Bind statements. The layout objects are read
    directly from the layout declaration's typed attributes.
    """

    def pass_fn(func, _mod, _ctx):
        _LayoutVisualVisitor(formats=formats).visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.LayoutVisual")
