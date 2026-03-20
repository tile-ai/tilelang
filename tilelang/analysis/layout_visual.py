from __future__ import annotations

from collections.abc import Sequence
import warnings

import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor

from tvm.tir.transform import prim_func_pass
from tilelang.tools.plot_layout import plot_layout


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
        lines = [
            f"  Shape: {input_shape} -> {output_shape}",
            f"  Thread: {layout.forward_thread}",
            f"  Index:  {layout.forward_index}",
            f"  Replicate:  {layout.replicate_size}",
        ]
        print("\n".join(lines))
    else:
        raise ValueError(f"Expected T.Fragment, but got {type(layout).__name__}")


@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
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

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]

            for key, layout in layout_map.items():
                if isinstance(layout, T.Fragment):
                    print(f"{key} inferred layout:")
                    print_fragment_format(layout)
                    for fmt in self.formats_list:
                        input_shape = layout.get_input_shape()
                        if len(input_shape) != 2:
                            warnings.warn(
                                f"Skip plotting {key} layout: input_shape={input_shape} is not 2D.",
                                stacklevel=2,
                            )
                            continue
                        plot_layout(layout, name=f"{key}_layout", formats=fmt)


def LayoutVisual(formats: str | Sequence[str] = ""):
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor(formats=formats).visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
