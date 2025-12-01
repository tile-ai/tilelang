import tilelang
import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor

from tvm.tir.transform import prim_func_pass
from tilelang.tools.plot_layout import plot_layout


def print_fragment_format(layout: T.Fragment) -> str:
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
            f"  Shape: {input_shape} -> {output_shape}", f"  Thread: {layout.forward_thread}",
            f"  Index:  {layout.forward_index}"
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

    def __init__(self, formats: str = "png"):
        super().__init__()
        self.layout_found = []
        self.processed_layouts = set()
        self.formats = formats

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]

            for key, layout in layout_map.items():
                if isinstance(layout, T.Fragment):
                    layout_id = str(layout)
                    if layout_id not in self.processed_layouts:
                        print(f"{key} layout inference:")
                        print_fragment_format(layout)
                        plot_layout(layout, name=f"{key}_layout", formats=self.formats)
                        self.processed_layouts.add(layout_id)

        self.visit_stmt(op.body)


def LayoutVisual():

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        pass_ctx = tilelang.transform.get_pass_context()
        config_value = pass_ctx.config.get(
            tilelang.PassConfigKey.TL_ENABLE_LAYOUT_VISUALIZATION.value, "")

        config_str = str(config_value).strip().lower()

        _LayoutVisualVisitor(formats=config_str).visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
