import tvm
import tilelang
import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor

from tvm.tir.transform import prim_func_pass
from tilelang.tools.plot_layout import plot_layout

def print_layout_format(layout: T.Fragment) -> str:
    input_shape = layout.get_input_shape()
    output_shape = layout.get_output_shape()
    lines = [
        f"  Shape: {input_shape} -> {output_shape}",
        f"  Thread: {layout.forward_thread}",
        f"  Index:  {layout.forward_index}"
    ]

    return "\n".join(lines)

@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.layout_found = []
        self.processed_layouts = set()
    
    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]

            for key, layout in layout_map.items():
                if isinstance(layout, T.Fragment):
                    layout_id = str(layout)
                    if layout_id not in self.processed_layouts:
                        print(f"{key} layout inference:")
                        print(print_layout_format(layout))
                        plot_layout(layout, name=f"{key}_layout")
                        self.processed_layouts.add(layout_id)
       
        self.visit_stmt(op.body)
    

def LayoutVisual():
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor().visit_stmt(func.body)
        return func
    
    return prim_func_pass(pass_fn, opt_level=0)