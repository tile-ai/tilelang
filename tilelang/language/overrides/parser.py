"""TVMScript parser overrides tailored for TileLang."""
from __future__ import annotations

from functools import partial

from tvm.script.ir_builder import tir as T
from tvm.script.parser._core import dispatch, doc
from tvm.tir import BufferLoad, Var

from tvm.script.parser.tir import parser as tvm_tir_parser
from tilelang.language.tir.ir import SerialStepSpec


def _get_node_span(node: doc.AST) -> tuple[int, int, int, int]:
    """Return the span (lineno, col, end_lineno, end_col) for a doc node."""
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_assign).
@dispatch.register(token="tir", type_name="Assign")
def tilelang_visit_assign(self, node: doc.Assign) -> None:  # pylint: disable=unused-argument
    """Override `Assign` to support chained writes and `local.var` buffers."""
    if not node.targets:
        self.report_error(node, "Assignment must have at least one target.")

    if isinstance(node.value, doc.Subscript):
        check_slices = []
        if isinstance(node.value.slice, doc.Slice):
            check_slices = [node.value.slice]
        elif isinstance(node.value.slice, doc.Tuple):
            for part in node.value.slice.elts:
                if isinstance(part, doc.Slice):
                    check_slices.append(part)
        for slice_node in check_slices:
            if not slice_node.step and slice_node.upper and slice_node.lower:
                slice_node.step = doc.Constant(
                    1,
                    None,
                    1,
                    1,
                    slice_node.upper.lineno,
                    slice_node.upper.end_col_offset + 1,
                    slice_node.upper.lineno,
                    slice_node.upper.end_col_offset + 2,
                )

    rhs = self.eval_expr(node.value)
    for lhs in node.targets:
        if isinstance(lhs, doc.Subscript):
            if isinstance(lhs.slice, doc.Tuple):
                indices = [self.eval_expr(index) for index in lhs.slice.elts]
            else:
                indices = self.eval_expr(lhs.slice)
            T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
            continue

        if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
            load_ctx = doc.Load()
            store_ctx = doc.Store()
            lhs.ctx = load_ctx
            lhs_value = self.eval_expr(lhs)
            lhs.ctx = store_ctx
            if (isinstance(lhs_value, BufferLoad) and lhs_value.buffer.scope() == "local.var" and
                    len(lhs_value.indices) == 1 and lhs_value.indices[0] == 0):
                T.buffer_store(lhs_value.buffer, rhs, indices=[0])
                continue

        self.eval_assign(target=lhs, source=rhs, bind_value=tvm_tir_parser.bind_assign_value)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_aug_assign).
@dispatch.register(token="tir", type_name="AugAssign")
def tilelang_visit_aug_assign(self, node: doc.AugAssign) -> None:  # pylint: disable=unused-argument
    """Override `AugAssign` to support writes into `local.var` buffers."""
    lhs_pos = _get_node_span(node.target)
    rhs_pos = _get_node_span(node.value)

    node.target.ctx = doc.Load()
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)

    lhs = node.target
    lhs.ctx = doc.Store()
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = [self.eval_expr(index) for index in lhs.slice.elts]
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
        return

    if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
        load_ctx = doc.Load()
        store_ctx = doc.Store()
        lhs.ctx = load_ctx
        lhs_value = self.eval_expr(lhs)
        lhs.ctx = store_ctx
        if (isinstance(lhs_value, BufferLoad) and lhs_value.buffer.scope() == "local.var" and
                len(lhs_value.indices) == 1 and lhs_value.indices[0] == 0):
            T.buffer_store(lhs_value.buffer, rhs, indices=[0])
            return

    self.eval_assign(target=lhs, source=rhs, bind_value=tvm_tir_parser.bind_assign_value)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_ann_assign).
@dispatch.register(token="tir", type_name="AnnAssign")
def tilelang_visit_ann_assign(self, node: doc.AnnAssign) -> None:  # pylint: disable=unused-argument
    """Override `AnnAssign` to support writes into `local.var` buffers."""
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_var = self.visit_tvm_annotation(node.annotation)
    if not isinstance(ann_var, Var):
        self.report_error(node.annotation, "Annotation should be Var")

    if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
        load_ctx = doc.Load()
        store_ctx = doc.Store()
        lhs.ctx = load_ctx
        lhs_value = self.eval_expr(lhs)
        lhs.ctx = store_ctx
        if (isinstance(lhs_value, BufferLoad) and lhs_value.buffer.scope() == "local.var" and
                len(lhs_value.indices) == 1 and lhs_value.indices[0] == 0):
            T.buffer_store(lhs_value.buffer, rhs, indices=[0])
            return

    self.eval_assign(target=lhs, source=ann_var, bind_value=tvm_tir_parser.bind_assign_value)
    frame = T.LetStmt(rhs, var=ann_var)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


# Override For to support stepped serial: T.serial(start, end, step)
@dispatch.register(token="tir", type_name="For")
def tilelang_visit_for(self, node: doc.For) -> None:  # pylint: disable=unused-argument
    """Override `For` to add support for T.serial(start, end, step).

    When the iterable is a SerialStepSpec, lower it to a unit-step loop over
    t in [0, floor_div(|end-start|, step)] and bind the loop variable using a
    Let to `start + t*step` (inclusive semantics).
    """
    iter_val = self.eval_expr(node.iter)

    # Fast path: fall back to TVM default behavior when not a SerialStepSpec
    if not isinstance(iter_val, SerialStepSpec):
        if not isinstance(iter_val, T.frame.ForFrame):
            self.report_error(
                node.iter,
                "Expect the for loop to be one of the following: "
                "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
            )
        with self.var_table.with_frame(), iter_val as iters:
            self.eval_assign(
                target=node.target, source=iters, bind_value=tvm_tir_parser.bind_for_value)
            self.visit_body(node.body)
        return

    # Stepped inclusive serial: require positive integer step
    start = iter_val.start
    end = iter_val.stop
    step = iter_val.step
    annotations = iter_val.annotations

    # Normalize step to Python int if possible, otherwise expect IntImm-like
    if isinstance(step, int):
        step_val = step
    else:
        step_val = getattr(step, "value", None)
        if step_val is None:
            self.report_error(node.iter, "T.serial step must be an integer or IntImm")
            return

    if step_val <= 0:
        self.report_error(node.iter, "T.serial step must be a positive integer")
        return

    # Use tvm.tir.floordiv via builder ops from tilelang.tir.ir if available
    # Avoid importing op wrappers; compute using arithmetic to keep it simple.
    # We construct: T.ceildiv((end - start), step)
    extent = T.ceildiv(end - start, step_val)  # type: ignore[operator]

    for_frame = T.serial(0, extent, annotations=annotations)
    with self.var_table.with_frame(), for_frame as t:
        # Bind loop target as Let var: i = start + t * step
        stepped_index = start + t * step_val  # type: ignore[operator]
        self.eval_assign(
            target=node.target,
            source=stepped_index,
            bind_value=tvm_tir_parser.bind_assign_value,
        )
        self.visit_body(node.body)
