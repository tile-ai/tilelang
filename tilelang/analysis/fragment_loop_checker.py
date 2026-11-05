"""Check fragment access dependencies in native loops with symbolic domains."""

import tvm_ffi
from tvm_tirx import ir, tirx

from ._visitor import NativeStmtVisitor, tensor_scope


def collect_fragment_accesses(statement):
    """Collect actual TensorLoad/Store accesses whose source type is a fragment."""
    accesses = []

    def collect(node):
        tensor = node.source if isinstance(node, ir.TensorLoad) else node.dest
        scope = tensor_scope(tensor)
        if scope is not None and scope.startswith("local.fragment"):
            accesses.append(node)

    tvm_ffi.structural_walk(statement, ((ir.TensorLoad, ir.TensorStore), collect), order="post")
    return accesses


class _FragmentLoopCheckVisitor(NativeStmtVisitor):
    def _check_indices(self, tensor, indices):
        scope = tensor_scope(tensor)
        if scope is None or not scope.startswith("local.fragment"):
            return
        for loop in self.loop_stack:
            if loop.kind != ir.ForKind.PARALLEL:
                continue
            if all(isinstance(value, ir.IntImm) for value in (loop.begin, loop.end, loop.step)):
                continue
            if any(self.uses_var(index, loop.var) for index in indices):
                raise ValueError(
                    "[Tilelang Semantic Check] "
                    f"Loop variable {loop.var} in a T.Parallel loop with symbolic range "
                    f"(begin={loop.begin}, end={loop.end}, step={loop.step}) is used to index "
                    "a fragment buffer, which is not allowed in Tilelang."
                )

    def visit_tensor_load(self, op, visitor):
        self._check_indices(op.source, op.indices)
        super().visit_tensor_load(op, visitor)

    def visit_tensor_store(self, op, visitor):
        self._check_indices(op.dest, op.indices)
        super().visit_tensor_store(op, visitor)


def FragmentLoopChecker():
    """Reject a fragment index depending on a symbolic parallel-loop domain.

    Symbolic parallel loops that do not index fragments through their iterator,
    ordinary serial loops and shared/global tensor accesses remain valid.
    """

    def pass_fn(func, _mod, _ctx):
        _FragmentLoopCheckVisitor().visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.FragmentLoopChecker")
