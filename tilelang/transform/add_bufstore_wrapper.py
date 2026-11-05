"""Expose standalone fragment element stores to parallel layout inference."""

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.analysis._visitor import NativeStmtVisitor, tensor_scope, tilelang_metadata


class _SingleStoreWrapper:
    def __init__(self):
        self.tile_operation_depth = 0
        self.thread_binding_vars = set()
        self.analysis = NativeStmtVisitor()

    def visit_stmt(self, statement):
        return tvm_ffi.structural_mutate(
            statement,
            [
                (ir.SeqStmt, self.visit_seq),
                (ir.Bind, self.visit_bind),
                (ir.For, self.visit_for),
                (ir.RegionStmt, self.visit_region),
                (ir.TensorStore, self.visit_store),
            ],
        )

    def visit_seq(self, node, mutator):
        with self.analysis.scope():
            return ir.SeqStmt([mutator.mutate(stmt) for stmt in node.seq], loc=node.loc)

    def visit_bind(self, node, mutator):
        result = mutator.default_mutate(node)
        self.analysis.bindings[result.var] = result.value
        return result

    def visit_for(self, node, mutator):
        is_tile_operation = node.kind == ir.ForKind.PARALLEL or "num_stages" in node.annotations
        self.tile_operation_depth += int(is_tile_operation)
        try:
            with self.analysis.scope():
                self.analysis.bindings.pop(node.var, None)
                return mutator.default_mutate(node)
        finally:
            self.tile_operation_depth -= int(is_tile_operation)

    def visit_region(self, node, mutator):
        previous = self.thread_binding_vars
        if node.op.name in {"tl.launch_kernel", "tirx.launch_thread"}:
            self.thread_binding_vars = previous | set(node.body_params)
        metadata = tilelang_metadata(node)
        pipeline = metadata is not None and metadata.get("key") == "tl.loop.annotations" and "num_stages" in metadata["value"]
        self.tile_operation_depth += int(pipeline)
        try:
            with self.analysis.scope():
                for parameter in node.body_params:
                    self.analysis.bindings.pop(parameter, None)
                return mutator.default_mutate(node)
        finally:
            self.thread_binding_vars = previous
            self.tile_operation_depth -= int(pipeline)

    def visit_store(self, node, _mutator):
        if self.tile_operation_depth:
            return node
        if any(self.analysis.uses_var(node, variable) for variable in self.thread_binding_vars):
            return node
        accesses = []

        def collect(access):
            tensor = access.dest if isinstance(access, ir.TensorStore) else access.source
            if tensor_scope(tensor) == "local.fragment":
                accesses.append(access)

        tvm_ffi.structural_walk(node, ((ir.TensorStore, ir.TensorLoad), collect), order="pre")
        if not accesses:
            return node
        # Check every access, including two different indices of the same
        # tensor in one expression. A dict keyed by tensor would lose one.
        for access in accesses:
            for index in access.indices:
                if isinstance(index, ir.IntImm) and index.value != 0:
                    raise ValueError(
                        f"Fragment buffer access with non-zero index [{index}] is not supported. Only fragment[0] access is allowed."
                    )
        return ir.For(
            ir.Var("_", ir.PrimType("int32")),
            ir.const(0),
            ir.const(1),
            ir.const(1),
            ir.ForKind.PARALLEL,
            node,
            loc=node.loc,
        )


def AddWrapperForSingleBufStore():
    """Wrap standalone fragment[0] accesses in a single-iteration parallel loop.

    Explicit parallel/pipelined loops and accesses depending on launch indices
    keep their current ownership. Both native launch regions and flat Bind
    aliases are examined without changing a tensor's identity or type.
    """

    def pass_fn(func, _mod, _ctx):
        if func.attrs.get("declaration", False):
            return func
        return func.with_body(_SingleStoreWrapper().visit_stmt(func.body), loc=func.loc)

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.AddWrapperForSingleBufStore")
