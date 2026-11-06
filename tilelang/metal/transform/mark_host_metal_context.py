"""Mark native packed Metal calls for PyTorch's MPS stream integration."""

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.analysis._visitor import tilelang_metadata
from tilelang.language._annotation_codec import encode_annotations


class _MarkHostMetalContextMutator:
    def __init__(self):
        self.compute_depth = 0
        self.metal_depth = 0

    def rewrite(self, body):
        return tvm_ffi.structural_mutate(body, [(ir.RegionStmt, self.region), (ir.Evaluate, self.evaluate)])

    def region(self, node, mutator):
        compute = node.op.name == "tirx.compute_scope"
        metadata = tilelang_metadata(node)
        metal = metadata is not None and metadata.get("key") == "metal_context"
        self.compute_depth += int(compute)
        self.metal_depth += int(metal)
        try:
            # Preserve region operands, definitions, results and lexical scope.
            # An unrelated native region is not an attribute to erase.
            return mutator.default_mutate(node)
        finally:
            self.compute_depth -= int(compute)
            self.metal_depth -= int(metal)

    def evaluate(self, node, mutator):
        rewritten = mutator.default_mutate(node)
        if (
            self.compute_depth
            and not self.metal_depth
            and isinstance(rewritten.value, ir.Call)
            and getattr(rewritten.value.op, "name", "") == "tirx.call_packed_lowered"
        ):
            schema, payload = encode_annotations({"node": None, "key": "metal_context", "value": ""})
            return ir.RegionStmt(
                "tl.attr",
                [payload],
                [],
                {"schema_version": 1, "annotation_schema": schema},
                [rewritten],
                loc=node.loc,
            )
        return rewritten


def MarkHostMetalContext():
    def pass_fn(function, _module, _context):
        if function.attrs.get("declaration", False):
            return function
        body = _MarkHostMetalContextMutator().rewrite(function.body)
        return function.with_body(body, loc=function.loc)

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.MarkHostMetalContext")


__all__ = ["MarkHostMetalContext"]
