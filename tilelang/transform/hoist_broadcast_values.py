"""Bind scalar broadcast constants next to their native statement consumers."""

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim


class HoistBroadcastValuesMutator:
    def visit_stmt(self, statement):
        return tvm_ffi.structural_mutate(
            statement,
            [
                (ir.SeqStmt, self.visit_seq),
                ((ir.Bind, ir.TensorStore), self.visit_consumer),
            ],
        )

    def visit_seq(self, node, mutator):
        # Constructing the normalized SeqStmt splices the new scalar Bind
        # before its consumer in the original lexical definition scope.
        return ir.SeqStmt([mutator.mutate(stmt) for stmt in node.seq], loc=node.loc)

    def visit_consumer(self, node, _mutator):
        pending = []

        def broadcast(value, mutator):
            # Native low-precision literals are Cast(FloatImm), because the
            # immediate carries the source precision.  Keep the complete
            # cast chain in the Bind: evaluating it as a Python number would
            # lose rounding and dtype semantics.  A cast of a Var, load, or
            # Call is not a literal and must keep its original read point.
            literal = value.value
            while isinstance(literal, prim.Cast):
                literal = literal.value
            if isinstance(literal, (ir.IntImm, ir.FloatImm)):
                scalar = value.value
                variable = ir.Var("broadcast_var", scalar.ty, loc=value.loc)
                pending.append(ir.Bind(variable, scalar, loc=node.loc))
                return prim.Broadcast(variable, value.lanes, loc=value.loc)
            return mutator.default_mutate(value)

        def map_value(value):
            return tvm_ffi.structural_mutate(
                value,
                [
                    (prim.Broadcast, broadcast),
                    # A function/lambda has its own definitions and captures.
                    # A statement-level hoist cannot introduce a free variable
                    # into it; its body is processed in its own compilation.
                    ((ir.LambdaExpr, tirx.Function), lambda value, _mutator: value),
                ],
            )

        if isinstance(node, ir.Bind):
            result = ir.Bind(node.var, map_value(node.value), loc=node.loc)
        else:
            indices = [map_value(index) for index in node.indices]
            result = ir.TensorStore(node.dest, indices, map_value(node.value), loc=node.loc)
        return ir.SeqStmt([*pending, result], loc=node.loc) if pending else result


def HoistBroadcastValues():
    """Hoist literal Broadcast values while preserving lanes, types and scope.

    Each extracted constant gets a native scalar Bind immediately before the
    original Bind or TensorStore. Existing Bind identities are retained so
    subsequent statement uses remain dominated by their definitions.
    """

    def pass_fn(func, _mod, _ctx):
        if func.attrs.get("declaration", False):
            return func
        return func.with_body(HoistBroadcastValuesMutator().visit_stmt(func.body), loc=func.loc)

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.HoistBroadcastValues")
