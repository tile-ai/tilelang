from tvm_tirx import ir
from tvm_tirx.ir.prim import Broadcast
from tvm_tirx.tirx import Function
from tvm_tirx.tirx.transform import function_pass

from tilelang._compat.functor import PyStmtExprMutator, mutator


@mutator
class HoistBroadcastValuesMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        self.pending_defs = []
        self.hoist_enabled = False

    def visit_broadcast_(self, op):
        value = self.visit_expr(op.value)
        lanes = self.visit_expr(op.lanes)
        if self.hoist_enabled and isinstance(value, (ir.IntImm, ir.FloatImm)):
            variable = ir.Var("broadcast_var", value.ty, loc=op.loc)
            self.pending_defs.append((variable, value))
            return Broadcast(variable, lanes, loc=op.loc)
        return Broadcast(value, lanes, loc=op.loc)

    def _with_hoisted_values(self, make_statement, loc):
        saved_enabled = self.hoist_enabled
        saved_definitions = self.pending_defs
        self.hoist_enabled = True
        self.pending_defs = []
        try:
            statement = make_statement()
            if not self.pending_defs:
                return statement
            definitions = [ir.Bind(variable, value, loc=loc) for variable, value in self.pending_defs]
            return ir.SeqStmt([*definitions, statement], loc=loc)
        finally:
            self.hoist_enabled = saved_enabled
            self.pending_defs = saved_definitions

    def visit_tensor_store_(self, op):
        def make_statement():
            indices = [self.visit_expr(index) for index in op.indices]
            value = self.visit_expr(op.value)
            return ir.TensorStore(op.dest, indices, value, loc=op.loc)

        return self._with_hoisted_values(make_statement, op.loc)

    def visit_bind_(self, op):
        return self._with_hoisted_values(lambda: ir.Bind(op.var, self.visit_expr(op.value), loc=op.loc), op.loc)


def HoistBroadcastValues():
    """Bind broadcast immediates immediately before their store or definition.

    Each constant is hoisted inside its existing lexical scope. Tensor types,
    variable identities, and source locations are preserved. The pass consumes
    and produces native TIRx IR and does not require a legacy Buffer dialect.
    """

    def pass_fn(func: Function, mod, ctx):
        if func.body is None:
            return func
        body = HoistBroadcastValuesMutator().visit_stmt(func.body)
        return func.with_body(body, loc=func.loc)

    return function_pass(pass_fn, opt_level=0, name="tl.HoistBroadcastValues")
