from tvm import tir
from tvm.tir import (
    BufferStore,
    LetStmt,
    Broadcast,
    Var,
    PrimFunc,
    PyStmtExprMutator,
)
from tvm.tir.transform import prim_func_pass


@tir.functor.mutator
class HoistBroadcastValuesMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        # Temporary queue: used to store variables that need to be defined within the current statement.
        self.pending_defs = []
        # Flag to indicate if hoist should be enabled.
        self.hoist_enabled = False

    def visit_broadcast_(self, op):
        if self.hoist_enabled and isinstance(op.value, (tir.IntImm, tir.FloatImm)):
            # 1. Intercept Broadcast nodes.
            # Extract the value to be hoisted into a variable.
            val = self.visit_expr(op.value)
            # 2. Create a new variable.
            new_var = Var("broadcast_var", dtype=val.dtype)

            # 3. Add the (variable, value) pair to the pending queue.
            # Note: Do not create the LetStmt here; it must wrap the statement.
            self.pending_defs.append((new_var, val))

            # 4. Return a new Broadcast node, using the new variable to replace the original value.
            return Broadcast(new_var, op.lanes)
        return Broadcast(self.visit_expr(op.value), self.visit_expr(op.lanes))

    # Intercept statement types that might contain expressions with broadcasts.
    # Currently handled: BufferStore, LetStmt.
    def visit_buffer_store_(self, op: BufferStore):
        # 1. Save the current state to handle nested statements correctly.
        saved_hoist_enabled = self.hoist_enabled
        saved_pending_defs = self.pending_defs

        # 2. Enable hoist flag and clear the pending queue for the current statement context.
        self.hoist_enabled = True
        self.pending_defs = []

        # 3. Visit child nodes normally (this will trigger visit_broadcast_).
        new_indices = [self.visit_expr(idx) for idx in op.indices]
        new_stmt = BufferStore(op.buffer, self.visit_expr(op.value), new_indices)

        # 4. Check if there are variables waiting to be defined.
        if self.pending_defs:
            # 5. Wrap the current statement with LetStmt.
            # Order: Traverse in reverse to ensure the first definition wraps the outermost layer.
            # Structure generated: Let my_var = val In BufferStore(...)
            for var, val in reversed(self.pending_defs):
                new_stmt = LetStmt(var, val, new_stmt)

        # 6. Restore the saved state.
        self.hoist_enabled = saved_hoist_enabled
        self.pending_defs = saved_pending_defs

        return new_stmt

    def visit_let_stmt_(self, op: LetStmt):
        # 1. Save the current state to handle nested statements correctly.
        saved_hoist_enabled = self.hoist_enabled
        saved_pending_defs = self.pending_defs

        # 2. Enable hoist flag and clear the pending queue for the current statement context.
        self.hoist_enabled = True
        self.pending_defs = []

        # 3. Visit the value expression (this will trigger visit_broadcast_).
        new_value = self.visit_expr(op.value)

        # 4. Capture the pending defs from the value expression before visiting body.
        value_pending_defs = self.pending_defs

        # 5. Disable hoist flag and clear pending defs before visiting body.
        self.hoist_enabled = False
        self.pending_defs = []

        # 6. Recursively visit the body.
        new_body = self.visit_stmt(op.body)

        # 7. Create the new LetStmt.
        new_stmt = LetStmt(op.var, new_value, new_body)

        # 8. Check if there are variables waiting to be defined from the value expression.
        if value_pending_defs:
            # 9. Wrap the current statement with LetStmt.
            for var, val in reversed(value_pending_defs):
                new_stmt = LetStmt(var, val, new_stmt)

        # 10. Restore the saved state.
        self.hoist_enabled = saved_hoist_enabled
        self.pending_defs = saved_pending_defs

        return new_stmt


def HoistBroadcastValues():
    """
    TVM Pass: HoistBroadcastValues.

    This pass scans the TIR for Broadcast operations involving immediate constants (IntImm, FloatImm).
    It extracts these constants into variables defined via LetStmt immediately surrounding
    the statement where the broadcast occurs.

    Example Transformation:
    -----------------------
    Before:
        A[i] = B[i] + T.Broadcast(3.14, 4) + T.Broadcast(3.14, 4)

    After:
        bv_3_14 = 3.14
        bv_3_14_1 = 3.14
        A[i] = B[i] + T.Broadcast(bv_3_14, 4) + T.Broadcast(bv_3_14_1, 4)
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        mutator = HoistBroadcastValuesMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
