from tvm_tirx import ir, tirx
from tvm_tirx.tirx.transform import function_pass

from tilelang._compat.functor import PyStmtExprMutator
from tilelang._compat.stmt_functor import post_order_visit


class _SingleStoreWrapper(PyStmtExprMutator):
    _cuda_thread_indices = frozenset(
        f"tirx.cuda.{name}"
        for name in (
            "block_idx",
            "thread_idx",
            "cluster_id",
            "cluster_cta_id",
            "linear_thread_id",
            "warp_id",
            "lane_id",
            "warpgroup_id",
            "warp_in_warpgroup",
            "thread_in_warpgroup",
            "cta_pair_id",
        )
    )

    def __init__(self):
        super().__init__()
        self.tile_operation_depth = 0
        self.thread_variables = set()

    def _depends_on_threads(self, operation):
        depends = False

        def visit(node):
            nonlocal depends
            if (isinstance(node, ir.Var) and node.__chandle__() in self.thread_variables) or (
                isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name in self._cuda_thread_indices
            ):
                depends = True

        post_order_visit(operation, visit)
        return depends

    def visit_seq_stmt_(self, statement):
        previous = self.thread_variables.copy()
        try:
            return super().visit_seq_stmt_(statement)
        finally:
            self.thread_variables = previous

    def visit_region_stmt_(self, statement):
        previous = self.thread_variables.copy()
        try:
            if statement.op.name == "tirx.launch_thread":
                self.thread_variables.update(variable.__chandle__() for variable in statement.body_params)
            return super().visit_region_stmt_(statement)
        finally:
            self.thread_variables = previous

    def visit_bind_(self, statement):
        if self._depends_on_threads(statement.value):
            self.thread_variables.add(statement.var.__chandle__())
        return super().visit_bind_(statement)

    def visit_for_(self, statement):
        is_tile_operation = statement.kind == ir.ForKind.PARALLEL or "num_stages" in statement.annotations
        self.tile_operation_depth += int(is_tile_operation)
        try:
            return super().visit_for_(statement)
        finally:
            self.tile_operation_depth -= int(is_tile_operation)

    def visit_tensor_store_(self, statement):
        if self.tile_operation_depth or self._depends_on_threads(statement):
            return statement

        fragment_accesses = []

        def collect(node):
            if isinstance(node, ir.TensorLoad):
                tensor = node.source
            elif isinstance(node, ir.TensorStore):
                tensor = node.dest
            else:
                return
            if isinstance(tensor.ty, tirx.TensorType) and tensor.ty.scope.name == "local.fragment":
                fragment_accesses.append(node.indices)

        post_order_visit(statement, collect)
        if not fragment_accesses:
            return statement
        for indices in fragment_accesses:
            for index in indices:
                if isinstance(index, ir.IntImm) and int(index) != 0:
                    raise ValueError(
                        f"Fragment buffer access with non-zero index [{index}] is not supported. Only fragment[0] access is allowed."
                    )

        variable = ir.Var("fragment_lane", "int32", loc=statement.loc)
        return ir.For(variable, 0, 1, 1, ir.ForKind.PARALLEL, statement, loc=statement.loc)


def AddWrapperForSingleBufStore():
    """Wrap scalar fragment stores outside tile loops in a unit parallel loop.

    Thread-region parameters and values derived from native thread-index calls
    remain thread-bound. Every fragment access is checked, and bindings are
    tracked only within their lexical scope.
    """

    def pass_fn(func, mod, ctx):
        if func.body is None:
            return func
        return func.with_body(_SingleStoreWrapper().visit_stmt(func.body), loc=func.loc)

    return function_pass(pass_fn, opt_level=0, name="tl.AddWrapperForSingleBufStore")
