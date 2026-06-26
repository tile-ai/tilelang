from tvm.tirx import BufferStore, For, AttrStmt, ForKind, Var, PrimFunc, BufferLoad, Buffer, IntImm
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass


def AddWrapperForSingleBufStore():
    """
    Creates a TVM pass that wraps single buffer stores with parallel loops.

    This transformation adds T.Parallel wrappers around buffer stores that:
    1. Access fragment buffers only at the constant all-zero index [0, 0, ...]
    2. Are not inside existing tile operations or thread bindings

    A fragment accessed at any other index (a constant non-zero or a dynamic
    index such as a T.serial loop variable) is rejected, since a fragment is
    distributed across threads and such an access has no valid thread-ownership
    mapping in this lowering path.

    Returns:
        A prim_func_pass that applies the transformation
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        # Counter for tracking nested tile operations
        tile_operation_depth = 0
        # Set of variables bound to threads
        thread_binding_vars = set()

        def get_used_variables(operation) -> set:
            """
            Collects all variables used in the given operation.

            Args:
                operation: The TIR operation to analyze

            Returns:
                Set of variables used in the operation
            """
            used_variables = set()

            def visit_variable(node):
                if isinstance(node, Var):
                    used_variables.add(node)

            post_order_visit(operation, visit_variable)
            return used_variables

        def collect_buffer_accesses(statement) -> tuple[list[Buffer], list[Buffer]]:
            """
            Categorizes buffers accessed in the statement by their scope.

            Args:
                statement: The TIR statement to analyze

            Returns:
                Tuple of (local_buffers, fragment_buffers)
            """
            accessed_buffers = set()

            def visit_buffer_access(node):
                if isinstance(node, (BufferLoad, BufferStore)):
                    accessed_buffers.add(node.buffer)

            post_order_visit(statement, visit_buffer_access)

            local_buffers = []
            fragment_buffers = []
            for buffer in accessed_buffers:
                if buffer.scope() == "local.fragment":
                    fragment_buffers.append(buffer)
                elif buffer.scope().startswith("local"):
                    local_buffers.append(buffer)
            return local_buffers, fragment_buffers

        def collect_buffer_indices(statement) -> dict[Buffer, list]:
            """
            Maps each buffer to the index vectors of every access to it.

            A buffer can be accessed more than once in a statement (e.g. a
            variable-index load and a constant-index store), so all occurrences are
            kept; recording only the last would let one slip past validation.

            Args:
                statement: The TIR statement to analyze

            Returns:
                Dictionary mapping buffers to a list of their access index vectors
            """
            buffer_to_indices = {}

            def visit_buffer_access(node):
                if isinstance(node, (BufferLoad, BufferStore)):
                    buffer_to_indices.setdefault(node.buffer, []).append(node.indices)

            post_order_visit(statement, visit_buffer_access)
            return buffer_to_indices

        def is_tile_operation_loop(loop: For) -> bool:
            """
            Determines if a For loop is a tile operation.

            Args:
                loop: The For loop to check

            Returns:
                True if the loop is a tile operation (parallel or has num_stages annotation)
            """
            return loop.kind == ForKind.PARALLEL or "num_stages" in loop.annotations

        def pre_visit(statement):
            """
            Pre-order visitor that tracks thread bindings and tile operation depth.
            """
            nonlocal tile_operation_depth

            if isinstance(statement, AttrStmt) and statement.attr_key == "thread_extent":
                thread_binding_vars.add(statement.node.var)
            elif isinstance(statement, For) and is_tile_operation_loop(statement):
                tile_operation_depth += 1

        def post_visit(statement):
            """
            Post-order visitor that applies transformations and updates counters.
            """
            nonlocal tile_operation_depth

            if isinstance(statement, For) and is_tile_operation_loop(statement):
                tile_operation_depth -= 1

            elif isinstance(statement, BufferStore):
                used_variables = get_used_variables(statement)
                thread_bound_variables = used_variables.intersection(thread_binding_vars)

                # Only transform if not inside tile operations and no thread bindings
                if tile_operation_depth == 0 and len(thread_bound_variables) == 0:
                    # Skip if no fragment buffers are accessed
                    _, fragment_buffers = collect_buffer_accesses(statement)
                    if len(fragment_buffers) == 0:
                        return statement

                    # Validate fragment buffer indices. In this fallback lowering
                    # path we only support the constant all-zero index [0, 0, ...]. A
                    # dynamic index, such as a T.serial loop variable, would
                    # address a thread-distributed fragment with no valid
                    # thread-ownership mapping and otherwise lowers to invalid
                    # code or races.
                    buffer_indices = collect_buffer_indices(statement)
                    for buffer, index_vectors in buffer_indices.items():
                        if buffer.scope() != "local.fragment":
                            continue
                        for indices in index_vectors:
                            if any(not (isinstance(index, IntImm) and index.value == 0) for index in indices):
                                index_str = ", ".join(str(index) for index in indices)
                                raise ValueError(
                                    f"Unsupported fragment access to '{buffer.name}' at index "
                                    f"[{index_str}]: only the constant all-zero index "
                                    "fragment[0, 0, ...] is supported in this fallback lowering "
                                    "path. A fragment is "
                                    "distributed across CUDA threads, so a dynamic index (e.g. a "
                                    "T.serial loop variable) or a non-zero index has no valid "
                                    "thread-ownership mapping. Use T.Parallel for elementwise "
                                    "access, or T.reduce_sum for reductions."
                                )

                    # Wrap the constant-index fragment[0, 0, ...] access with T.Parallel loop
                    return For(Var("_", "int32"), 0, 1, ForKind.PARALLEL, statement)

            return statement

        new_body = ir_transform(func.body, pre_visit, post_visit)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
