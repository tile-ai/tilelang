from tvm.tir import BufferStore, For, AttrStmt, ForKind, Var, PrimFunc, BufferLoad, Buffer, IntImm
from tvm.tir.stmt_functor import ir_transform, post_order_visit
from tvm.tir.transform import prim_func_pass


def AddWrapperForSingleBufStore():
    """
    Creates a TVM pass that wraps single buffer stores with parallel loops.

    This transformation adds T.Parallel wrappers around buffer stores that:
    1. Access fragment buffers with index 0
    2. Are not inside existing tile operations or thread bindings
    3. Don't access fragment buffers with non-zero indices

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

        def collect_buffer_indices(statement) -> dict[Buffer, list[int]]:
            """
            Maps each buffer to its access indices.

            Args:
                statement: The TIR statement to analyze

            Returns:
                Dictionary mapping buffers to their access indices
            """
            buffer_to_indices = {}

            def visit_buffer_access(node):
                if isinstance(node, (BufferLoad, BufferStore)):
                    buffer_to_indices[node.buffer] = node.indices

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

                    # Validate fragment buffer indices - only index 0 is supported
                    buffer_indices = collect_buffer_indices(statement)
                    for buffer, indices in buffer_indices.items():
                        if buffer.scope() != "local.fragment":
                            continue
                        for index in indices:
                            if isinstance(index, IntImm) and index != 0:
                                raise ValueError(
                                    f"Fragment buffer access with non-zero index [{index}] is not supported. "
                                    "Only fragment[0] access is allowed."
                                )

                    # Wrap fragment[0] access with T.Parallel loop
                    return For(Var("_", "int32"), 0, 1, ForKind.PARALLEL, statement)

            return statement

        new_body = ir_transform(func.body, pre_visit, post_visit)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
