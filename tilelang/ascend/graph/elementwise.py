"""Conservative scalar-TIR to Ascend SIMT scheduling.

The output coordinates are the ownership contract: every output element is
computed by exactly one core and one VF thread. Injective producers are inlined
before rebuilding the launch, so their values never become GM intermediates.
"""

from __future__ import annotations

import math

from tilelang import tvm
from tvm import s_tir, tirx


_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_SCALAR_INTRINSICS = tuple(tvm.ir.Op.get(name) for name in ("tirx.exp", "tirx.sqrt", "tirx.rsqrt"))


class ElementwiseScheduleError(ValueError):
    """A valid TIR function is outside the initial injective schedule family."""


@tirx.functor.mutator
class _LegalizeScalarArithmetic(tirx.PyStmtExprMutator):
    """Keep low-precision operation boundaries and Torch NaN propagation.

    CCE's C++ bfloat16 arithmetic promotes operands, while its max overloads
    cannot select between that promoted result and a bfloat16 argument. Explicit
    fp32 arithmetic followed by a narrowing cast both resolves the overload and
    preserves every fp16/bf16 rounding boundary after producer inlining.
    """

    def _arithmetic(self, op, constructor):
        lhs, rhs = self.visit_expr(op.a), self.visit_expr(op.b)
        if str(op.dtype) in ("float16", "bfloat16"):
            value = constructor(tirx.Cast("float32", lhs), tirx.Cast("float32", rhs))
            return tirx.Cast(op.dtype, value)
        return constructor(lhs, rhs)

    def visit_add_(self, op):
        return self._arithmetic(op, tirx.Add)

    def visit_sub_(self, op):
        return self._arithmetic(op, tirx.Sub)

    def visit_mul_(self, op):
        return self._arithmetic(op, tirx.Mul)

    def visit_div_(self, op):
        return self._arithmetic(op, tirx.Div)

    def _minmax(self, op, maximum):
        lhs, rhs = self.visit_expr(op.a), self.visit_expr(op.b)
        floating = str(op.dtype) in _FLOAT_DTYPES
        if str(op.dtype) in ("float16", "bfloat16"):
            lhs, rhs = tirx.Cast("float32", lhs), tirx.Cast("float32", rhs)
        selected = tirx.Select(lhs < rhs if maximum else lhs > rhs, rhs, lhs)
        if floating:
            selected = tirx.Select(tirx.isnan(lhs), lhs, tirx.Select(tirx.isnan(rhs), rhs, selected))
        return tirx.Cast(op.dtype, selected) if str(op.dtype) in ("float16", "bfloat16") else selected

    def visit_max_(self, op):
        return self._minmax(op, True)

    def visit_min_(self, op):
        return self._minmax(op, False)

    def visit_call_(self, op):
        if any(op.op.same_as(intrinsic) for intrinsic in _SCALAR_INTRINSICS):
            argument = self.visit_expr(op.args[0])
            if str(op.dtype) in ("float16", "bfloat16"):
                return tirx.Cast(
                    op.dtype, tirx.Call("float32", op.op, [tirx.Cast("float32", argument)], annotations=op.annotations, span=op.span)
                )
            return tirx.Call(op.dtype, op.op, [argument], annotations=op.annotations, span=op.span)
        return tirx.Call(op.dtype, op.op, [self.visit_expr(argument) for argument in op.args], annotations=op.annotations, span=op.span)


def legalize_scalar_arithmetic(expr):
    """Preserve floating-point boundaries in a graph-generated VF expression."""
    return _LegalizeScalarArithmetic().visit_expr(expr)


def static_shape(buffer: tirx.Buffer) -> tuple[int, ...]:
    """Require positive, static, compact float buffers at the graph boundary."""
    if str(buffer.dtype) not in _FLOAT_DTYPES:
        raise ElementwiseScheduleError(f"Unsupported graph buffer dtype: {buffer.dtype}")
    if buffer.strides or buffer.axis_separators:
        raise ElementwiseScheduleError("Graph kernels require compact buffers")
    if not isinstance(buffer.elem_offset, tirx.IntImm) or int(buffer.elem_offset) != 0:
        raise ElementwiseScheduleError("Graph kernels require zero buffer offsets")
    shape = []
    for extent in buffer.shape:
        if not isinstance(extent, tirx.IntImm) or int(extent) <= 0:
            raise ElementwiseScheduleError("Graph kernels require positive static shapes")
        shape.append(int(extent))
    return tuple(shape)


def make_simt_kernel(
    func: tirx.PrimFunc,
    body: tirx.Stmt,
    *,
    core_var: tirx.Var,
    thread_var: tirx.Var,
    grid: int,
    threads: int,
    allocations=(),
) -> tirx.PrimFunc:
    """Wrap a scalar per-thread body in the native Ascend launch/VF regions.

    This intentionally uses the same opaque VF and local thread domain as
    ``T.SimtVF``. Kernel-wide CUDA thread bindings would erase that boundary.
    """
    vf_body = tirx.AttrStmt(tirx.StringImm("simtvf"), "tl.simtvf_scope", 1, body)
    for dimension, var, extent in reversed(
        [
            ("x", thread_var, threads),
            ("y", tirx.Var("thread_y", "int32"), 1),
            ("z", tirx.Var("thread_z", "int32"), 1),
        ]
    ):
        axis = tirx.IterVar(tvm.ir.Range(0, extent), var, tirx.IterVar.ThreadIndex, f"threadIdx.{dimension}")
        vf_body = tirx.AttrStmt(axis, "thread_extent", extent, vf_body)
    vf = tirx.SBlock([], [], [], "SIMT_VF", vf_body, alloc_buffers=list(allocations), annotations={"tl.vf_source_index": 0})
    root = tirx.SBlock([], [], [], "tilelang_root", vf)
    core_axis = tirx.IterVar(tvm.ir.Range(0, grid), core_var, tirx.IterVar.ThreadIndex, "blockIdx.x")
    scheduled = func.with_body(tirx.AttrStmt(core_axis, "thread_extent", grid, root))
    return scheduled.with_attr("tirx.is_scheduled", True).with_attr("tirx.is_tilelang_kernel", True)


def _blocks(func: tirx.PrimFunc) -> list[tirx.SBlock]:
    blocks = []

    def collect(node):
        if isinstance(node, tirx.SBlock) and node.iter_vars:
            blocks.append(node)

    tirx.stmt_functor.post_order_visit(func.body, collect)
    return blocks


def _validate_unscheduled_structure(func: tirx.PrimFunc, output: tirx.Buffer) -> None:
    """Reject effects, predicates and partial loop domains before rebuilding IR.

    A scalar expression alone cannot establish that the source wrote the whole
    output. Check each realization against its enclosing loop domains rather
    than dropping the original control flow and silently filling missing data.
    """
    parameters = set(func.buffer_map.values())
    domains = {}
    output_writers = 0

    def visit(stmt):
        nonlocal output_writers
        if isinstance(stmt, tirx.For):
            if stmt.kind != tirx.ForKind.SERIAL or (
                stmt.step is not None and (not isinstance(stmt.step, tirx.IntImm) or int(stmt.step) != 1)
            ):
                raise ElementwiseScheduleError("Expected unscheduled unit-step spatial loops")
            domains[stmt.loop_var] = (stmt.min, stmt.extent)
            visit(stmt.body)
            del domains[stmt.loop_var]
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, tirx.SBlockRealize):
            if not isinstance(stmt.predicate, tirx.IntImm) or not int(stmt.predicate):
                raise ElementwiseScheduleError("Predicated TIR blocks require a different schedule")
            analyzer = tvm.arith.Analyzer()
            for axis, value in zip(stmt.block.iter_vars, stmt.iter_values):
                if (
                    isinstance(value, tirx.IntImm)
                    and analyzer.can_prove_equal(axis.dom.extent, 1)
                    and analyzer.can_prove_equal(value, axis.dom.min)
                ):
                    continue
                if not isinstance(value, tirx.Var) or value not in domains:
                    raise ElementwiseScheduleError("Spatial block bindings must cover complete loop domains")
                minimum, extent = domains[value]
                if not analyzer.can_prove_equal(minimum, axis.dom.min) or not analyzer.can_prove_equal(extent, axis.dom.extent):
                    raise ElementwiseScheduleError("A block realization covers only part of its spatial domain")
            visit(stmt.block)
        elif isinstance(stmt, tirx.SBlock):
            if stmt.init is not None:
                raise ElementwiseScheduleError("Reduction blocks require the reduction schedule")
            if stmt.iter_vars:
                if not isinstance(stmt.body, tirx.BufferStore):
                    raise ElementwiseScheduleError("Expected a pure scalar store in every injective block")
                buffer = stmt.body.buffer
                if buffer in parameters and not buffer.same_as(output):
                    raise ElementwiseScheduleError("Input mutation is not supported by graph schedules")
                output_writers += int(buffer.same_as(output))
            elif stmt.name_hint != "root":
                raise ElementwiseScheduleError("Expected an unscheduled root and spatial blocks")
            visit(stmt.body)
        elif not isinstance(stmt, tirx.BufferStore):
            raise ElementwiseScheduleError(f"Unsupported injective statement: {type(stmt).__name__}")

    visit(func.body)
    if output_writers != 1:
        raise ElementwiseScheduleError("Expected exactly one complete output writer")


def _inline_producers(func: tirx.PrimFunc, output: tirx.Buffer) -> tirx.PrimFunc:
    blocks = _blocks(func)
    if not blocks:
        raise ElementwiseScheduleError("Expected unscheduled scalar TIR blocks")
    for block in blocks:
        if block.init is not None or any(axis.iter_type != tirx.IterVar.DataPar for axis in block.iter_vars):
            raise ElementwiseScheduleError("Reduction blocks require the reduction schedule")
        if not isinstance(block.body, tirx.BufferStore):
            raise ElementwiseScheduleError("Expected a pure scalar store in every injective block")
    schedule = s_tir.Schedule(func)
    for block in blocks:
        if not block.body.buffer.same_as(output):
            try:
                schedule.compute_inline(schedule.get_sblock(block.name_hint))
            except tvm.error.TVMError as error:
                raise ElementwiseScheduleError(f"Cannot inline injective producer {block.name_hint}") from error
    return schedule.mod["main"]


def _scalar_output(func: tirx.PrimFunc, output: tirx.Buffer):
    blocks = _blocks(func)
    if len(blocks) != 1:
        raise ElementwiseScheduleError("The injective schedule requires one output block")
    block = blocks[0]
    store = block.body
    if not isinstance(store, tirx.BufferStore) or not store.buffer.same_as(output):
        raise ElementwiseScheduleError("The last buffer parameter must be the unique output")
    shape = static_shape(output)
    if len(store.indices) != len(shape):
        raise ElementwiseScheduleError("Output index rank mismatch")
    coordinates = [tirx.Var(f"coord_{axis}", "int64") for axis in range(len(shape))]
    substitutions = {}
    block_vars = {axis.var for axis in block.iter_vars}
    for index, coordinate, extent in zip(store.indices, coordinates, shape):
        if isinstance(index, tirx.Var) and index in block_vars and index not in substitutions:
            axis = next(axis for axis in block.iter_vars if axis.var.same_as(index))
            if (
                not isinstance(axis.dom.min, tirx.IntImm)
                or int(axis.dom.min) != 0
                or not tvm.arith.Analyzer().can_prove_equal(axis.dom.extent, extent)
            ):
                raise ElementwiseScheduleError("Output axes must cover the complete output shape")
            substitutions[index] = tirx.Cast(index.dtype, coordinate)
        elif not (isinstance(index, tirx.IntImm) and int(index) == 0 and extent == 1):
            raise ElementwiseScheduleError("Output stores must use independent spatial coordinates")
    for axis in block.iter_vars:
        if axis.var not in substitutions:
            if not isinstance(axis.dom.extent, tirx.IntImm) or int(axis.dom.extent) != 1:
                raise ElementwiseScheduleError("An output block axis is not owned by an output coordinate")
            substitutions[axis.var] = axis.dom.min
    expression = tirx.stmt_functor.substitute(store.value, substitutions)
    return coordinates, expression


def _validate_scalar(expression, coordinates, shape, inputs):
    analyzer = tvm.arith.Analyzer()
    for coordinate, extent in zip(coordinates, shape):
        analyzer.bind(coordinate, tvm.ir.Range(0, extent))
    input_buffers = set(inputs)
    allowed_vars = set(coordinates)
    errors = []
    # Only arithmetic and an exact pure-intrinsic whitelist are accepted.
    # Calls may carry effects; data-dependent indices must not bypass the proof.
    allowed = (
        tirx.IntImm,
        tirx.FloatImm,
        tirx.Var,
        tirx.BufferLoad,
        tirx.Call,
        tirx.Add,
        tirx.Sub,
        tirx.Mul,
        tirx.Div,
        tirx.FloorDiv,
        tirx.FloorMod,
        tirx.Min,
        tirx.Max,
        tirx.Cast,
        tirx.Select,
        tirx.LT,
        tirx.LE,
        tirx.GT,
        tirx.GE,
        tirx.EQ,
        tirx.NE,
        tirx.And,
        tirx.Or,
        tirx.Not,
    )

    def check(node):
        # TVM's native visitor turns unknown Python exception subclasses into
        # RuntimeError. Report outside the callback so callers can distinguish
        # an unsupported schedule from a compiler failure.
        if errors:
            return
        if isinstance(node, tirx.PrimExpr) and not isinstance(node, allowed):
            errors.append(f"Unsupported injective expression: {type(node).__name__}")
            return
        if isinstance(node, tirx.PrimExpr) and str(node.dtype).startswith(("float", "bfloat")) and str(node.dtype) not in _FLOAT_DTYPES:
            errors.append(f"Unsupported scalar dtype: {node.dtype}")
            return
        if isinstance(node, tirx.Cast) and str(node.dtype).startswith(("int", "uint")) and str(node.value.dtype) in _FLOAT_DTYPES:
            errors.append("Float-to-integer casts are outside the graph schedule capability")
            return
        if isinstance(node, tirx.FloorDiv | tirx.FloorMod) and str(node.dtype) in _FLOAT_DTYPES:
            errors.append("Floating floor division and modulo are outside the graph schedule capability")
            return
        if isinstance(node, tirx.Call) and (
            not isinstance(node.op, tvm.ir.Op)
            or not any(node.op.same_as(intrinsic) for intrinsic in _SCALAR_INTRINSICS)
            or str(node.dtype) not in _FLOAT_DTYPES
            or len(node.args) != 1
            or str(node.args[0].dtype) not in _FLOAT_DTYPES
        ):
            errors.append("Only unary floating exp/sqrt/rsqrt intrinsics are supported")
            return
        if isinstance(node, tirx.Var) and node not in allowed_vars:
            # Buffer data Vars are visited through Buffer metadata, not through
            # scalar expression children; any other Var is an unbound axis.
            errors.append(f"Unbound scalar variable: {node}")
            return
        if isinstance(node, tirx.BufferLoad):
            if node.buffer not in input_buffers:
                errors.append("An intermediate or output load remains after inlining")
                return
            if len(node.indices) != len(node.buffer.shape):
                errors.append("Buffer index rank mismatch")
                return
            for index, extent in zip(node.indices, node.buffer.shape):
                index_loads = []
                tirx.stmt_functor.post_order_visit(
                    index, lambda value, loads=index_loads: loads.append(value) if isinstance(value, tirx.BufferLoad) else None
                )
                if index_loads or not analyzer.can_prove(index >= 0) or not analyzer.can_prove(index < extent):
                    errors.append("Cannot prove an input read is inside its static region")
                    return

    tirx.stmt_functor.post_order_visit(expression, check)
    if errors:
        raise ElementwiseScheduleError(errors[0])


def _vector_width(expression, coordinates, shape, tile_size, threads, config):
    """Prove aligned contiguous lanes or singleton broadcast before vectorizing.

    Chunks stay within the innermost output row. Every nonuniform input read
    must use the same right-aligned coordinates, with an equally aligned row
    length. More general affine maps use the scalar schedule.
    """
    requested = int(getattr(config, "vector_width", 0))
    if requested not in (0, 1, 4):
        raise ElementwiseScheduleError("vector_width must be 0 (automatic), 1, or 4")
    if requested == 1 or not shape or shape[-1] % 4 or tile_size % (threads * 4):
        return 1
    analyzer = tvm.arith.Analyzer()
    contiguous = True

    def check(node):
        nonlocal contiguous
        if not isinstance(node, tirx.BufferLoad):
            return
        input_shape = tuple(int(extent) for extent in node.buffer.shape)
        if len(input_shape) > len(shape):
            contiguous = False
            return
        start = len(shape) - len(input_shape)
        for dimension, (extent, index) in enumerate(zip(input_shape, node.indices)):
            if extent == 1:
                contiguous = contiguous and analyzer.can_prove_equal(index, 0)
            else:
                output_axis = start + dimension
                coordinate = tirx.Cast(index.dtype, coordinates[output_axis])
                contiguous = contiguous and extent == shape[output_axis] and analyzer.can_prove_equal(index, coordinate)
        if input_shape and input_shape[-1] != 1 and input_shape[-1] % 4:
            contiguous = False

    tirx.stmt_functor.post_order_visit(expression, check)
    return 4 if contiguous else 1


@tirx.functor.mutator
class _ReplaceInputLoads(tirx.PyStmtExprMutator):
    def __init__(self, buffers, index):
        super().__init__()
        self.buffers = buffers
        self.index = index

    def visit_buffer_load_(self, op):
        return tirx.BufferLoad(self.buffers[op.buffer], [self.index])


def _vector_memory_stages(value, output, decoded, vector_lane, width):
    """Vectorize GM transfers separately from scalar floating-point semantics.

    NaN tests and other scalar intrinsics need not be vectorizable to benefit
    from adjacent GM transfers. Local arrays belong to each SIMT thread; they
    hold only one chunk and are allocated inside the VF helper.
    """
    loads = {}

    def collect(node):
        if isinstance(node, tirx.BufferLoad):
            loads.setdefault(node.buffer, node)

    tirx.stmt_functor.post_order_visit(value, collect)
    locals_ = {buffer: tirx.decl_buffer((width,), buffer.dtype, name=f"{buffer.name}_chunk", scope="local") for buffer in loads}
    result = tirx.decl_buffer((width,), output.dtype, name="output_chunk", scope="local")
    stages = []

    def loop(kind, body):
        index = tirx.Var("chunk_lane", "int32")
        return tirx.For(index, 0, width, kind, tirx.stmt_functor.substitute(body, {vector_lane: index}))

    for buffer, load in loads.items():
        store = tirx.BufferStore(locals_[buffer], load, [vector_lane])
        stages.append(loop(tirx.ForKind.VECTORIZED, store))
    scalar = _ReplaceInputLoads(locals_, vector_lane).visit_expr(value)
    stages.append(loop(tirx.ForKind.SERIAL, tirx.BufferStore(result, scalar, [vector_lane])))
    store = tirx.BufferStore(output, tirx.BufferLoad(result, [vector_lane]), decoded)
    stages.append(loop(tirx.ForKind.VECTORIZED, store))
    return tirx.SeqStmt(stages), [*locals_.values(), result]


def schedule_elementwise(func: tirx.PrimFunc, config=None) -> tirx.PrimFunc:
    """Schedule one pure injective function, including inlined producer chains."""
    if len(func.params) < 2 or any(param not in func.buffer_map for param in func.params):
        raise ElementwiseScheduleError("Expected tensor inputs and one tensor output")
    buffers = [func.buffer_map[param] for param in func.params]
    for buffer in buffers:
        static_shape(buffer)
    output = buffers[-1]
    shape = static_shape(output)
    _validate_unscheduled_structure(func, output)
    inlined = _inline_producers(func, output)
    output = inlined.buffer_map[inlined.params[-1]]
    inputs = [inlined.buffer_map[param] for param in inlined.params[:-1]]
    coordinates, expression = _scalar_output(inlined, output)
    _validate_scalar(expression, coordinates, shape, inputs)
    expression = legalize_scalar_arithmetic(expression)

    tile_size = int(getattr(config, "tile_size", 1024))
    threads = int(getattr(config, "threads", 128))
    num_cores = int(getattr(config, "num_cores", 32))
    if min(tile_size, threads, num_cores) <= 0 or threads > 1024:
        raise ElementwiseScheduleError("Positive tile_size/num_cores and 1..1024 threads are required")
    vector_width = _vector_width(expression, coordinates, shape, tile_size, threads, config)
    size = math.prod(shape)
    num_tiles = (size + tile_size - 1) // tile_size
    grid = min(num_cores, num_tiles)
    index_limit = (1 << 31) - 1
    if num_tiles + grid > index_limit or tile_size + threads * vector_width > index_limit:
        raise ElementwiseScheduleError("Graph tile and lane ownership exceeds the supported int32 indexing range")
    core = tirx.Var("core", "int32")
    thread = tirx.Var("thread", "int32")
    tile_loop = tirx.Var("tile_loop", "int32")
    lane_loop = tirx.Var("lane_loop", "int32")
    vector_lane = tirx.Var("vector_lane", "int32")
    lane = (lane_loop * threads + thread) * vector_width
    tile = tile_loop * grid + core
    linear = tirx.Cast("int64", tile) * tile_size + tirx.Cast("int64", lane)
    remaining = linear + vector_lane if vector_width > 1 else linear
    decoded = []
    for extent in reversed(shape):
        decoded.append(remaining % extent)
        remaining = remaining // extent
    decoded.reverse()
    value = tirx.stmt_functor.substitute(expression, dict(zip(coordinates, decoded)))
    store = tirx.BufferStore(output, value, decoded)
    allocations = []
    if vector_width > 1:
        store, allocations = _vector_memory_stages(value, output, decoded, vector_lane, vector_width)
    guarded = tirx.IfThenElse(tirx.And(lane < tile_size, linear < size), store, None)
    body = tirx.For(lane_loop, 0, (tile_size + threads * vector_width - 1) // (threads * vector_width), tirx.ForKind.SERIAL, guarded)
    body = tirx.For(tile_loop, 0, (num_tiles + grid - 1) // grid, tirx.ForKind.SERIAL, body)
    scheduled = make_simt_kernel(inlined, body, core_var=core, thread_var=thread, grid=grid, threads=threads, allocations=allocations)
    return scheduled.with_attr("tl.graph_vector_width", vector_width)
