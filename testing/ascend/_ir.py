"""Native IR builders and inspection for host-only Ascend pass tests.

Scheduling helpers stop at the requested pass. They never run device lowering,
code generation or compilation. Prepared pass inputs are constructed directly.
"""

import tilelang
import tvm_ffi
import tvm_tirx as tvm
from tilelang.ascend import transform as ascend
from tvm_tirx import ir, tirx

SCHEDULED_BIND = "tl.ascend.scheduled_bind"


def seq(*stmts):
    return stmts[0] if len(stmts) == 1 else ir.SeqStmt(stmts)


def tensor(shape, dtype="float32", *, name="buffer", scope="global", layout="default", align=0):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout, data_alignment=align))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def region(buffer, mask, indices=None, shape=None):
    # TensorRegion describes geometry. Ascend copy/fill operator contracts own
    # read/write effects; the old fixture's mask documents the operand role.
    if mask not in (1, 2, 3):
        raise ValueError("Region fixture access mask must be read, write or read-write")
    indices = [0] * len(buffer.ty.shape) if indices is None else indices
    shape = list(buffer.ty.shape) if shape is None else shape
    return ir.TensorRegion(
        buffer, [ir.Range.from_min_extent(index, extent) for index, extent in zip(indices, shape)], ir.TensorRegionType()
    )


def copy(src, dst, *, src_indices=None, dst_indices=None, src_shape=None, dst_shape=None, annotations=None):
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"),
            ir.Op.get("tl.tileop.ascend_copy"),
            [region(src, 1, src_indices, src_shape), region(dst, 2, dst_indices, dst_shape)],
            annotations or {},
            ir.UnknownLoc(),
        )
    )


def fill(dst, value, *, indices=None, shape=None):
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"),
            ir.Op.get("tl.tileop.fill"),
            [region(dst, 2, indices, shape), value],
            {},
            ir.UnknownLoc(),
        )
    )


def annotation(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def attribute(node, key, value, body, *, loc=None):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    if not isinstance(body, ir.SeqStmt):
        body = ir.SeqStmt(body if isinstance(body, (list, tuple)) else [body])
    return ir.RegionStmt(
        "tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body, loc=ir.UnknownLoc() if loc is None else loc
    )


def attribute_fields(node):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])


def attributes(value, key):
    return [node for node in nodes(value, ir.RegionStmt) if node.op.name == "tl.attr" and attribute_fields(node)["key"] == key]


def scheduled_fields(node):
    if (
        not isinstance(node, ir.Bind)
        or not isinstance(node.value, ir.Call)
        or not isinstance(node.value.op, ir.Op)
        or node.value.op.name != SCHEDULED_BIND
    ):
        return None
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.value.args[1], node.value.attrs["annotation_schema"])


def scheduled_bind(binding, stage, task):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"stage": stage, "task": task})
    value = ir.Call(
        SCHEDULED_BIND,
        [binding.value, payload],
        attrs={"schema_version": 1, "operand_count": 1, "annotation_schema": schema},
        loc=binding.value.loc,
    )
    return ir.Bind(binding.var, value, loc=binding.loc)


def kernel(
    body,
    *,
    buffers=(),
    params=(),
    annotations=None,
    allocation_stage=None,
    allocation_core=7,
    allocation_cost=None,
):
    # Allocation definitions must dominate metadata operands that refer to
    # them. Constant root attributes retain their original scope; maps with
    # variable operands are statements after those definitions.
    constants, operands = {}, {}
    for key, value in (annotations or {}).items():
        variables = []
        # The annotation codec visits expression-valued map keys as operands;
        # generic Map reflection does not expose those keys as SSA uses.
        _, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({key: value})
        tvm_ffi.structural_walk(payload, (ir.Var, variables.append))
        (operands if variables else constants)[key] = value
    statements = [allocate(buffer) for buffer in buffers]
    if allocation_stage is not None:
        task = {"core_mask": allocation_core} if allocation_core else {}
        if allocation_cost is not None:
            latency, ii = allocation_cost
            task.update(latency=latency, ii=ii)
        statements = [scheduled_bind(binding, allocation_stage, task) for binding in statements]
    if operands:
        statements.append(annotation(operands))
    statements.extend(body.seq if isinstance(body, ir.SeqStmt) else body if isinstance(body, (list, tuple)) else [body])
    root = attribute(None, "tl.sblock", {"name": "tilelang_root", "no_realize": True, "annotations": constants}, statements)
    function = tirx.Function(list(params), [root]).with_attr("global_symbol", "main").with_attr("target", tvm.target.Target("ascend"))
    return ir.IRModule({"main": function})


def nodes(value, kind):
    if isinstance(value, ir.IRModule):
        value = value["main"].body
    elif isinstance(value, tirx.Function):
        value = value.body
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="post")
    return found


def calls(value, name):
    return [node for node in nodes(value, ir.Call) if isinstance(node.op, ir.Op) and node.op.name == name]


def annotations(value):
    for call in nodes(value, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            yield tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
    for node in nodes(value, ir.RegionStmt):
        if node.op.name != "tl.attr":
            continue
        fields = attribute_fields(node)
        if fields["key"] == "tl.sblock":
            yield fields["value"].get("annotations", {})
        elif fields["key"] == "tl.loop.annotations":
            yield fields["value"]


def loop_annotations(loop):
    """Read directives belonging to this logical loop, retaining lexical scope."""
    result = dict(loop.annotations)

    def merge(fields):
        for key, value in fields.items():
            if key == "conflict_hint" and key in result:
                result[key] = [*result[key], *value]
            else:
                result[key] = value

    def read(body):
        for statement in body.seq:
            if isinstance(statement, ir.SeqStmt):
                read(statement)
            elif isinstance(statement, ir.RegionStmt) and statement.op.name == "tl.attr":
                fields = attribute_fields(statement)
                if fields["key"] == "tl.loop.annotations":
                    merge(fields["value"])
                    read(statement.body)
            elif isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call):
                call = statement.value
                if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
                    merge(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"]))

    inner = loop
    while (
        len(inner.body.seq) == 1
        and isinstance(inner.body.seq[0], ir.For)
        and inner.kind == ir.ForKind.PARALLEL
        and inner.body.seq[0].kind == ir.ForKind.PARALLEL
    ):
        inner = inner.body.seq[0]
    read(inner.body)
    return result


def serial(variable, begin, extent, body, *, annotations=None):
    """Build a native serial loop from a test's declared iteration domain."""
    statements = list(body.seq) if isinstance(body, ir.SeqStmt) else list(body) if isinstance(body, (list, tuple)) else [body]
    if annotations:
        statements = [attribute(None, "tl.loop.annotations", annotations, statements)]
    begin = ir.const(begin) if isinstance(begin, int) else begin
    extent = ir.const(extent) if isinstance(extent, int) else extent
    return ir.For(variable, begin, begin + extent, ir.const(1), ir.ForKind.DEFAULT, statements)


def schedule_snapshots(program, pass_names, *, auto_schedule=True, infer_layouts=False):
    """Run scheduling prerequisites and stop at the last requested boundary."""
    pipeline = (
        ascend.NormalizeControlFlowForSchedule,
        ascend.NormalizeConflictHints,
        ascend.MaterializeScheduleUnits,
        ascend.AssignCore,
        ascend.AnnotateMultiBufferEligible,
        ascend.EstimateLatency,
        ascend.AutoSchedule,
        ascend.PrepareMultiBuffer,
        ascend.ResolveCore,
        ascend.InsertSync,
        ascend.MaterializeMultiBuffer,
        ascend.LowerScheduledTIR,
        ascend.RestoreWhileLoops,
    )
    if not auto_schedule:
        pipeline = tuple(factory for factory in pipeline if factory != ascend.EstimateLatency)

    def use_source_order(mod):
        def assign_region(node):
            if node.op.name != "tl.attr":
                return node
            fields = attribute_fields(node)
            if fields["key"] == "tl.schedule_unit" and int(fields["node"]["stage"]) < 0:
                return attribute({**dict(fields["node"]), "stage": 0}, fields["key"], fields["value"], node.body, loc=node.loc)
            return node

        def assign_binding(node):
            fields = scheduled_fields(node)
            if fields is not None and int(fields["stage"]) < 0:
                binding = ir.Bind(node.var, node.value.args[0], loc=node.loc)
                return scheduled_bind(binding, 0, fields["task"])
            return node

        return tvm_ffi.structural_map(mod, [(ir.RegionStmt, assign_region), (ir.Bind, assign_binding)])

    requested = set(pass_names)
    available = {"tl." + factory.__name__ for factory in pipeline}
    if not requested:
        raise ValueError("At least one scheduling pass must be requested")
    if not requested <= available:
        raise ValueError(f"Unknown or disabled scheduling passes: {sorted(requested - available)}")
    snapshots = {}
    target = tvm.target.Target("ascend")
    with target:
        mod = ir.IRModule({"main": program.with_attr("global_symbol", "main")})
        mod = tilelang.transform.NormalizeAliasAnnotations()(mod)
        mod = tilelang.transform.LowerTensorSpec()(mod)
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch(
            lower_grid_binding=True,
            lower_thread_binding=False,
            default_threads=None,
            unsupported_annotations=["cluster_dims"],
            launch_dim_tags=["cthread"],
        )(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = ascend.RewriteDualCopy()(mod)
        mod = ascend.UnrollLoopSkipVF()(mod)
        mod = tilelang.transform.Simplify()(mod)
        if infer_layouts:
            mod = ascend.AscendLayoutInference()(mod)
            mod = ascend.NormalizeAscendFractalStorage()(mod)
        for factory in pipeline:
            mod = use_source_order(mod) if not auto_schedule and factory == ascend.AutoSchedule else factory()(mod)
            name = "tl." + factory.__name__
            if name in requested:
                snapshots[name] = mod
                if snapshots.keys() == requested:
                    return snapshots


def schedule(program, through="InsertSync", *, auto_schedule=True, infer_layouts=False):
    name = "tl." + through
    return schedule_snapshots(program, {name}, auto_schedule=auto_schedule, infer_layouts=infer_layouts)[name]


def disjoint_argument_fixture(value, *pairs):
    """Copy a prepared fixture with explicit disjoint external tensor pairs.

    These are caller preconditions for tests of ring allocation and conflict
    hints. The original function continues to allow aliasing. A function
    attribute alone is not an Ascend scheduling hint, so declare the pairs at
    the kernel root and every logical loop where they apply.
    """
    parameters = {parameter.name: parameter for parameter in value["main"].params}
    hints = []
    for left, right in pairs:
        a, b = parameters[left], parameters[right]
        assert not a.same_as(b)
        assert isinstance(a.ty, tirx.TensorType) and isinstance(b.ty, tirx.TensorType)
        hints.append([a, b, ir.IntImm("int32", -1), ir.IntImm("bool", False)])

    def mark_loop(loop):
        return ir.For(
            loop.var,
            loop.begin,
            loop.end,
            loop.step,
            loop.kind,
            loop.body,
            annotations={**dict(loop.annotations), "conflict_hint": [*loop.annotations.get("conflict_hint", []), *hints]},
            loc=loop.loc,
        )

    def mark_root(node):
        if node.op.name != "tl.attr":
            return node
        fields = attribute_fields(node)
        if fields["key"] != "tl.sblock" or fields["value"]["name"] != "tilelang_root":
            return node
        metadata = dict(fields["value"])
        annotations = dict(metadata.get("annotations", {}))
        annotations["conflict_hint"] = [*annotations.get("conflict_hint", []), *hints]
        metadata["annotations"] = annotations
        return attribute(fields["node"], fields["key"], metadata, node.body, loc=node.loc)

    return tvm_ffi.structural_map(value, [(ir.For, mark_loop), (ir.RegionStmt, mark_root)])


def statements(value, ancestors=()):
    """Yield native statements with lexical parents, without inspecting text."""
    if isinstance(value, ir.IRModule):
        value = value["main"].body
    elif isinstance(value, tirx.Function):
        value = value.body
    yield value, ancestors
    parents = (*ancestors, value)
    if isinstance(value, ir.SeqStmt):
        children = value.seq
    elif isinstance(value, ir.If):
        children = [value.then_case] + ([value.else_case] if value.else_case is not None else [])
    elif isinstance(value, (ir.RegionStmt, ir.For, ir.While)):
        children = [value.body]
    else:
        children = []
    for child in children:
        yield from statements(child, parents)


def allocated_buffer(value, name):
    for binding in nodes(value, ir.Bind):
        expression = binding.value.args[0] if scheduled_fields(binding) is not None else binding.value
        if (
            tirx.is_tensor_var(binding.var)
            and binding.var.name == name
            and isinstance(expression, ir.Call)
            and isinstance(expression.op, ir.Op)
            and expression.op.name == "tirx.alloc_tensor"
        ):
            return binding.var
    raise StopIteration(name)


def single_statement(value):
    """Unwrap one-statement native sequences without crossing other scopes."""
    while isinstance(value, ir.SeqStmt) and len(value.seq) == 1:
        value = value.seq[0]
    return value


def attribute_key(value):
    return attribute_fields(value)["key"] if isinstance(value, ir.RegionStmt) and value.op.name == "tl.attr" else None


def sblocks(value, name=None):
    return [node for node in attributes(value, "tl.sblock") if name is None or attribute_fields(node)["value"]["name"] == name]


def is_sblock(value):
    return attribute_key(value) == "tl.sblock"


def sblock_metadata(value):
    assert is_sblock(value)
    return attribute_fields(value)["value"]


def sblock(name, body, *, buffers=(), annotations=None, reads=(), writes=()):
    """Construct the native lexical block with dominating allocation Binds."""
    constants, operands = {}, {}
    for key, value in (annotations or {}).items():
        variables = []
        _, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({key: value})
        tvm_ffi.structural_walk(payload, (ir.Var, variables.append))
        (operands if variables else constants)[key] = value
    body = list(body.seq) if isinstance(body, ir.SeqStmt) else list(body) if isinstance(body, (list, tuple)) else [body]
    body = [*[allocate(buffer) for buffer in buffers], *([annotation(operands)] if operands else []), *body]
    fields = {"name": name, "no_realize": True, "annotations": constants}
    if reads:
        fields["reads"] = list(reads)
    if writes:
        fields["writes"] = list(writes)
    return attribute(None, "tl.sblock", fields, body)


def launch_thread(variable, tag, extent, body):
    extent = ir.const(extent) if isinstance(extent, int) else extent
    body = body if isinstance(body, ir.SeqStmt) else ir.SeqStmt(body if isinstance(body, (list, tuple)) else [body])
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), extent], [variable], {}, body)


def task_core_masks(value, op_name):
    masks = {int(attribute_fields(task)["node"]["core_mask"]) for task in attributes(value, "tl.ascend_task") if calls(task.body, op_name)}
    for binding in nodes(value, ir.Bind):
        fields = scheduled_fields(binding)
        if fields is not None and "core_mask" in fields["task"] and calls(binding, op_name):
            masks.add(int(fields["task"]["core_mask"]))
    return masks


def access_tensor(value):
    """Return the actual tensor identity of an access or tensor binding.

    Views remain distinct Vars; this does not identify aliases by their names,
    storage pointers, shapes, or dtypes. Callers that need physical ownership
    must trace the corresponding declaration/view bindings explicitly.
    """
    if isinstance(value, (ir.TensorLoad, ir.TensorRegion)):
        value = value.source
    elif isinstance(value, ir.TensorStore):
        value = value.dest
    elif isinstance(value, ir.Bind):
        value = value.var
    if not tirx.is_tensor_var(value):
        raise TypeError("Expected a native tensor Var, access, region, or binding")
    return value


def allocations(value, *, include_scheduled=False):
    """Return real alloc_tensor Binds in lexical statement order.

    Scheduled bindings are a different intermediate contract. They are included
    only on request, and the original wrapper Bind is returned unchanged. View,
    declaration, scalar, and arbitrary tensor-returning calls are not allocations.
    """
    result = []
    for statement, _ in statements(value):
        if not isinstance(statement, ir.Bind) or not tirx.is_tensor_var(statement.var):
            continue
        expression = statement.value
        if scheduled_fields(statement) is not None:
            if not include_scheduled:
                continue
            expression = expression.args[0]
        if isinstance(expression, ir.Call) and isinstance(expression.op, ir.Op) and expression.op.name == "tirx.alloc_tensor":
            result.append(statement)
    return result


def allocated_tensors(value, *, include_scheduled=False):
    """Return the defining tensor Vars of allocations selected above."""
    return [binding.var for binding in allocations(value, include_scheduled=include_scheduled)]


def walk_ir(value, visitor):
    """Visit each native Expr/Stmt occurrence in postorder, including type uses.

    The registered structural protocol controls scope and descent. Metadata and
    dynamic types may contain expression uses; scalar reflection fields are not
    reported to the visitor. Repeated uses are not identity-deduplicated.
    """
    if isinstance(value, ir.IRModule):
        value = value["main"].body
    elif isinstance(value, tirx.Function):
        value = value.body
    return tvm_ffi.structural_walk(value, ((ir.Expr, ir.Stmt), visitor), order="post")


def substitute(value, replacements):
    """Simultaneously replace scalar identities for native test evaluation.

    Values keep the replaced scalar's dtype. Explicit casts remain part of the
    expression. Tensor shapes/layouts and their defining/use identities update
    together, including free tensor Vars. Replacing a binder by a constant is
    rejected: eliminating function parameters or loop definitions is not an
    observation operation and requires a dedicated owning transform.
    """
    normalized = {}
    for variable, replacement in replacements.items():
        if not isinstance(variable, ir.Var) or not isinstance(variable.ty, ir.PrimType):
            raise TypeError("Substitution keys must be primitive native Vars")
        if isinstance(replacement, (int, float, bool)):
            replacement = ir.const(replacement, variable.ty.dtype)
        if not isinstance(replacement, ir.Expr) or not tvm_ffi.structural_equal(variable.ty, replacement.ty):
            raise TypeError("Substitution must preserve each scalar dtype")
        normalized[variable] = replacement
    memo = {}

    def replace(variable, mutator):
        if variable in normalized:
            replacement = normalized[variable]
            if mutator.def_region_kind() != tvm_ffi.DefRegionKind.NONE and not isinstance(replacement, ir.Var):
                raise ValueError("Cannot replace a bound variable with a non-Var")
            return replacement
        # Bind's native hook may refine a result type and install a newer Var
        # after our callback. Its identity must win over the free-use memo.
        remapped = mutator.var_remap_get(variable)
        if remapped is not None:
            return remapped
        if variable in memo:
            return memo[variable]
        # A simple binder defines the Var, not variables used in its type.
        # Pattern binders (for example shape-bound function parameters) retain
        # the native protocol's distinction and reject constant binder rewrites.
        ty = (
            mutator.with_def_region_kind(tvm_ffi.DefRegionKind.NONE, lambda: mutator.mutate(variable.ty))
            if mutator.def_region_kind() == tvm_ffi.DefRegionKind.DEF_SIMPLE
            else mutator.mutate(variable.ty)
        )
        mapped = variable if ty.same_as(variable.ty) else type(variable)(variable.name, ty)
        memo[variable] = mapped
        return mapped

    return tvm_ffi.structural_mutate(value, (ir.Var, replace))


def call_annotations(call):
    """Decode a TL schema-v1 Call's metadata payload without losing operands."""
    if not isinstance(call, ir.Call):
        raise TypeError("Expected a native annotated Call")
    if call.attrs is None or "annotation_schema" not in call.attrs:
        return {}
    if int(call.attrs["schema_version"]) != 1:
        raise ValueError("Unsupported TileLang annotation schema")
    count = int(call.attrs["operand_count"])
    if count < 0 or len(call.args) != count + 1:
        raise ValueError("Annotated Call needs declared operands and one payload")
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[count], call.attrs["annotation_schema"])


def sblock_annotations(block):
    """Read constant and SSA annotations belonging to one lexical SBlock.

    A single-statement SeqStmt is an unambiguous wrapper; a general sequence is
    not a block carrier. Only direct annotation statements (and transparent
    nested sequences) belong here. Child blocks, loops, and regions keep their
    own metadata, even when a child defines the same annotation key.
    """
    block = single_statement(block)
    if not is_sblock(block):
        raise TypeError("Expected one native tl.sblock RegionStmt")
    result = {}
    storage_maps = {"tl.buffer_versions_map", "tl.manual_multi_buffer", "tl.buffer_version_mode"}

    def merge(fields):
        for key, value in fields.items():
            if key not in storage_maps:
                result[key] = value
                continue
            # Each entry may follow a different allocation definition. An
            # empty or later map is not a replacement for earlier entries.
            entries = dict(result.get(key, {}))
            for storage, entry in value.items():
                if storage in entries:
                    previous = entries[storage]
                    # The production codec reads version counts as integers,
                    # accepting both scalar metadata and typed IntImm values.
                    previous = int(previous) if isinstance(previous, ir.IntImm) else previous
                    current = int(entry) if isinstance(entry, ir.IntImm) else entry
                    if not tvm_ffi.structural_equal(previous, current):
                        raise ValueError(f"Conflicting {key} for storage {storage}")
                entries[storage] = entry
            result[key] = entries

    merge(sblock_metadata(block).get("annotations", {}))

    def collect(body):
        for statement in body.seq:
            if isinstance(statement, ir.SeqStmt):
                collect(statement)
            elif isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call):
                call = statement.value
                if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
                    merge(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"]))

    collect(block.body)
    return result
