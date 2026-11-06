"""Finite synchronization proof from actual native statements and event IDs.

The graph uses hardware pipeline order, flag completion and barriers only.
Source order between different pipelines never adds a completion edge.
Unversioned local addresses are compared conservatively, even when metadata
will later split a storage into ring slots. Clock metadata is reported too.
"""

from itertools import product

import tvm_ffi
from tvm_tirx import ir, sym
from testing.ascend._ir import is_annotation, allocated_tensors, annotations, attribute_fields, nodes, scheduled_fields, substitute


def extract(module, parameters=None):
    parameters = parameters or {}
    function = module["main"]
    assert "tirx.noalias" not in function.attrs
    tensors = [value for value in function.params if hasattr(value.ty, "scope")]
    tensors += allocated_tensors(module, include_scheduled=True)
    tensors = list(dict.fromkeys(tensors))
    assert len({value.name for value in tensors}) == len(tensors)
    identities = {value: index for index, value in enumerate(tensors)}
    scopes = {value: str(value.ty.scope) for value in tensors}
    versions, counters = {}, {}
    maps = list(annotations(module)) + [dict(loop.annotations) for loop in nodes(module, ir.For)]
    for mapping in maps:
        for key, destination in (("tl.buffer_versions_map", versions), ("tl.multi_buffer_counter_map", counters)):
            for storage, value in mapping.get(key, {}).items():
                assert storage in identities
                value = int(value) if key == "tl.buffer_versions_map" else value
                assert storage not in destination or (
                    destination[storage] == value if key == "tl.buffer_versions_map" else destination[storage].same_as(value)
                )
                destination[storage] = value
    assert all(0 < count <= 5 for count in versions.values())
    environment = {}
    for parameter in function.params:
        if not hasattr(parameter.ty, "scope"):
            assert parameter.name in parameters and str(parameter.ty.dtype) == "bool", parameter
            environment[parameter] = ir.const(bool(parameters[parameter.name]), "bool")
    local_values = {}
    analyzer = sym.Analyzer()
    events = []

    def concrete(expression, environment):
        expression = substitute(expression, environment)

        def read_local(load):
            assert scopes[load.source] == "local.var", ("runtime memory predicate or coordinate", load)
            index = tuple(concrete(index, environment) for index in load.indices)
            assert (load.source, index) in local_values, ("uninitialized local counter", load)
            return ir.const(local_values[load.source, index], load.ty.dtype)

        expression = tvm_ffi.structural_map(expression, [(ir.TensorLoad, read_local)])
        expression = analyzer.simplify(expression)
        assert isinstance(expression, ir.IntImm), ("unmodeled dynamic expression", expression)
        return int(expression)

    def access(storage, indices, loops):
        item = {"storage": identities[storage], "name": storage.name, "scope": scopes[storage], "indices": indices}
        if storage in versions:
            if storage in counters:
                clock = concrete(ir.TensorLoad(counters[storage], [0]), current_environment)
                clock_name = counters[storage].name
            else:
                assert loops
                clock_name, clock = loops[-1]
            item["ring"] = {"versions": versions[storage], "clock": clock_name, "value": clock, "slot": clock % versions[storage]}
        return item

    def scalar_accesses(expression, environment, loops):
        assert not nodes(expression, ir.Call), ("unmodeled scalar call", expression)
        return [
            access(load.source, [[concrete(index, environment) for index in load.indices]], loops)
            for load in nodes(expression, ir.TensorLoad)
        ]

    def scalar_value(expression, environment):
        expression = substitute(expression, environment)

        def read_counter(load):
            if scopes[load.source] != "local.var":
                return load
            return ir.const(concrete(load, environment), load.ty.dtype)

        expression = tvm_ffi.structural_map(expression, [(ir.TensorLoad, read_counter)])
        expression = analyzer.simplify(expression)
        # The caller validates unique variable names and exact tensor types.
        # Preserve the selected scalar expression, including external reads;
        # only the finite local counter state is evaluated here.
        return {"dtype": str(expression.ty.dtype), "expression": str(expression)}

    def region_access(region, environment, loops):
        assert isinstance(region, ir.TensorRegion)
        ranges = [
            range(concrete(value.begin, environment), concrete(value.begin, environment) + concrete(value.extent, environment))
            for value in region.region
        ]
        indices = [list(index) for index in product(*ranges)]
        assert 0 < len(indices) <= 128
        return access(region.source, indices, loops)

    current_environment = environment

    def visit(statement, environment, loops=(), core=None, stage=0, vector=False, guards=()):
        nonlocal current_environment
        current_environment = environment
        context = {"loops": [list(loop) for loop in loops], "core": core, "stage": stage, "guards": list(guards)}

        def emit(**event):
            assert stage == 0 and core in (1, 2, 4, 5), ("unsupported stage or core", context)
            events.append({**context, **event})

        if isinstance(statement, ir.SeqStmt):
            for child in statement.seq:
                visit(child, environment, loops, core, stage, vector, guards)
        elif isinstance(statement, ir.RegionStmt):
            assert statement.op.name in {"tl.attr", "tl.ascend_block", "tl.ascend.task", "tl.ascend.schedule_unit"}, (
                "unsupported region",
                statement.op.name,
            )
            fields = attribute_fields(statement)
            key = fields["key"]
            if key == "tl.ascend_task":
                core = int(fields["node"]["core_mask"])
            elif key == "tl.schedule_unit":
                stage = int(fields["node"]["stage"])
            elif key == "tl.sblock":
                name = fields["value"]["name"]
                assert name in ("tilelang_root", "root", "SIMD_VF"), ("unsupported block", name)
                vector = vector or name == "SIMD_VF"
            else:
                assert key == "tl.loop.annotations", ("unsupported attribute", key)
            visit(statement.body, environment, loops, core, stage, vector, guards)
        elif isinstance(statement, ir.For):
            assert statement.kind == ir.ForKind.DEFAULT
            begin, end, step = [concrete(value, environment) for value in (statement.begin, statement.end, statement.step)]
            assert step > 0 and 0 <= len(range(begin, end, step)) <= 16
            for iteration in range(begin, end, step):
                visit(
                    statement.body,
                    {**environment, statement.var: ir.const(iteration, statement.var.ty.dtype)},
                    (*loops, (statement.var.name, iteration)),
                    core,
                    stage,
                    vector,
                    guards,
                )
        elif isinstance(statement, ir.If):
            decision = concrete(statement.condition, environment)
            assert decision in (0, 1)
            selected = statement.then_case if decision else statement.else_case
            if selected is not None:
                visit(selected, environment, loops, core, stage, vector, (*guards, [str(statement.condition), bool(decision)]))
        elif isinstance(statement, ir.Bind):
            fields = scheduled_fields(statement)
            value = statement.value.args[0] if fields is not None else statement.value
            if isinstance(value, ir.Call) and isinstance(value.op, ir.Op) and value.op.name == "tirx.alloc_tensor":
                return
            assert fields is not None and int(fields["stage"]) == 0 and int(fields["task"]["core_mask"]) == 5
            assert not nodes(value, ir.TensorLoad), "The finite model only supports pure SSA bindings"
            environment[statement.var] = ir.const(concrete(value, environment), statement.var.ty.dtype)
            events.append(
                {
                    "kind": "bind",
                    "pipe": "S",
                    "core": 5,
                    "stage": 0,
                    "loops": context["loops"],
                    "guards": context["guards"],
                    "name": statement.var.name,
                    "value": int(environment[statement.var]),
                }
            )
        elif isinstance(statement, ir.TensorStore):
            reads = scalar_accesses(statement.value, environment, loops)
            indices = [[concrete(index, environment) for index in statement.indices]]
            writes = [access(statement.dest, indices, loops)]
            emit(
                kind="memory",
                pipe="V" if vector else "S",
                operation="scalar",
                reads=reads,
                writes=writes,
                scalar_value=scalar_value(statement.value, environment),
            )
            if scopes[statement.dest] == "local.var":
                assert not vector
                local_values[statement.dest, tuple(indices[0])] = concrete(statement.value, environment)
        elif isinstance(statement, ir.Evaluate):
            call = statement.value
            assert isinstance(call, ir.Call) and isinstance(call.op, ir.Op)
            name = call.op.name
            if is_annotation(call):
                return
            if name == "tl.tileop.ascend_copy":
                source, destination = call.args[:2]
                direction = (scopes[source.source], scopes[destination.source])
                assert direction in (("global", "shared.dyn"), ("global", "shared.l1"), ("shared.dyn", "global"))
                pipe = "MTE2" if direction[0] == "global" else "MTE3"
                emit(
                    kind="memory",
                    pipe=pipe,
                    operation="copy",
                    reads=[region_access(source, environment, loops)],
                    writes=[region_access(destination, environment, loops)],
                )
            elif name == "tl.ascend_pipe_barrier":
                pipe = call.args[0].value.removeprefix("PIPE_")
                assert pipe in ("MTE2", "MTE3", "V")
                emit(kind="barrier", pipe=pipe)
            elif name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                kind = "set" if name.endswith("set_flag") else "wait"
                channel = call.args[0].value
                source, destination = channel.split("_")
                assert source in ("MTE2", "MTE3", "V") and destination in ("MTE2", "MTE3", "V")
                token = concrete(call.args[1], environment)
                assert core == 5 and 0 <= token < 8
                emit(
                    kind=kind,
                    pipe=source if kind == "set" else destination,
                    channel=["local", core, channel, token],
                    token=token,
                    token_expression=str(call.args[1]),
                )
            elif name in ("tl.ascend_cross_core_set_flag", "tl.ascend_cross_core_wait_flag"):
                assert concrete(call.args[0], environment) == 4
                kind = "set" if name.endswith("set_flag") else "wait"
                pipe = call.args[1].value.removeprefix("PIPE_")
                assert pipe in ("MTE2", "MTE3") and core in (1, 2, 4)
                token = concrete(call.args[2], environment)
                if core == 2:
                    assert 0 <= token < 32
                    sid, slot = divmod(token, 16)
                    direction = "cube_to_vector" if kind == "set" else "vector_to_cube"
                else:
                    assert 0 <= token < 16
                    sid, slot = (0 if core == 1 else 1), token
                    direction = "vector_to_cube" if kind == "set" else "cube_to_vector"
                emit(kind=kind, pipe=pipe, channel=["mode4", direction, sid, slot], token=token, token_expression=str(call.args[2]))
            else:
                raise AssertionError(("unmodeled executable call", name))
        else:
            raise AssertionError(("unmodeled statement", type(statement).__name__))

    visit(function.body, environment)
    return events


def memory_signature(events):
    # Guard shells may be factored or duplicated without changing the selected
    # execution path. Actual operations, values, coordinates, stages and cores
    # must still be identical for each valuation.
    return [{key: value for key, value in event.items() if key != "guards"} for event in events if event["kind"] in ("memory", "bind")]
