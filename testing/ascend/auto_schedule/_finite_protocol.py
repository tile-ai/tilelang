"""Check the seven reviewed finite InsertSync fixtures against current output.

This is an observer for fixed 32-bit row-major allocations and zero-stage
protocols. Unknown layouts, views, alias groups, calls and runtime predicates
are rejected. It does not claim a proof for arbitrary kernels or NPU execution.
"""

from itertools import product

import tvm_ffi
from tvm_tirx import ir, sym, tirx
from testing.ascend._ir import is_annotation, allocated_tensors, annotations, attribute_fields, attributes, nodes, scheduled_fields
from testing.ascend.auto_schedule._finite_protocol_events import extract, memory_signature
from testing.ascend.auto_schedule._finite_protocol_order import prove


_SCHEMAS = {
    "parent_child": {
        "shape": (4, 128),
        "local": {"ub": (128,)},
        "loops": {"i": 4, "j": 1},
        "copies": 4,
        "stores": 0,
        "parameters": ("p", "q"),
    },
    "branch_switch": {"shape": (4, 64), "local": {"ub": (64,), "out": (64,)}, "loops": {"i": 4, "j": 2}, "copies": 4, "stores": 2},
    "iteration_ring": {"shape": (4, 64), "local": {"ub": (64,)}, "loops": {"i0": 4}, "copies": 2, "stores": 0, "versions": {"ub": 2}},
    "counter_ring_one": {
        "shape": (4, 64),
        "local": {"ub": (64,)},
        "loops": {"i0": 4},
        "copies": 2,
        "stores": 2,
        "versions": {"ub": 2},
        "counter": True,
    },
    "counter_ring_two": {
        "shape": (8, 64),
        "local": {"ub": (64,)},
        "loops": {"i0": 4, "i1": 4},
        "copies": 4,
        "stores": 3,
        "versions": {"ub": 2},
        "counter": True,
    },
    "manual_ring": {"shape": (4, 64), "local": {"ub": (2, 64)}, "loops": {"i": 4}, "copies": 2, "stores": 0, "manual": {"ub": 2}},
    "cube_bridge": {"loops": {}, "copies": 3, "stores": 0},
}
_CALLS = {
    "tirx.alloc_tensor",
    "tl.ascend.scheduled_bind",
    "tl.tileop.ascend_copy",
    "tl.ascend_pipe_barrier",
    "tl.ascend_set_flag",
    "tl.ascend_wait_flag",
    "tl.ascend_cross_core_set_flag",
    "tl.ascend_cross_core_wait_flag",
}


def _domain(module, case):
    schema = _SCHEMAS[case]
    function = module["main"]
    assert "tirx.noalias" not in function.attrs, "The reviewed original fixtures permit global aliasing"
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    tensors = list(
        dict.fromkeys(
            [parameter for parameter in function.params if isinstance(parameter.ty, tirx.TensorType)]
            + allocated_tensors(module, include_scheduled=True)
        )
    )
    variables = list(
        dict.fromkeys([*function.params, *(node.var for node in nodes(module, ir.For)), *(node.var for node in nodes(module, ir.Bind))])
    )
    assert len({variable.name for variable in variables}) == len(variables), "Ambiguous variable names are outside the finite observer"
    assert len({tensor.name for tensor in tensors}) == len(tensors)
    if case == "cube_bridge":
        expected = {
            name: ("float32", (64,), scope)
            for name, scope in (
                ("shared", "global"),
                ("unrelated", "global"),
                ("ub0", "shared.dyn"),
                ("ub1", "shared.dyn"),
                ("l1", "shared.l1"),
            )
        }
    else:
        expected = {name: ("float32", schema["shape"], "global") for name in ("A", "C")}
        expected.update({name: ("float32", shape, "shared.dyn") for name, shape in schema["local"].items()})
        if schema.get("counter"):
            expected["epoch"] = ("int32", (1,), "local.var")
    descriptions = {}
    for tensor in tensors:
        ty = tensor.ty
        assert ty.has_default_layout(), "Only the reviewed default layouts are modeled"
        assert all(isinstance(extent, ir.IntImm) for extent in ty.shape), "Unknown tensor extent"
        shape = tuple(int(extent) for extent in ty.shape)
        descriptions[tensor.name] = (str(ty.dtype), shape, str(ty.scope))
        assert descriptions[tensor.name] == expected.get(tensor.name), "Tensor outside the selected fixed fixture"
        for coordinate in product(*(range(extent) for extent in shape)):
            mapped = ty.layout.apply(*coordinate)
            offset = 0
            for index, extent in zip(coordinate, shape):
                offset = offset * extent + index
            assert set(mapped) == {"m"} and int(mapped["m"]) == offset, "The actual layout must be injective row-major"
    assert descriptions == expected
    parameters = tuple(parameter.name for parameter in function.params if not isinstance(parameter.ty, tirx.TensorType))
    assert parameters == schema.get("parameters", ()), "Unknown runtime parameter"
    assert all(str(parameter.ty.dtype) == "bool" for parameter in function.params if not isinstance(parameter.ty, tirx.TensorType))
    loops = {}
    analyzer = sym.Analyzer()
    for loop in nodes(module, ir.For):
        bounds = [analyzer.simplify(value) for value in (loop.begin, loop.end, loop.step)]
        assert all(isinstance(value, ir.IntImm) for value in bounds), "Unknown loop domain"
        assert loop.kind == ir.ForKind.DEFAULT and str(loop.var.ty.dtype) == "int32"
        assert loop.var.name not in loops and int(bounds[0]) == 0 and int(bounds[2]) == 1
        loops[loop.var.name] = int(bounds[1])
    assert loops == schema["loops"], "Loop outside the selected fixed fixture"
    for unit in attributes(module, "tl.schedule_unit"):
        assert int(attribute_fields(unit)["node"]["stage"]) == 0, "Unmodeled stage"
    for task in attributes(module, "tl.ascend_task"):
        core = int(attribute_fields(task)["node"]["core_mask"])
        assert core in ({1, 2, 4} if case == "cube_bridge" else {5}), "Unmodeled core"
    for branch in nodes(module, ir.If):
        assert all(str(load.source.ty.scope) == "local.var" for load in nodes(branch.condition, ir.TensorLoad)), (
            "Unmodeled runtime memory predicate"
        )
    calls = nodes(module, ir.Call)
    assert all(is_annotation(call) or (isinstance(call.op, ir.Op) and call.op.name in _CALLS) for call in calls), (
        "Unmodeled call or alias view"
    )
    for binding in nodes(module, ir.Bind):
        fields = scheduled_fields(binding)
        assert fields is not None, "Unmodeled unscheduled binding"
        assert int(fields["stage"]) == 0, "Unmodeled binding stage"
        value = binding.value.args[0]
        allocation = isinstance(value, ir.Call) and value.op.name == "tirx.alloc_tensor"
        # Root allocations belong to the common stage; executable scalar
        # definitions belong to the two AIVs, just like statement units.
        assert int(fields["task"].get("core_mask", 0)) == (7 if allocation else 5), "Unmodeled binding core"
    copies = [call for call in calls if call.op.name == "tl.tileop.ascend_copy"]
    assert len(copies) == schema["copies"] and len(nodes(module, ir.TensorStore)) == schema["stores"]
    for call in copies:
        assert int(call.attrs["operand_count"]) == 2 and int(call.attrs["schema_version"]) == 1 and len(call.args) == 3
        assert not tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[2], call.attrs["annotation_schema"]), "Unmodeled copy contract"
    versions, counters, manual = {}, {}, {}
    for mapping in [*annotations(module), *(dict(loop.annotations) for loop in nodes(module, ir.For))]:
        assert not mapping.get("layout_map", {}), "Unmodeled layout annotation"
        assert all(not aliases for aliases in mapping.get("tl.buffer_alias_map", {}).values()), (
            "Local reuse aliases are outside this observer"
        )
        for key, values in (("tl.buffer_versions_map", versions), ("tl.manual_multi_buffer", manual)):
            for tensor, count in mapping.get(key, {}).items():
                assert tensor.name not in values or values[tensor.name] == int(count)
                values[tensor.name] = int(count)
        for tensor, counter in mapping.get("tl.multi_buffer_counter_map", {}).items():
            assert tensor.name not in counters or counters[tensor.name] == counter.name
            counters[tensor.name] = counter.name
    assert versions == schema.get("versions", {}) and manual == schema.get("manual", {})
    assert counters == ({"ub": "epoch"} if schema.get("counter") else {})
    return descriptions


def assert_finite_protocol(before, after, case):
    """Preserve accesses/values/clocks and prove actual completion and tokens."""
    assert case in _SCHEMAS, "Only the seven explicitly reviewed fixtures are supported"
    assert _domain(before, case) == _domain(after, case)
    parameters = _SCHEMAS[case].get("parameters", ())
    reports = []
    for values in product((False, True), repeat=len(parameters)):
        valuation = dict(zip(parameters, values))
        old, new = extract(before, valuation), extract(after, valuation)
        assert memory_signature(old) == memory_signature(new), "InsertSync changed accesses, scalar values or storage clocks"
        report = prove(new, require_drain=case != "cube_bridge")
        assert report["safe"], {key: report[key] for key in ("token_errors", "cycle", "unordered", "undrained")}
        if case == "manual_ring":
            for event in new:
                if event["kind"] != "memory":
                    continue
                assert len(event["loops"]) == 1 and event["loops"][0][0] == "i"
                iteration = event["loops"][0][1]
                assert iteration in (0, 2)
                for access in event["reads"] + event["writes"]:
                    if access["name"] == "ub":
                        assert {index[0] for index in access["indices"]} == {iteration % 2}
        reports.append({"parameters": valuation, "proof": report})
    return reports
