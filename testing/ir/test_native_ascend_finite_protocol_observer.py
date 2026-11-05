"""Challenge the finite observer using current, original-consumer pass output."""

import json

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S
from testing.ascend._ir import allocated_tensors, annotation, attribute, attribute_fields, scheduled_bind, scheduled_fields, tensor
from testing.ascend.auto_schedule import test_cross_core_flags, test_sync_boundaries, test_sync_dependencies
from testing.ascend.auto_schedule._finite_protocol import assert_finite_protocol
from testing.ascend.auto_schedule._finite_protocol_events import extract, memory_signature
from testing.ascend.auto_schedule._finite_protocol_order import prove


CASES = {
    "parent_child": (test_sync_boundaries, "test_narrow_child_handshake_cannot_replace_wider_parent_dependency", ()),
    "branch_switch": (test_sync_boundaries, "test_branch_switch_acquires_previous_iteration_before_nested_overwrite", ()),
    "iteration_ring": (test_sync_dependencies, "test_ring_flags_use_the_data_clock", ("iteration", 1, False)),
    "counter_ring_one": (test_sync_dependencies, "test_ring_flags_use_the_data_clock", ("counter", 1, True)),
    "counter_ring_two": (test_sync_dependencies, "test_ring_flags_use_the_data_clock", ("counter", 2, True)),
    "manual_ring": (test_sync_dependencies, "test_manually_indexed_storage_keeps_the_lexical_clock_under_a_guard", ()),
    "cube_bridge": (test_cross_core_flags, "test_only_actual_cube_dependencies_can_order_aiv_accesses", (False,)),
}


@pytest.fixture(scope="module")
def actual_protocols(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-finite-protocols")
    captured = {}
    snapshots = {}
    for case, (module, name, arguments) in CASES.items():

        def observe(before, after, selected, *, expected=case):
            assert selected == expected and selected not in captured
            reports = assert_finite_protocol(before, after, selected)
            captured[selected] = (before, after, reports)
            snapshots[selected] = (ir.save_json(before), ir.save_json(after))
            return reports

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(module, "assert_finite_protocol", observe)
            getattr(module, name)(*arguments)
        assert case in captured
        case_directory = directory / case
        case_directory.mkdir()
        for phase, source in zip(("before", "after"), snapshots[case]):
            (case_directory / f"{phase}.json").write_text(source)
        (case_directory / "proof.json").write_text(json.dumps(captured[case][2], indent=2) + "\n")
    yield captured
    for case, (before, after, _) in captured.items():
        assert (ir.save_json(before), ir.save_json(after)) == snapshots[case]


def _reject(before, changed, case, directory, message):
    # A malformed mutation is not evidence that the observer rejects a legal
    # native program outside its finite proof domain.
    assert tirx.analysis.check_well_formed(changed)
    assert not tirx.analysis.undefined_vars(changed["main"])
    (directory / "changed.json").write_text(ir.save_json(changed))
    with pytest.raises(AssertionError, match=message) as error:
        assert_finite_protocol(before, changed, case)
    (directory / "rejection.txt").write_text(str(error.value) + "\n")


def _append_root(module, statement):
    changed = False

    def append(region):
        nonlocal changed
        fields = attribute_fields(region)
        if fields["key"] == "tl.sblock" and fields["value"]["name"] == "tilelang_root":
            assert not changed
            changed = True
            return attribute(fields["node"], fields["key"], fields["value"], [*region.body.seq, statement], loc=region.loc)
        return region

    result = tvm_ffi.structural_map(module, [(ir.RegionStmt, append)])
    assert changed
    return result


@pytest.mark.parametrize("case", CASES)
def test_current_consumer_output_has_a_complete_finite_proof(actual_protocols, case):
    before, after, reports = actual_protocols[case]
    assert "tirx.noalias" not in before["main"].attrs and "tirx.noalias" not in after["main"].attrs
    assert len(reports) == (4 if case == "parent_child" else 1)
    assert all(report["proof"]["safe"] for report in reports)


def test_changed_scalar_value_is_rejected_even_when_addresses_and_order_are_safe(actual_protocols, tmp_path):
    before, after, _ = actual_protocols["branch_switch"]
    changed = False

    def change_value(store):
        nonlocal changed
        if not changed and store.dest.name == "out":
            changed = True
            return ir.TensorStore(store.dest, store.indices, store.value + ir.const(1, "float32"), loc=store.loc)
        return store

    bad = tvm_ffi.structural_map(after, [(ir.TensorStore, change_value)])
    assert changed
    original_signature = memory_signature(extract(after))
    bad_events = extract(bad)
    bad_signature = memory_signature(bad_events)
    assert original_signature != bad_signature
    assert [{key: value for key, value in event.items() if key != "scalar_value"} for event in original_signature] == [
        {key: value for key, value in event.items() if key != "scalar_value"} for event in bad_signature
    ]
    assert prove(bad_events)["safe"]
    _reject(before, bad, "branch_switch", tmp_path, "changed accesses, scalar values or storage clocks")


@pytest.mark.parametrize("strides", [(128, 1), (1, 4)], ids=["padded-rows", "transposed"])
def test_nondefault_native_layout_is_rejected(actual_protocols, tmp_path, strides):
    before, after, _ = actual_protocols["branch_switch"]
    source = after["main"].params[0]
    replacement = tensor((4, 64), name=source.name, layout=tirx.TileLayout(S[(4, 64) : strides]))
    bad = tvm_ffi.structural_map(after, [(ir.Var, lambda variable: replacement if variable.same_as(source) else variable)])
    assert bad["main"].params[0].same_as(replacement)
    _reject(before, bad, "branch_switch", tmp_path, "reviewed default layouts")


def test_local_reuse_alias_is_rejected_inside_its_valid_scope(actual_protocols, tmp_path):
    before, after, _ = actual_protocols["branch_switch"]
    tensors = {value.name: value for value in allocated_tensors(after, include_scheduled=True)}
    bad = _append_root(after, annotation({"tl.buffer_alias_map": {tensors["ub"]: [tensors["out"]]}}))
    _reject(before, bad, "branch_switch", tmp_path, "Local reuse aliases")


def test_declared_tensor_view_is_rejected(actual_protocols, tmp_path):
    before, after, _ = actual_protocols["branch_switch"]
    view = tensor((4, 64), name="alias_view")
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [after["main"].params[0].data], ty_args=[view.ty]))
    bad = _append_root(after, declaration)
    _reject(before, bad, "branch_switch", tmp_path, "Unmodeled call or alias view")


@pytest.mark.parametrize(
    "key, field, value, message",
    [("tl.schedule_unit", "stage", 1, "Unmodeled stage"), ("tl.ascend_task", "core_mask", 2, "Unmodeled core")],
    ids=["stage-one", "cube-core"],
)
def test_unmodeled_stage_or_core_is_rejected(actual_protocols, tmp_path, key, field, value, message):
    before, after, _ = actual_protocols["branch_switch"]
    changed = False

    def change_metadata(region):
        nonlocal changed
        fields = attribute_fields(region)
        if not changed and fields["key"] == key:
            changed = True
            node = {**fields["node"], field: ir.IntImm("int64", value)}
            return attribute(node, key, fields["value"], region.body, loc=region.loc)
        return region

    bad = tvm_ffi.structural_map(after, [(ir.RegionStmt, change_metadata)])
    assert changed
    _reject(before, bad, "branch_switch", tmp_path, message)


@pytest.mark.parametrize(
    "field, value, message",
    [("stage", 1, "Unmodeled binding stage"), ("core_mask", 1, "Unmodeled binding core")],
    ids=["allocation-stage-one", "allocation-on-one-aiv"],
)
def test_unmodeled_allocation_binding_stage_or_core_is_rejected(actual_protocols, tmp_path, field, value, message):
    before, after, _ = actual_protocols["branch_switch"]
    changed = False

    def change_binding(binding):
        nonlocal changed
        fields = scheduled_fields(binding)
        if not changed and fields is not None and isinstance(binding.var.ty, tirx.TensorType):
            changed = True
            task = dict(fields["task"])
            if field == "core_mask":
                task[field] = value
            return scheduled_bind(ir.Bind(binding.var, binding.value.args[0]), value if field == "stage" else fields["stage"], task)
        return binding

    bad = tvm_ffi.structural_map(after, [(ir.Bind, change_binding)])
    assert changed
    _reject(before, bad, "branch_switch", tmp_path, message)


@pytest.mark.parametrize("unreachable", [False, True], ids=["executed-branch", "unreachable-branch"])
def test_runtime_memory_predicate_is_rejected_before_path_selection(actual_protocols, tmp_path, unreachable):
    before, after, _ = actual_protocols["branch_switch"]
    condition = ir.TensorLoad(after["main"].params[0], [0, 0]) > 0
    changed = False

    def change_guard(branch):
        nonlocal changed
        if not changed:
            changed = True
            return ir.If(condition, branch.then_case, branch.else_case, loc=branch.loc)
        return branch

    if unreachable:
        hidden = ir.If(condition, [ir.Evaluate(ir.const(0))], None)
        bad = _append_root(after, ir.If(ir.const(False, "bool"), [hidden], None))
        assert memory_signature(extract(bad)) == memory_signature(extract(after))
    else:
        bad = tvm_ffi.structural_map(after, [(ir.If, change_guard)])
        assert changed
    _reject(before, bad, "branch_switch", tmp_path, "Unmodeled runtime memory predicate")


@pytest.mark.parametrize(
    "operation, fault",
    [("set", "remove"), ("wait", "remove"), ("set", "wrong-slot"), ("wait", "wrong-bank")],
    ids=["lost-release", "lost-acquire", "wrong-release-slot", "wrong-cube-acquire-bank"],
)
def test_broken_actual_mode4_bridge_cannot_order_the_aivs(actual_protocols, tmp_path, operation, fault):
    before, after, _ = actual_protocols["cube_bridge"]
    changed = False

    def break_bridge(statement):
        nonlocal changed
        call = statement.value
        if changed or not isinstance(call, ir.Call) or call.op.name != f"tl.ascend_cross_core_{operation}_flag":
            return statement
        changed = True
        assert int(call.args[0]) == 4
        if fault == "remove":
            return ir.SeqStmt([])
        arguments = list(call.args)
        if fault == "wrong-bank":
            # The first acquire belongs to the Cube; moving by 16 selects the
            # other AIV bank while retaining a valid mode4 token number.
            arguments[2] = (arguments[2] + 16) % 32
        else:
            arguments[2] = (arguments[2] + 1) % 16
        return ir.Evaluate(ir.Call(call.op, arguments, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc, ty=call.ty), loc=statement.loc)

    bad = tvm_ffi.structural_map(after, [(ir.Evaluate, break_bridge)])
    assert changed
    bad_events = extract(bad)
    assert memory_signature(extract(after)) == memory_signature(bad_events)
    report = prove(bad_events, require_drain=False)
    assert not report["safe"] and report["unordered"]
    (tmp_path / "broken-bridge-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    _reject(before, bad, "cube_bridge", tmp_path, "token_errors")
