"""Scheduled vector counts are consumed only from the owning input carriers."""

import pytest
import tvm_ffi
from tilelang.ascend import transform
from tvm_tirx import ir, tirx

from testing.ascend._ir import (
    annotation,
    annotations,
    attribute,
    attribute_fields,
    attributes,
    kernel,
    launch_thread,
    nodes,
    sblock,
    sblock_annotations,
    sblock_metadata,
    sblocks,
    statements,
    tensor,
)
from testing.ascend.auto_schedule._scheduled_ir import unit


def tasks(output):
    return [
        unit(ir.TensorStore(output, [0], ir.IntImm("int32", 31)), core=2),
        unit(ir.TensorStore(output, [1], ir.IntImm("int32", 47)), core=1),
    ]


def replace_body(module, body):
    return ir.IRModule({"main": module["main"].with_body(body)})


def anchor(body, name="block"):
    return launch_thread(ir.Var(name, "int32"), "blockIdx.x", 1, body)


def lower(before):
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    after = transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    return after


def finish(after):
    saved = ir.save_json(after)
    result = transform.LowerAscendAnnotations()(after)
    result = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(result)
    result = tvm_ffi.get_global_func("tl.transform.VerifyLoweredAnnotations")()(result)
    assert ir.save_json(after) == saved
    assert tirx.analysis.check_well_formed(result)
    return result


def generated_count(module, count):
    (vector,) = sblocks(module, "VECTOR")
    assert vector.op.name == "tl.ascend_block" and not vector.args
    assert set(vector.attrs.keys()) == {"name", "annotations"}
    assert int(sblock_metadata(vector)["annotations"]["vector_count"]) == count
    launches = [
        node for node in nodes(vector, ir.RegionStmt) if node.op.name == "tirx.launch_thread" and str(node.args[0].value) == "cthread"
    ]
    assert len(launches) == 1 and int(launches[0].args[1]) == count
    (cube,) = sblocks(module, "CUBE")
    assert cube.op.name == "tl.ascend_block" and not cube.args
    assert not any(node.op.name == "tirx.launch_thread" and str(node.args[0].value) == "cthread" for node in nodes(cube, ir.RegionStmt))
    assert [int(store.value) for store in nodes(cube, ir.TensorStore)] == [31]
    assert [int(store.value) for store in nodes(vector, ir.TensorStore)] == [47]


def only_generated_counts(module):
    for block in sblocks(module):
        if sblock_metadata(block)["name"] != "VECTOR":
            assert "vector_count" not in sblock_metadata(block).get("annotations", {})
    for carrier in attributes(module, "tl.kernel"):
        assert "vector_count" not in attribute_fields(carrier)["value"]
    for call in nodes(module, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
            assert "vector_count" not in fields


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("carrier", ["root-fields", "prefix", "middle", "suffix"])
def test_root_vector_count_is_consumed_independently_of_lexical_position(count, carrier):
    output = tensor((2,), "int32", name="output")
    body = tasks(output)
    fields = {"vector_count": count} if carrier == "root-fields" else {}
    if carrier != "root-fields":
        position = {"prefix": 0, "middle": 1, "suffix": 2}[carrier]
        body.insert(position, annotation({"vector_count": ir.IntImm("int64", count), "pragma_count_owner": 19}))
    before = kernel(body, params=[output], annotations=fields)
    after = lower(before)
    generated_count(after, count)
    only_generated_counts(after)
    if carrier != "root-fields":
        retained = [int(fields["pragma_count_owner"]) for fields in annotations(after) if "pragma_count_owner" in fields]
        assert retained == [19, 19]
        for name in ("CUBE", "VECTOR"):
            (stream,) = sblocks(after, name)
            events = []
            for statement, _ in statements(stream):
                if isinstance(statement, ir.TensorStore):
                    events.append("store")
                elif (
                    isinstance(statement, ir.Evaluate)
                    and isinstance(statement.value, ir.Call)
                    and isinstance(statement.value.op, ir.Op)
                    and statement.value.op.name == "tl.annotation"
                ):
                    events.append("hint")
            hint_first = carrier == "prefix" or carrier == "middle" and name == "VECTOR"
            assert events == (["hint", "store"] if hint_first else ["store", "hint"])
    finalized = finish(after)
    generated_count(finalized, count)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("root_present", [False, True])
def test_outer_kernel_count_is_consumed_after_lowering_its_root(count, root_present):
    output = tensor((2,), "int32", name="output")
    module = kernel(tasks(output), params=[output])
    (root,) = sblocks(module, "tilelang_root")
    carrier = attribute(None, "tl.kernel", {"vector_count": count, "pragma_count_owner": 23}, [root] if root_present else root.body)
    after = lower(replace_body(module, [anchor([carrier])]))
    generated_count(after, count)
    only_generated_counts(after)
    (carrier,) = attributes(after, "tl.kernel")
    assert int(attribute_fields(carrier)["value"]["pragma_count_owner"]) == 23
    generated_count(finish(after), count)


def test_sibling_kernel_counts_and_execution_scopes_stay_independent():
    output = tensor((2,), "int32", name="output")
    modules = [kernel(tasks(output), params=[output]) for _ in range(2)]
    launches = []
    for count, module in enumerate(modules, 1):
        (root,) = sblocks(module, "tilelang_root")
        launches.append(anchor([attribute(None, "tl.kernel", {"vector_count": count}, [root])], f"block{count}"))
    after = lower(replace_body(modules[0], launches))
    only_generated_counts(after)
    roots = sblocks(after, "tilelang_root")
    assert len(roots) == 2
    for count, root in enumerate(roots, 1):
        generated_count(root, count)
    finalized = finish(after)
    assert [int(sblock_annotations(block)["vector_count"]) for block in sblocks(finalized, "VECTOR")] == [1, 2]


@pytest.mark.parametrize("count", [1, 2])
def test_nested_kernel_input_carriers_are_consumed_in_the_same_launch_chain(count):
    output = tensor((2,), "int32", name="output")
    module = kernel(tasks(output), params=[output])
    (root,) = sblocks(module, "tilelang_root")
    inner = attribute(None, "tl.kernel", {"vector_count": count}, [root])
    outer = attribute(None, "tl.kernel", {"vector_count": 3 - count}, [inner])
    after = lower(replace_body(module, [anchor([outer])]))
    assert len(attributes(after, "tl.kernel")) == 2
    generated_count(after, count)
    only_generated_counts(after)
    generated_count(finish(after), count)


@pytest.mark.parametrize("foreign", ["nested-block", "nested-root", "nested-kernel", "branch", "loop"])
def test_nested_foreign_counts_are_preserved_for_their_own_consumer(foreign):
    output = tensor((2,), "int32", name="output")
    body = tasks(output)
    if foreign in ("nested-block", "nested-root", "nested-kernel"):
        body = []
        for index, core, value in ((0, 2, 31), (1, 1, 47)):
            store = ir.TensorStore(output, [index], ir.IntImm("int32", value))
            if foreign == "nested-kernel":
                nested = attribute(None, "tl.kernel", {"vector_count": 13}, [store])
            else:
                name = "tilelang_root" if foreign == "nested-root" else "other"
                nested = sblock(name, [store], annotations={"vector_count": 13})
            body.append(unit(nested, core=core))
    else:
        nested = [annotation({"vector_count": 13}), *body]
        if foreign == "branch":
            body = [ir.If(ir.const(True, "bool"), nested, None)]
        else:
            body = [unit(ir.For(ir.Var("i", "int32"), 0, 1, 1, ir.ForKind.DEFAULT, nested), core=None)]
    after = lower(kernel(body, params=[output], annotations={"vector_count": 1}))
    generated_count(after, 1)
    retained = [int(fields["vector_count"]) for fields in annotations(after) if "vector_count" in fields]
    retained.extend(
        int(attribute_fields(carrier)["value"]["vector_count"])
        for carrier in attributes(after, "tl.kernel")
        if "vector_count" in attribute_fields(carrier)["value"]
    )
    assert retained.count(13) == 2
    with pytest.raises(ValueError, match="vector_count"):
        finish(after)


@pytest.mark.parametrize("enclosure", ["without-anchor", "shadowed-by-inner-anchor"])
def test_unowned_outer_kernel_count_is_not_discarded(enclosure):
    output = tensor((2,), "int32", name="output")
    module = kernel(tasks(output), params=[output], annotations={"vector_count": 1})
    (root,) = sblocks(module, "tilelang_root")
    if enclosure == "shadowed-by-inner-anchor":
        body = [anchor([attribute(None, "tl.kernel", {"vector_count": 1}, [root])], "inner_block")]
    else:
        body = [root]
    foreign = attribute(None, "tl.kernel", {"vector_count": 13}, body)
    body = [anchor([foreign], "outer_block")] if enclosure == "shadowed-by-inner-anchor" else [foreign]
    after = lower(replace_body(module, body))
    generated_count(after, 1)
    values = [
        int(attribute_fields(carrier)["value"]["vector_count"])
        for carrier in attributes(after, "tl.kernel")
        if "vector_count" in attribute_fields(carrier)["value"]
    ]
    assert values == [13]
    with pytest.raises(ValueError, match="vector_count"):
        finish(after)


@pytest.mark.parametrize("count", [0, 3])
def test_invalid_owned_count_remains_rejected(count):
    output = tensor((2,), "int32", name="output")
    before = kernel(tasks(output), params=[output], annotations={"vector_count": count})
    saved = ir.save_json(before)
    with pytest.raises(ValueError, match=r"count == 1 \|\| count == 2"):
        transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved


def test_dynamic_owned_count_remains_rejected():
    output = tensor((2,), "int32", name="output")
    count = ir.Var("count", "int32")
    before = kernel(tasks(output), params=[output, count], annotations={"vector_count": count})
    saved = ir.save_json(before)
    with pytest.raises(ValueError, match="Mixed-kernel vector_count must be a constant scalar integer"):
        transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
