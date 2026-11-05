"""Indirect vector addresses need immutable storage, not a global Pure annotation."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import tilelang
from tilelang.language._native import encode_annotations
from tilelang.testing.ir import collect_nodes
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


CUDA = Target({"kind": "cuda", "arch": "sm_90"})


def tensor(name, dtype, *, alignment=64):
    return ir.Var(name, tirx.TensorType((128,), dtype, layout="default", data_alignment=alignment))


def attribute(node, key, value, body):
    schema, payload = encode_annotations({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def make_function(*, case="readonly", alignment=64, start=0, stride=1, bound=False):
    source, destination, indices = tensor("A", "float32"), tensor("B", "float32", alignment=alignment), tensor("indices", "int64")
    lane = ir.Var("lane", "int32")
    prefix, body = [], []
    row = indices[0]
    attrs = {"target": CUDA, "tirx.noalias": case != "may_alias"}
    if case == "non_restrict":
        attrs["tl.non_restrict_params"] = [indices]
    if case == "alias_indices":
        attrs["tl.may_alias_param_indices"] = [2]
    if case == "written_index":
        body.append(ir.TensorStore(indices, [ir.const(0)], ir.const(1, "int64")))
    if case == "destination_read":
        row = ir.prim.Cast("int64", destination[0])
    if case == "destination_view":
        alias = ir.Var("alias", tirx.TensorType((128,), "float32", layout="default", data=destination.data, data_alignment=alignment))
        prefix.append(ir.Bind(alias, ir.Call("tirx.decl_tensor", [destination.data], ty_args=[alias.ty])))
        row = ir.prim.Cast("int64", alias[0])
    if case in ("opaque", "opaque_pointer"):
        args = [destination.data] if case == "opaque_pointer" else []
        body.append(ir.Evaluate(tirx.call_extern("may_modify_indices", *args, ty="int32")))
    if case == "effectful_index":
        row = tirx.call_extern("next_index", ty="int64")
    if case in ("assume", "assume_readonly", "assume_opaque"):
        condition = ir.const(True)
        if case == "assume_readonly":
            condition = indices[0] >= ir.const(0, "int64")
        if case == "assume_opaque":
            condition = tirx.call_extern("may_update_indices", ty="bool")
        prefix.append(ir.Evaluate(ir.Call("prim.assume", [condition])))
    if bound:
        index_value = ir.Var("row", "int64")
        body.append(ir.Bind(index_value, row))
        row = index_value
        if bound == "chain":
            next_value = ir.Var("same_row", "int64")
            body.append(ir.Bind(next_value, row + ir.const(0, "int64")))
            row = next_value
    offset = row * ir.const(4, "int64") + ir.prim.Cast("int64", lane * stride + start)
    pointer = ir.Call("tl.access_ptr", [destination[offset], ir.const(1), ir.const(3)])
    body.append(ir.Evaluate(ir.Call("tl.atomic_add_elem_op", [pointer, source[lane]])))
    loop = ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, body)
    if case == "volatile_scope":
        loop = attribute(indices.data, "volatile_scope", 1, loop)
    if case == "volatile_allocation":
        volatile = tensor("volatile", "int32")
        prefix.append(ir.Bind(volatile, ir.Call("tirx.alloc_tensor", [], ty_args=[volatile.ty], attrs={"tirx.volatile": True})))
    return tirx.Function([source, destination, indices], prefix + [loop]).with_attrs(attrs)


def run(function, *, owning=True, legalize=False):
    snapshot = ir.save_json(function)
    module = ir.IRModule({"main": function})
    if legalize:
        module = tilelang.transform.LegalizeVectorizedLoop()(module)
    compiler_pass = tilelang.transform.VectorizeLoop() if owning else tirx.transform.VectorizeLoop()
    result = compiler_pass(module)["main"]
    assert snapshot == ir.save_json(function)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    for call in collect_nodes(result, ir.Call):
        call.validate()
    return result


def packed(function):
    return [call for call in collect_nodes(function, ir.Call) if call.op.name == "tl.atomic_addx4_elem_op"]


@pytest.mark.parametrize("legalize", [False, True])
@pytest.mark.parametrize("case", ["readonly", "assume", "assume_readonly"])
def test_readonly_indirect_address_keeps_four_lanes(legalize, case):
    function = make_function(case=case)
    result = run(function, legalize=legalize)
    assert len(packed(result)) == 1
    assert not collect_nodes(result, ir.For)
    pointer = packed(result)[0].args[0]
    loads = collect_nodes(pointer, ir.TensorLoad)
    assert any(load.source.name == "indices" for load in loads)
    assert any(load.source.name == "B" and load.indices[0].ty == ir.PrimType("int64") for load in loads)
    # The operation remains an ordinary read-state TensorLoad. No effect
    # annotation or global registry contract was rewritten by the owner.
    assert not packed(run(function, owning=False))


@pytest.mark.parametrize(
    "case",
    [
        "may_alias",
        "non_restrict",
        "alias_indices",
        "written_index",
        "destination_read",
        "destination_view",
        "opaque",
        "opaque_pointer",
        "effectful_index",
        "volatile_scope",
        "volatile_allocation",
        "assume_opaque",
    ],
)
@pytest.mark.parametrize("legalize", [False, True])
def test_writes_effects_and_volatile_do_not_acquire_readonly_permission(case, legalize):
    result = run(make_function(case=case), legalize=legalize)
    assert not packed(result)
    assert collect_nodes(result, ir.For)
    assert any(call.op.name == "tl.atomic_add_elem_op" for call in collect_nodes(result, ir.Call))


@pytest.mark.parametrize("alignment,start,stride", [(1, 0, 1), (4, 0, 1), (64, 1, 1), (64, 0, 0), (64, 0, 2), (64, 12, -1)])
def test_readonly_does_not_waive_alignment_or_unique_contiguous_destination(alignment, start, stride):
    result = run(make_function(alignment=alignment, start=start, stride=stride))
    assert not packed(result)
    assert collect_nodes(result, ir.For)


def test_local_bound_readonly_index_has_no_stale_definition():
    result = run(make_function(bound=True), legalize=True)
    assert len(packed(result)) == 1
    assert not tirx.analysis.undefined_vars(result)


@pytest.mark.parametrize("case", ["volatile_scope", "written_index", "may_alias"])
@pytest.mark.parametrize("legalize", [False, True])
@pytest.mark.parametrize("binding", [True, "chain"])
def test_local_bound_index_keeps_its_read_state_contract(case, legalize, binding):
    result = run(make_function(case=case, bound=binding), legalize=legalize)
    assert not packed(result)
    assert collect_nodes(result, ir.For)


@pytest.mark.parametrize("reverse", [False, True])
def test_each_function_has_its_own_alias_facts(reverse):
    functions = [("safe", make_function()), ("unsafe", make_function(case="non_restrict"))]
    module = ir.IRModule(dict(reversed(functions) if reverse else functions))
    result = tilelang.transform.VectorizeLoop()(module)
    assert packed(result["safe"])
    assert not packed(result["unsafe"])
    assert not packed(run(make_function(), owning=False))


def fail_inside_vectorizer():
    # Unsupported extents may legally fall back to scalar code. Inject the
    # exception after TileLang installs its memory scope and enters the native
    # vectorizer, so unwinding is tested independently of that admission policy.
    @ir.transform.pass_instrument
    class Fail:
        def run_before_pass(self, module, info):
            if info.name == "tirx.VectorizeLoop":
                raise RuntimeError("injected vectorization failure")

    with ir.transform.PassContext(instruments=[Fail()]):
        run(make_function())


@pytest.mark.parametrize("raise_inside", [False, True])
def test_nested_pass_restores_outer_scope_after_return_or_exception(raise_inside):
    @ir.transform.pass_instrument
    class Nested:
        entered = False
        called = False

        def run_before_pass(self, module, info):
            if info.name != "tirx.VectorizeLoop" or self.entered:
                return
            self.entered = self.called = True
            try:
                if raise_inside:
                    with pytest.raises(RuntimeError, match="injected vectorization failure"):
                        fail_inside_vectorizer()
                else:
                    assert not packed(run(make_function(case="non_restrict")))
            finally:
                self.entered = False

    instrument = Nested()
    with ir.transform.PassContext(instruments=[instrument]):
        assert packed(run(make_function()))
    assert instrument.called
    assert not packed(run(make_function(), owning=False))


def test_exception_does_not_leave_a_memory_scope_installed():
    with pytest.raises(RuntimeError, match="injected vectorization failure"):
        fail_inside_vectorizer()
    assert not packed(run(make_function(), owning=False))
    assert packed(run(make_function()))


def test_concurrent_functions_have_independent_thread_local_scopes():
    barrier = threading.Barrier(2)

    def compile_one(case):
        @ir.transform.pass_instrument
        class Interleave:
            def run_before_pass(self, module, info):
                if info.name == "tirx.VectorizeLoop":
                    barrier.wait(timeout=15)

        with ir.transform.PassContext(instruments=[Interleave()]):
            result = bool(packed(run(make_function(case=case))))
        assert not packed(run(make_function(), owning=False))
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(compile_one, ["readonly", "non_restrict"])) == [True, False]
