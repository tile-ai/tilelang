import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.runtime import MemoryScope

from tilelang._compat.tensor import TensorView
from tilelang.transform import AddWrapperForSingleBufStore


def make_fragment(shape=(2,)):
    return TensorView.create("fragment", shape, "int32", scope="local.fragment").tensor


def transform(body, params=(), *, loc=ir.UNKNOWN_LOC):
    function = tirx.Function(params, body, ret_type=ir.TupleType([]), loc=loc)
    return AddWrapperForSingleBufStore()(ir.IRModule({"main": function}))["main"]


def test_fragment_scope_has_distinct_native_identity():
    fragment = make_fragment()
    assert fragment.ty.scope.name == "local.fragment"
    assert fragment.ty.scope != MemoryScope("local")
    assert make_fragment().ty.scope == fragment.ty.scope


def test_wrap_fragment_store_preserves_location_and_identity():
    fragment = make_fragment()
    location = ir.SourceLoc(ir.SourceName("kernel.py"), 10, 0, 10, 15)
    store = ir.TensorStore(fragment, [0], 1, loc=location)
    result = transform(store, [fragment], loc=location)
    wrapper = result.body.seq[0]
    assert isinstance(wrapper, ir.For)
    assert wrapper.kind == ir.ForKind.PARALLEL
    assert int(wrapper.begin) == 0
    assert int(wrapper.end) == int(wrapper.step) == 1
    assert wrapper.body.seq[0].same_as(store)
    assert wrapper.loc.same_as(location)
    assert result.loc.same_as(location)


def test_wrap_store_reading_fragment():
    fragment = make_fragment()
    output = TensorView.create("output", [1], "int32").tensor
    store = ir.TensorStore(output, [0], ir.TensorLoad(fragment, [0]))
    result = transform(store, [fragment, output])
    assert isinstance(result.body.seq[0], ir.For)


@pytest.mark.parametrize("scope", ["global", "local", "shared"])
def test_do_not_wrap_other_scopes(scope):
    tensor = TensorView.create("tensor", [1], "int32", scope=scope).tensor
    store = ir.TensorStore(tensor, [0], 1)
    assert transform(store, [tensor]).body.seq[0].same_as(store)


@pytest.mark.parametrize("kind,annotations", [(ir.ForKind.PARALLEL, {}), (ir.ForKind.DEFAULT, {"num_stages": 2})])
def test_do_not_wrap_nested_tile_operation(kind, annotations):
    fragment = make_fragment()
    store = ir.TensorStore(fragment, [0], 1)
    loop = ir.For(ir.Var("index", "int32"), 0, 4, 1, kind, store, annotations)
    result = transform([loop, store], [fragment])
    assert tvm_ffi.structural_equal(result.body.seq[0], loop)
    assert isinstance(result.body.seq[1], ir.For)


def test_thread_region_and_derived_variables_remain_bound():
    fragment = make_fragment()
    thread = ir.Var("thread", "int32")
    alias = ir.Var("alias", "int32")
    store = ir.TensorStore(fragment, [0], alias)
    region = ir.RegionStmt(
        "tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, [ir.Bind(alias, thread + 1), store]
    )
    result = transform([region, ir.TensorStore(fragment, [0], 1)], [fragment])
    assert tvm_ffi.structural_equal(result.body.seq[0], region)
    assert isinstance(result.body.seq[1], ir.For)


def test_independent_fragment_access_within_thread_region_is_wrapped():
    fragment = make_fragment()
    thread = ir.Var("thread", "int32")
    store = ir.TensorStore(fragment, [0], 1)
    region = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, store)
    assert isinstance(transform(region, [fragment]).body.seq[0].body.seq[0], ir.For)


@pytest.mark.parametrize("bind", [False, True])
def test_native_thread_builtin_is_not_replicated(bind):
    fragment = make_fragment()
    thread_index = ir.Call("tirx.cuda.thread_idx", [ir.StringImm("x")])
    if bind:
        thread = ir.Var("thread", "int32")
        alias = ir.Var("alias", "int32")
        statements = [ir.Bind(thread, thread_index), ir.Bind(alias, thread + 1), ir.TensorStore(fragment, [0], alias)]
    else:
        statements = [ir.TensorStore(fragment, [0], thread_index)]
    assert tvm_ffi.structural_equal(transform(statements, [fragment]).body, ir.SeqStmt(statements))


def test_all_fragment_accesses_are_validated():
    fragment = make_fragment()
    store = ir.TensorStore(fragment, [0], ir.TensorLoad(fragment, [1]) + ir.TensorLoad(fragment, [0]))
    with pytest.raises(ValueError, match=r"Only fragment\[0\] access"):
        transform(store, [fragment])


def test_fragment_access_with_nonzero_store_index_is_rejected():
    fragment = make_fragment((2, 2))
    with pytest.raises(ValueError, match="non-zero index"):
        transform(ir.TensorStore(fragment, [0, 1], 1), [fragment])


def test_function_declaration_remains_a_declaration():
    assert transform(None).body is None
