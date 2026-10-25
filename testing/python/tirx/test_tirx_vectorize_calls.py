import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import VectorizeLoop

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def collect(body, kind):
    nodes = []
    stmt_functor.post_order_visit(body, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def emit(loop, tensors, enable=True):
    function = tirx.Function([tensor.tensor for tensor in tensors], loop, ret_type=ir.TupleType([]))
    target = Target({"kind": "cuda", "arch": "sm_90"})
    function = function.with_attr("target", target)
    with target:
        lowered = VectorizeLoop(enable)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(lowered)
    ir.assert_structural_equal(lowered, ir.load_json(ir.save_json(lowered)))
    return lowered


@pytest.mark.parametrize(
    "dtype,scope,width,expected",
    [("float32", "global", 4, 4), ("float32", "shared", 4, 1), ("float16", "shared", 2, 2), ("float16", "global", 4, 1)],
)
@pytest.mark.parametrize("lower_pointer", [False, True])
def test_native_atomic_specialization(dtype, scope, width, expected, lower_pointer):
    destination = TensorView.create("destination", [16], dtype, scope=scope)
    index = ir.Var("index", "int32")
    pointer = native("MakeAccessPtrFromBufferLoad", destination.load(index), 3)
    if lower_pointer:
        pointer = native("LowerAccessPtrCall", pointer)
    call = ir.Call("tl.atomic_add_elem_op", [pointer, ir.const(2.0, dtype), 0], ty=dtype)
    call.validate()
    loop = ir.For(index, 0, width, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call))
    lowered = emit(loop, [destination])
    atomic = [node for node in collect(lowered.body, ir.Call) if "atomic_add" in node.op.name]
    assert len(atomic) == 1
    expected_name = "tl.atomic_add_elem_op" if expected == 1 else f"tl.atomic_addx{expected}_elem_op"
    assert atomic[0].op.name == expected_name
    assert int(atomic[0].args[2]) == 0
    if expected > 1:
        address = atomic[0].args[0]
        assert address.ty == ir.PtrType(dtype, scope)
        assert int(address.attrs["tl.access_extent"]) == expected
        assert int(address.attrs["tl.access_mask"]) == 3
        assert atomic[0].args[1].ty == ir.PrimType(f"{dtype}x{expected}")
        assert not collect(lowered.body, ir.For)
    else:
        assert collect(lowered.body, ir.For)[0].kind == ir.ForKind.DEFAULT


@pytest.mark.parametrize("offset", [lambda index: index // 4, lambda index: index + 1, lambda index: index * 2])
def test_atomic_rejects_aliasing_or_misaligned_vector_destinations(offset):
    destination = TensorView.create("destination", [16], "float32")
    index = ir.Var("index", "int32")
    pointer = native("MakeAccessPtrFromBufferLoad", destination.load(offset(index)), 3)
    call = ir.Call("tl.atomic_add_elem_op", [pointer, ir.const(1.0, "float32"), 0], ty="float32")
    lowered = emit(ir.For(index, 0, 4, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call)), [destination])
    assert collect(lowered.body, ir.For)[0].kind == ir.ForKind.DEFAULT


@pytest.mark.parametrize(
    "dtype,count,width,expected",
    [("float32", 1, 4, 4), ("float16", 1, 8, 8), ("float16", 2, 4, 8), ("int4", 1, 32, 32), ("float32", 2, 4, None)],
)
def test_cp_async_widens_logical_counts_and_pointer_metadata(dtype, count, width, expected):
    source = TensorView.create("source", [32], dtype)
    destination = TensorView.create("destination", [32], dtype, scope="shared")
    index = ir.Var("index", "int32")
    source_pointer = ir.Call("tl.access_ptr", [source.load(index * count), count, 1])
    destination_pointer = ir.Call("tl.access_ptr", [destination.load(index * count), count, 2])
    call = ir.Call("tl.ptx_cp_async", [destination_pointer, source_pointer, count], ty="int32")
    lowered = emit(ir.For(index, 0, width, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call)), [source, destination])
    async_copy = [node for node in collect(lowered.body, ir.Call) if node.op.name == "tl.ptx_cp_async"][0]
    if expected is None:
        assert collect(lowered.body, ir.For)[0].kind == ir.ForKind.DEFAULT
    else:
        assert not collect(lowered.body, ir.For)
        assert int(async_copy.args[2]) == expected
        assert int(async_copy.args[0].attrs["tl.access_extent"]) == expected
        assert int(async_copy.args[1].attrs["tl.access_mask"]) == 1
        assert int(async_copy.args[0].attrs["tl.access_mask"]) == 2


def test_vectorization_can_be_disabled_without_dropping_iterations():
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.VECTORIZED, output.store(index, index))
    lowered = emit(loop, [output], enable=False)
    assert collect(lowered.body, ir.For)[0].kind == ir.ForKind.DEFAULT


def test_multidimensional_atomic_and_nondefault_order_are_preserved():
    destination = TensorView.create("destination", [4, 2], "float32")
    index = ir.Var("index", "int32")
    pointer = native("MakeAccessPtrFromBufferLoad", destination.load([index // 2, index % 2]), 3)
    call = ir.Call("tl.atomic_add_elem_op", [pointer, ir.const(1.0, "float32"), 2], ty="float32")
    lowered = emit(ir.For(index, 0, 4, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call)), [destination])
    atomic = [node for node in collect(lowered.body, ir.Call) if "atomic_add" in node.op.name][0]
    assert atomic.op.name == "tl.atomic_addx4_elem_op"
    assert int(atomic.args[2]) == 2
    assert atomic.args[0].attrs["tl.access_region"].source.same_as(destination.tensor)
    assert [int(axis) for axis in atomic.args[0].args[0].indices] == [0, 0]


@pytest.mark.parametrize("varying", [False, True])
def test_cp_async_predicate_must_be_uniform(varying):
    source = TensorView.create("source", [16], "float32")
    destination = TensorView.create("destination", [16], "float32", scope="shared")
    index = ir.Var("index", "int32")
    source_pointer = native("MakeAccessPtrFromBufferLoad", source.load(index), 1)
    destination_pointer = native("MakeAccessPtrFromBufferLoad", destination.load(index), 2)
    source_pointer = native("LowerAccessPtrCall", source_pointer)
    destination_pointer = native("LowerAccessPtrCall", destination_pointer)
    predicate = index % 2 == 0 if varying else ir.const(True)
    call = ir.Call("tl.ptx_cp_async", [destination_pointer, source_pointer, 1, predicate], ty="int32")
    lowered = emit(ir.For(index, 0, 4, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call)), [source, destination])
    loops = collect(lowered.body, ir.For)
    assert bool(loops) == varying
    copy = [node for node in collect(lowered.body, ir.Call) if node.op.name == "tl.ptx_cp_async"][0]
    assert int(copy.args[2]) == (1 if varying else 4)


def test_planning_substitution_rebinds_pointer_region_metadata():
    from tvm_tirx import sym

    destination = TensorView.create("destination", [16], "float32")
    index = ir.Var("index", "int32")
    pointer = native("MakeAccessPtrFromBufferLoad", destination.load(index), 3)
    pointer = native("LowerAccessPtrCall", pointer)
    call = ir.Call("tl.atomic_add_elem_op", [pointer, ir.const(1.0, "float32"), 0], ty="float32")
    loop = ir.For(index, 0, 16, 1, ir.ForKind.PARALLEL, ir.Evaluate(call))
    rewritten = native("VectorizeLoop", loop, sym.Analyzer(), {}, 4)
    function = tirx.Function([destination.tensor], rewritten, ret_type=ir.TupleType([]))
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    assert tirx.analysis.check_well_formed(function)
