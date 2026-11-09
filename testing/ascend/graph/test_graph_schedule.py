"""Scalar graph TIR must acquire Ascend ownership and preserve VF boundaries."""

from types import SimpleNamespace

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend.analysis import VFChecker
from tilelang.ascend.graph.elementwise import ElementwiseScheduleError, legalize_scalar_arithmetic, schedule_elementwise
from tilelang.engine import lower
from tvm import te, tirx


_BACKENDS = [pytest.param("ascend", "tvm_ffi", id="ascendc"), pytest.param("pto", "pto", id="pto", marks=pytest.mark.pto)]
_TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


def _compile(program, target, execution_backend):
    kernel = tilelang.compile(program, target=target, execution_backend=execution_backend, out_idx=-1)
    assert kernel.execution_backend == execution_backend
    if execution_backend == "pto":
        from tilelang.jit.adapter.pto import PTOKernelAdapter

        assert isinstance(kernel.adapter, PTOKernelAdapter)
    else:
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        assert isinstance(kernel.adapter, TVMFFIKernelAdapter)
    return kernel


def _injective(shape, dtype="float32", *, broadcast=False, cast=False):
    x = te.placeholder(shape, dtype, name="x")
    y = te.placeholder((shape[-1],) if broadcast else shape, dtype, name="y")
    added = te.compute(shape, lambda *indices: x[indices] + y[indices[-1] if broadcast else indices], name="added")
    result_dtype = "float32" if cast else dtype
    result = te.compute(
        shape,
        lambda *indices: tirx.Cast(result_dtype, te.max(added[indices], tirx.const(0, dtype))),
        name="result",
    )
    return te.create_prim_func([x, y, result]).with_attr("global_symbol", "main")


def _collect(body, cls):
    nodes = []
    tirx.stmt_functor.post_order_visit(body, lambda node: nodes.append(node) if isinstance(node, cls) else None)
    return nodes


def test_injective_producer_chain_has_no_gm_intermediate():
    original = _injective((7, 65), broadcast=True)
    scheduled = schedule_elementwise(original, SimpleNamespace(tile_size=192, threads=64, num_cores=2))

    module = tvm.IRModule.from_expr(scheduled)
    VFChecker()(module)
    vfs = [block for block in _collect(scheduled.body, tirx.SBlock) if block.name_hint == "SIMT_VF"]
    assert len(vfs) == 1
    stores = _collect(scheduled.body, tirx.BufferStore)
    assert len(stores) == 1
    assert stores[0].buffer.same_as(scheduled.buffer_map[scheduled.params[-1]])
    input_buffers = {scheduled.buffer_map[param] for param in scheduled.params[:-1]}
    assert all(load.buffer in input_buffers for load in _collect(scheduled.body, tirx.BufferLoad))
    assert not any(block.alloc_buffers for block in _collect(scheduled.body, tirx.SBlock))
    thread_tags = [node.node.thread_tag for node in _collect(scheduled.body, tirx.AttrStmt) if node.attr_key == "thread_extent"]
    assert thread_tags.count("blockIdx.x") == 1
    assert thread_tags.count("threadIdx.x") == 1


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("target", _TARGETS)
def test_injective_schedule_source_codegen(dtype, target):
    scheduled = schedule_elementwise(_injective((7, 65), dtype, broadcast=True, cast=True))
    artifact = lower(scheduled, target=target)

    if target == "pto":
        assert "with pto.simt(" in artifact.kernel_source
        assert 'kernel_kind="vector"' in artifact.kernel_source
    else:
        assert "__simt_vf__" in artifact.kernel_source
        assert "__vector__" in artifact.kernel_source
        assert "asc_vf_call" in artifact.kernel_source


def test_injective_schedule_rejects_out_of_bounds_read():
    x = te.placeholder((8,), "float32", name="x")
    output = te.compute((8,), lambda i: x[i + 1], name="output")
    function = te.create_prim_func([x, output])

    with pytest.raises(ElementwiseScheduleError, match="input read"):
        schedule_elementwise(function)


def test_injective_schedule_rejects_input_mutation():
    x = te.placeholder((8,), "float32", name="x")
    intermediate = te.compute((8,), lambda i: x[i] + 1, name="intermediate")
    output = te.compute((8,), lambda i: intermediate[i] * 2, name="output")
    function = te.create_prim_func([x, intermediate, output])

    with pytest.raises(ElementwiseScheduleError, match="Input mutation"):
        schedule_elementwise(function)


def test_injective_schedule_rejects_partial_output_domain():
    function = _injective((65,))

    def shorten(node):
        if isinstance(node, tirx.For):
            return tirx.For(node.loop_var, node.min, 32, node.kind, node.body)
        return None

    partial = function.with_body(tirx.stmt_functor.ir_transform(function.body, None, shorten))
    with pytest.raises(ElementwiseScheduleError, match="part of its spatial domain"):
        schedule_elementwise(partial)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_injective_schedule_preserves_low_precision_operation_boundaries(dtype):
    x = te.placeholder((65,), dtype, name="x")
    y = te.placeholder((65,), dtype, name="y")
    summed = te.compute((65,), lambda i: x[i] + y[i], name="summed")
    output = te.compute((65,), lambda i: summed[i] - x[i], name="output")
    scheduled = schedule_elementwise(te.create_prim_func([x, y, output]).with_attr("global_symbol", "main"))
    store = _collect(scheduled.body, tirx.BufferStore)[0]

    assert isinstance(store.value, tirx.Cast)
    assert str(store.value.dtype) == dtype
    assert isinstance(store.value.value, tirx.Sub)
    assert str(store.value.value.dtype) == "float32"
    assert any(str(cast.dtype) == dtype and isinstance(cast.value, tirx.Add) for cast in _collect(store.value, tirx.Cast))


def _unary_intrinsic(dtype, operation):
    source = te.placeholder((65,), dtype, name="source")
    output = te.compute((65,), lambda i: getattr(tirx, operation)(source[i]), name="output")
    return te.create_prim_func([source, output]).with_attr("global_symbol", "main")


@pytest.mark.parametrize("operation", ["exp", "sqrt", "rsqrt"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("target", _TARGETS)
def test_injective_schedule_unary_intrinsic_codegen(dtype, operation, target):
    scheduled = schedule_elementwise(_unary_intrinsic(dtype, operation))
    artifact = lower(scheduled, target=target)
    assert ("with pto.simt(" if target == "pto" else "__simt_vf__") in artifact.kernel_source
    calls = _collect(scheduled.body, tirx.Call)
    arithmetic = [call for call in calls if call.op.same_as(tvm.ir.Op.get(f"tirx.{operation}"))]
    assert len(arithmetic) == 1
    assert str(arithmetic[0].dtype) == "float32"
    if dtype != "float32":
        store = _collect(scheduled.body, tirx.BufferStore)[0]
        assert isinstance(store.value, tirx.Cast)
        assert str(store.value.dtype) == dtype


def test_injective_schedule_rejects_non_whitelisted_call():
    with pytest.raises(ElementwiseScheduleError, match="Only unary floating"):
        schedule_elementwise(_unary_intrinsic("float32", "log"))


def test_scalar_legalization_preserves_other_call_arguments_and_annotations():
    condition = tirx.Var("condition", "bool")
    value = tirx.Var("value", "bfloat16")
    expression = tirx.Call(
        "bfloat16",
        tvm.ir.Op.get("tirx.if_then_else"),
        [condition, value + value, value],
        annotations={"graph_test": tirx.const(1, "int32")},
    )
    legalized = legalize_scalar_arithmetic(expression)

    assert isinstance(legalized, tirx.Call)
    assert legalized.op.same_as(expression.op)
    assert legalized.args[0].same_as(condition)
    assert int(legalized.annotations["graph_test"]) == 1
    assert isinstance(legalized.args[1], tirx.Cast)
    assert str(legalized.args[1].dtype) == "bfloat16"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("target", _TARGETS)
def test_injective_schedule_vectorizes_aligned_gm_transfers(dtype, target):
    scheduled = schedule_elementwise(_injective((64, 128), dtype, broadcast=True))
    assert int(scheduled.attrs["tl.graph_vector_width"]) == 4
    artifact = lower(scheduled, target=target)

    # The current Ascend target caps transfers at 64 bits: chunks of four
    # float32 values become two float2 transfers. The scalar NaN-preserving
    # expression stays between those loads and stores in thread-local arrays.
    if target == "pto":
        assert "pto.alloc_buffer((4,)" in artifact.kernel_source
        assert f"contiguous={2 if dtype == 'float32' else 4}" in artifact.kernel_source
    else:
        vector_type = "float2" if dtype == "float32" else "uint2"
        assert f"__gm__ {vector_type}*" in artifact.kernel_source
    assert "output_chunk" in artifact.kernel_source


def test_injective_schedule_keeps_non_aligned_rows_scalar():
    scheduled = schedule_elementwise(_injective((7, 65), broadcast=True))
    assert int(scheduled.attrs["tl.graph_vector_width"]) == 1
    forced_scalar = schedule_elementwise(_injective((64, 128)), SimpleNamespace(vector_width=1))
    assert int(forced_scalar.attrs["tl.graph_vector_width"]) == 1


@pytest.mark.parametrize(
    "shape,config",
    [
        (((1 << 31) * 1024,), SimpleNamespace(tile_size=1024)),
        ((128,), SimpleNamespace(tile_size=(1 << 31) - 1)),
    ],
)
def test_injective_schedule_rejects_index_overflow_without_allocating(shape, config):
    source = te.placeholder(shape, "float32", name="source")
    output = te.compute(shape, lambda *indices: source[indices] + 1, name="output")
    function = te.create_prim_func([source, output])

    with pytest.raises(ElementwiseScheduleError, match="int32 indexing range"):
        schedule_elementwise(function, config)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize(
    "shape,dtype,broadcast,cast",
    [
        ((1,), "float32", False, False),
        ((65,), "float16", False, True),
        ((7, 65), "float32", True, False),
        ((65536,), "bfloat16", False, False),
    ],
)
def test_injective_schedule_runtime(shape, dtype, broadcast, cast, target, execution_backend):
    scheduled = schedule_elementwise(_injective(shape, dtype, broadcast=broadcast, cast=cast))
    kernel = _compile(scheduled, target, execution_backend)
    x = torch.randn(shape, device="npu", dtype=getattr(torch, dtype))
    y = torch.randn((shape[-1],) if broadcast else shape, device="npu", dtype=getattr(torch, dtype))
    expected = torch.relu(x + y)
    if cast:
        expected = expected.float()
    actual = kernel(x, y)

    torch.testing.assert_close(actual, expected)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_injective_schedule_runtime_nan_and_infinity(dtype, target, execution_backend):
    kernel = _compile(schedule_elementwise(_injective((7,), dtype)), target, execution_backend)
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0, -0.0, 0.0, 1.0], device="npu", dtype=getattr(torch, dtype))
    y = torch.zeros_like(x)
    actual = kernel(x, y)
    expected = torch.relu(x + y)

    torch.testing.assert_close(actual, expected, equal_nan=True)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_injective_schedule_runtime_low_precision_rounding(dtype, target, execution_backend):
    x = te.placeholder((65,), dtype, name="x")
    y = te.placeholder((65,), dtype, name="y")
    summed = te.compute((65,), lambda i: x[i] + y[i], name="summed")
    output = te.compute((65,), lambda i: summed[i] - x[i], name="output")
    scheduled = schedule_elementwise(te.create_prim_func([x, y, output]).with_attr("global_symbol", "main"))
    kernel = _compile(scheduled, target, execution_backend)
    x_value = torch.ones((65,), device="npu", dtype=getattr(torch, dtype))
    y_value = torch.full_like(x_value, torch.finfo(getattr(torch, dtype)).eps / 2)

    actual = kernel(x_value, y_value)
    expected = (x_value + y_value) - x_value
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("operation", ["exp", "sqrt", "rsqrt"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_injective_schedule_runtime_unary_intrinsic(dtype, operation, target, execution_backend):
    scheduled = schedule_elementwise(_unary_intrinsic(dtype, operation))
    kernel = _compile(scheduled, target, execution_backend)
    value = torch.linspace(0.25, 2, 65, dtype=getattr(torch, dtype), device="npu")
    value[:6] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), -1.0, float("nan")], dtype=value.dtype, device=value.device)

    actual = kernel(value)
    expected = getattr(torch, operation)(value)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual)[zeros], torch.signbit(expected)[zeros])
