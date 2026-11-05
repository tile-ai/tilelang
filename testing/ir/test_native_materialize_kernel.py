"""TileLang Kernel regions preserve launch coordinates through native lowering."""

import numpy as np
import pytest
import tvm_ffi

import tilelang  # noqa: F401
import tvm_tirx
from tvm_tirx import ir, tirx


def _integer(value, dtype="int64"):
    return ir.IntImm(dtype, value)


def _launch(end=7, *, begin=0, step=1, threads=None, body=None, metadata=None):
    ends = [end] if isinstance(end, ir.Expr) else [_integer(end)]
    params = [ir.Var("bx", "int64"), *(ir.Var(name, "int32") for name in ("tx", "ty", "tz"))]
    attrs = {
        "schema_version": 1,
        "threads_was_specified": threads is not None,
        "grid_tags": ["blockIdx.x"],
        "thread_tags": ["threadIdx.x", "threadIdx.y", "threadIdx.z"],
    }
    if metadata:
        attrs["hint_schema"] = ["map", *[[["str", key], ["int", value]] for key, value in metadata.items()]]
    args = [
        ir.Tuple([_integer(begin)]),
        ir.Tuple(ends),
        ir.Tuple([_integer(step)]),
        ir.Tuple([_integer(x, "int32") for x in threads] if threads else []),
        ir.Tuple([]),
    ]
    region = ir.RegionStmt("tl.launch_kernel", args, params, attrs, body(params) if body else [ir.Evaluate(params[0])])
    return region, params


def _materialize(function, *, grid=False, threads=False, defaults=(128, 1, 1), unsupported=None):
    transform = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(
        grid,
        threads,
        None if defaults is None else [_integer(x, "int32") for x in defaults],
        unsupported,
        None,
    )
    return transform(tvm_tirx.IRModule({"main": function}))["main"]


def _function(region, params=()):
    return tirx.Function(list(params), [region], ir.NoneType()).with_attr("global_symbol", "main")


def _kernel_body(sequence):
    # Materialization keeps the source owner even when there are no hints.
    # Check that owner explicitly before inspecting the original kernel body.
    assert len(sequence) == 1
    owner = sequence[0]
    assert isinstance(owner, ir.RegionStmt) and owner.op.name == "tl.attr"
    assert not owner.body_params and not owner.result_vars
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(owner.args[0], owner.attrs["annotation_schema"])
    assert metadata["node"] is None and metadata["key"] == "tl.kernel"
    assert not metadata["value"]
    return owner.body


def test_cpu_loop_keeps_begin_end_step_and_identity():
    region, params = _launch(11, begin=2, step=3)
    result = _materialize(_function(region))
    loop = result.body[0]
    assert isinstance(loop, ir.For)
    assert loop.var.same_as(params[0])
    assert tuple(map(int, (loop.begin, loop.end, loop.step))) == (2, 11, 3)
    assert _kernel_body(loop.body)[0].value.same_as(params[0])
    assert tirx.analysis.check_well_formed(result)
    # The pass must not consume or mutate the source region.
    assert region.op.name == "tl.launch_kernel" and len(region.body_params) == 4


def test_cpu_rejects_used_simt_identity():
    region, _ = _launch(body=lambda params: [ir.Evaluate(params[1])])
    with pytest.raises(ValueError, match="has no SIMT threads"):
        _materialize(_function(region))


@pytest.mark.parametrize("specified", [None, (32, 2, 1)])
def test_simt_launch_preserves_explicit_and_default_dimensions(specified):
    region, params = _launch(5, threads=specified)
    result = _materialize(_function(region), grid=True, threads=True)
    body = result.body
    expected = [("blockIdx.x", 5), *zip(("threadIdx.x", "threadIdx.y", "threadIdx.z"), specified or (128, 1, 1))]
    for i, (tag, extent) in enumerate(expected):
        launch = body[0]
        assert isinstance(launch, ir.RegionStmt) and launch.op.name == "tirx.launch_thread"
        assert launch.args[0].value == tag
        assert int(launch.args[1]) == extent
        assert launch.body_params[0].same_as(params[i])
        body = launch.body
    assert tirx.analysis.check_well_formed(result)


def test_missing_backend_default_is_rejected():
    region, _ = _launch()
    with pytest.raises(ValueError, match="backend default"):
        _materialize(_function(region), grid=True, threads=True, defaults=None)


def test_unsupported_hints_are_not_dropped():
    region, _ = _launch(metadata={"cluster_dims": 2})
    with pytest.raises(ValueError, match="cluster_dims.*not supported"):
        _materialize(_function(region), unsupported=["cluster_dims"])


@pytest.mark.parametrize("simt", [False, True])
def test_zero_grid_axes_remains_one_kernel_body(simt):
    params = [ir.Var(name, "int32") for name in ("tx", "ty", "tz")]
    region = ir.RegionStmt(
        "tl.launch_kernel",
        [ir.Tuple([]) for _ in range(5)],
        params,
        {"schema_version": 1, "threads_was_specified": False, "grid_tags": []},
        [ir.Evaluate(ir.const(9))],
    )
    result = _materialize(_function(region), grid=simt, threads=simt)
    assert tirx.analysis.check_well_formed(result)
    body = result.body
    if simt:
        for i in range(3):
            assert body[0].body_params[0].same_as(params[i])
            body = body[0].body
    assert int(_kernel_body(body)[0].value) == 9


def test_dynamic_kernel_hint_remains_a_visible_operand():
    n = ir.Var("n", "int64")
    original, params = _launch(n)
    args = list(original.args)
    args[4] = ir.Tuple([n])
    attrs = dict(original.attrs)
    attrs["hint_schema"] = ["map", [["str", "dynamic_hint"], ["expr", 0]]]
    region = ir.RegionStmt("tl.launch_kernel", args, params, attrs, original.body)
    result = _materialize(_function(region, [n]))
    wrapper = result.body[0].body[0]
    assert wrapper.op.name == "tl.attr"
    assert wrapper.args[0].fields[0].same_as(n)
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(wrapper.args[0], wrapper.attrs["annotation_schema"])
    assert metadata["key"] == "tl.kernel"
    assert metadata["value"]["dynamic_hint"].same_as(n)
    assert tirx.analysis.check_well_formed(result)


def test_device_nonzero_coordinate_uses_distinct_physical_identity():
    region, params = _launch(12, begin=3, step=2)
    result = _materialize(_function(region), grid=True)
    launch = result.body[0]
    assert int(launch.args[1]) == 5
    assert not launch.body_params[0].same_as(params[0])
    binding = launch.body[0]
    assert isinstance(binding, ir.Bind) and binding.var.same_as(params[0])
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is required")
@pytest.mark.parametrize("extent", [1, 17, 257])
def test_tilelang_kernel_region_to_cpu_numeric_execution(extent):
    # This compiles a real TileLang region through its C++ materializer, then the
    # native tensor/ABI/LLVM pipeline. Full frontend and tile-op tests are separate.
    n = ir.Var("n", "int64")
    tensor_type = tirx.TensorType((n,), "float32", layout="default")
    a, b = ir.Var("A", tensor_type), ir.Var("B", tensor_type)
    region, _ = _launch(n, body=lambda params: [ir.TensorStore(b, [params[0]], ir.TensorLoad(a, [params[0]]) + ir.const(1, "float32"))])
    func = _function(region, [n, a, b])
    lowered = _materialize(func)
    assert tirx.analysis.check_well_formed(lowered)
    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"})
    executable = tirx.compile(lowered, target=target)
    expected = np.arange(extent, dtype="float32")
    source = tvm_tirx.runtime.tensor(expected)
    result = tvm_tirx.runtime.tensor(np.zeros_like(expected))
    executable(extent, source, result)
    np.testing.assert_array_equal(result.numpy(), expected + 1)
