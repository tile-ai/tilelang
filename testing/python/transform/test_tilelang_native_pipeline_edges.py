"""Edge probes: native loop ordinals and explicit strided alias geometry."""

import numpy as np
import pytest
from tvm_tirx import ir, tirx, runtime

import tilelang as tl
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def inject(function):
    original = ir.save_json(function)
    after = tl.transform.InjectSoftwarePipeline()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == original
    assert tirx.analysis.check_well_formed(after)
    assert not tirx.analysis.undefined_vars(after)
    return after


def execute(function, data, expected):
    code = compile_packed_function(function, c_codegen="compute")
    result = runtime.tensor(np.full(expected.shape, -99, dtype="int32"))
    code(runtime.tensor(data), result)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("bounds", [(1, 8, 3), (8, 0, -3), (1, 8, -2)])
def test_public_stepped_pipeline_keeps_source_range(bounds):
    begin, end, step = bounds

    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        for i in T.serial(begin, end, step, annotations={"software_pipeline_stage": [0, 2], "software_pipeline_order": [0, 1]}):
            temporary[0] = A[i] + 2
            B[i] = temporary[0] * 3

    data = np.arange(16, dtype="int32") * 2 - 5
    expected = np.full(16, -99, dtype="int32")
    for index in range(*bounds):
        expected[index] = (data[index] + 2) * 3
    execute(kernel, data, expected)
    execute(inject(kernel), data, expected)


def raw_function(dtype, begin, end, step=1):
    source = ir.Var("A", tirx.TensorType((256,), "int32"))
    output = ir.Var("B", tirx.TensorType((256,), "int32"))
    temporary = ir.Var("tmp", tirx.TensorType((1,), "int32", scope="local"))
    index = ir.Var("index", dtype)
    offset = index.astype("int64") - ir.const(begin, "int64")
    body = [
        ir.Bind(temporary, ir.Call("tirx.alloc_tensor", [], ty_args=[temporary.ty])),
        ir.For(
            index,
            ir.const(begin, dtype),
            ir.const(end, dtype),
            ir.const(step, dtype),
            ir.ForKind.DEFAULT,
            [
                ir.TensorStore(temporary, [0], ir.TensorLoad(source, [offset]) + 2),
                ir.TensorStore(output, [offset], ir.TensorLoad(temporary, [0]) * 3),
            ],
            annotations={"software_pipeline_stage": [0, 3], "software_pipeline_order": [0, 1]},
        ),
    ]
    return tirx.Function([source, output], body, ir.NoneType()).with_attr({"global_symbol": "main", "tirx.noalias": True})


@pytest.mark.parametrize(
    "dtype, begin, end",
    [
        ("int32", 5, 10),
        ("int64", 2**31 + 3, 2**31 + 8),
        ("uint32", 3, 8),
        ("int64", 11, 11),
        ("int8", -100, 100),
        ("uint8", 250, 255),
    ],
)
def test_raw_unit_step_pipeline_keeps_origin_and_index_type(dtype, begin, end):
    kernel = raw_function(dtype, begin, end)
    data = np.arange(256, dtype="int32") - 31
    expected = np.full(256, -99, dtype="int32")
    expected[: end - begin] = (data[: end - begin] + 2) * 3
    execute(kernel, data, expected)
    execute(inject(kernel), data, expected)


@pytest.mark.parametrize("step", [2, 3])
def test_raw_stepped_pipeline_requires_explicit_ordinal_normalization(step):
    # Native raw For has a distinct step field. TileLang's public SerialFor
    # already normalizes it; the low-level pipeline pass must reject raw steps
    # it cannot represent rather than quietly treating them as unit steps.
    with pytest.raises(ValueError, match="unit-step"):
        inject(raw_function("int32", 1, 8, step))


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("origin", [0, 3])
def test_strided_alias_pipeline_uses_current_version_and_preserves_origin(inside, origin):
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32"), B: T.Tensor((7,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((16,), "int32", scope="local")
        if not inside:
            alias = T.decl_buffer((2,), "int32", data=temporary.data, strides=(3,), elem_offset=origin, scope="local")
        for i in T.serial(7, annotations={"software_pipeline_stage": [0, 2], "software_pipeline_order": [0, 1]}):
            if inside:
                alias = T.decl_buffer((2,), "int32", data=temporary.data, strides=(3,), elem_offset=origin, scope="local")
            temporary[origin + 3] = A[i] + 7
            B[i] = alias[1] * 5

    data = np.arange(7, dtype="int32") - 10
    expected = (data + 7) * 5
    execute(kernel, data, expected)
    execute(inject(kernel), data, expected)
