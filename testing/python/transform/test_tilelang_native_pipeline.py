"""Native pipeline storage, definition scope, and numerical invariants."""

import numpy as np
import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang as tl
import tilelang.language as T
from tilelang.testing.ir import collect_nodes
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("non_restrict", [False, True])
def test_replay_rejects_potentially_written_parameter_alias(non_restrict):
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32"), B: T.Tensor((7,), "int32")):
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        for index in T.serial(
            7,
            annotations={
                "software_pipeline_stage": [0, 2],
                "software_pipeline_order": [0, 1],
                "software_pipeline_replayable_scalar_binds": [1, 0, 0],
            },
        ):
            value: T.int32 = A[index]
            temporary[0] = value
            B[index] = temporary[0] + value

    if non_restrict:
        kernel = kernel.with_attr({"tirx.noalias": True, "tl.may_alias_param_indices": [0, 1]})
    with pytest.raises(RuntimeError, match="cannot be replayed safely"):
        inject(kernel)


def inject(function):
    before = tvm_tirx.ir.save_json(function)
    module = tvm_tirx.IRModule.from_expr(function.with_attr("global_symbol", "main"))
    module = tl.transform.InjectSoftwarePipeline()(module)
    assert tvm_tirx.ir.save_json(function) == before
    function = module["main"]
    assert tirx.analysis.check_well_formed(function)
    return function


def execute(function, data):
    compiled = compile_packed_function(function, c_codegen="compute")
    source = tvm_tirx.runtime.tensor(data, tvm_tirx.cpu())
    result = tvm_tirx.runtime.tensor(np.zeros_like(data), tvm_tirx.cpu())
    compiled(source, result)
    return result.numpy()


@pytest.mark.parametrize("extent", [0, 1, 2, 5, 9])
@pytest.mark.parametrize("distance", [1, 2, 3])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_native_pipeline_versions_numeric(extent, distance, order):
    @T.prim_func
    def original(A: T.Tensor((extent,), "int32"), B: T.Tensor((extent,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        for index in T.serial(extent, annotations={"software_pipeline_stage": [0, distance], "software_pipeline_order": order}):
            temporary[0] = A[index] + 1
            B[index] = temporary[0] * 3

    transformed = inject(original)
    allocations = [
        node
        for node in collect_nodes(transformed, ir.Bind)
        if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor"
    ]
    assert len(allocations) == 1
    assert int(allocations[0].var.ty.shape[-1]) == 1
    data = np.arange(extent, dtype="int32") - 4
    expected = (data + 1) * 3
    np.testing.assert_array_equal(execute(original, data), expected)
    np.testing.assert_array_equal(execute(transformed, data), expected)


def test_native_pipeline_replays_readonly_scalar_chain():
    @T.prim_func
    def original(A: T.Tensor((7,), "int32"), B: T.Tensor((7,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        for index in T.serial(7, annotations={"software_pipeline_stage": [0, 2], "software_pipeline_order": [0, 1]}):
            value: T.int32 = A[index]
            shifted: T.int32 = value + index
            temporary[0] = shifted
            B[index] = temporary[0] + shifted

    transformed = inject(original)
    data = np.arange(7, dtype="int32") * 2 - 3
    expected = 2 * (data + np.arange(7, dtype="int32"))
    np.testing.assert_array_equal(execute(original, data), expected)
    np.testing.assert_array_equal(execute(transformed, data), expected)


@pytest.mark.parametrize("distance", [1, 2, 3])
def test_zero_iteration_preserves_nonempty_output(distance):
    @T.prim_func
    def original(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        B[0] = 123
        for index in T.serial(0, annotations={"software_pipeline_stage": [0, distance], "software_pipeline_order": [0, 1]}):
            temporary[0] = A[index]
            B[index] = temporary[0]

    data = np.array([7], dtype="int32")
    np.testing.assert_array_equal(execute(original, data), [123])
    np.testing.assert_array_equal(execute(inject(original), data), [123])


@pytest.mark.parametrize("inside", [False, True])
def test_versioned_alias_reads_selected_ring_slot(inside):
    @T.prim_func
    def original(A: T.Tensor((7,), "int32"), B: T.Tensor((7,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        if not inside:
            alias = T.decl_buffer((1,), "int32", data=T.tensor_data_ptr(temporary), scope="local")
        for index in T.serial(7, annotations={"software_pipeline_stage": [0, 2], "software_pipeline_order": [0, 1]}):
            if inside:
                alias = T.decl_buffer((1,), "int32", data=T.tensor_data_ptr(temporary), scope="local")
            temporary[0] = A[index] + 5
            B[index] = alias[0] * 2

    data = np.arange(7, dtype="int32") - 3
    expected = (data + 5) * 2
    np.testing.assert_array_equal(execute(original, data), expected)
    np.testing.assert_array_equal(execute(inject(original), data), expected)


@pytest.mark.parametrize("size", [1, 2, 7, 11])
def test_dynamic_extent_keeps_pipeline_guards(size):
    n = T.dynamic("n")

    @T.prim_func
    def original(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        T.func_attr({"tirx.noalias": True})
        temporary = T.alloc_buffer((1,), "int32", scope="local")
        for index in T.serial(n, annotations={"software_pipeline_stage": [0, 3], "software_pipeline_order": [0, 1]}):
            temporary[0] = A[index] + 2
            B[index] = temporary[0] * 7

    data = np.arange(size, dtype="int32") - 3
    expected = (data + 2) * 7
    np.testing.assert_array_equal(execute(original, data), expected)
    np.testing.assert_array_equal(execute(inject(original), data), expected)
