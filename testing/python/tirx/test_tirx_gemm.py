import itertools

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def gemm_args(left, right, output, rows, columns, reduction, transpose_left=False, transpose_right=False, clear=False):
    return [
        left,
        right,
        output,
        ir.const(transpose_left, "bool"),
        ir.const(transpose_right, "bool"),
        ir.const(rows),
        ir.const(columns),
        ir.const(reduction),
        ir.const(0),
        clear if isinstance(clear, ir.Expr) else ir.const(clear, "bool"),
        ir.const(0),
        ir.const(0),
        ir.const(0),
    ]


def parse(arguments, name="gemm", **annotations):
    return native("ParseOperator", ir.Call(f"tl.tileop.{name}", arguments, attrs=annotations), {})


def lower(operation, target):
    with target:
        return native("testing.LowerOperator", operation, target, ir.Range(0, 1), ir.const(0), {}, {})


@pytest.mark.parametrize("transpose_left,transpose_right,clear", list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize("dtype,accum_dtype", [("float32", "float32"), ("float16", "float32"), ("int8", "int32")])
@pytest.mark.parametrize("leading", [(), (1,)])
def test_cpu_gemm_native_regions_and_accumulator(transpose_left, transpose_right, clear, dtype, accum_dtype, leading, *, compile_tilelang):
    rows, columns, reduction = 5, 7, 3
    left_shape = (reduction, rows) if transpose_left else (rows, reduction)
    right_shape = (columns, reduction) if transpose_right else (reduction, columns)
    left = TensorView.create("left", leading + tuple(extent + 2 for extent in left_shape), dtype)
    right = TensorView.create("right", leading + tuple(extent + 4 for extent in right_shape), dtype)
    output = TensorView.create("output", leading + (rows + 6, columns + 6), accum_dtype)
    leading_regions = [ir.Range(0, 1) for _ in leading]
    left_region = left.region(leading_regions + [ir.Range(1, left_shape[0] + 1), ir.Range(1, left_shape[1] + 1)])
    right_region = right.region(leading_regions + [ir.Range(2, right_shape[0] + 2), ir.Range(2, right_shape[1] + 2)])
    output_region = output.region(leading_regions + [ir.Range(3, rows + 3), ir.Range(3, columns + 3)])
    operation = parse(gemm_args(left_region, right_region, output_region, rows, columns, reduction, transpose_left, transpose_right, clear))
    reads, writes = native("testing.GetOperatorAccessRegions", operation)
    assert len(reads) == (2 if clear else 3)
    ir.assert_structural_equal(writes[0], output_region)
    cloned = native("testing.CloneOperator", operation)
    assert cloned.a.same_as(left.tensor)
    target = Target("c")
    body = lower(cloned, target)
    function = tirx.Function([left.tensor, right.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "matrix_product"
    )
    assert ir.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    executable = compile_tilelang(function, target=target)
    generator = np.random.default_rng(1)
    left_values = generator.integers(-3, 4, tuple(int(extent) for extent in left.shape)).astype(dtype)
    right_values = generator.integers(-3, 4, tuple(int(extent) for extent in right.shape)).astype(dtype)
    expected = np.full(tuple(int(extent) for extent in output.shape), 17, dtype=accum_dtype)
    actual = runtime.tensor(expected.copy())
    executable(runtime.tensor(left_values), runtime.tensor(right_values), actual)
    left_matrix = left_values[..., 1 : left_shape[0] + 1, 1 : left_shape[1] + 1].astype(accum_dtype)
    right_matrix = right_values[..., 2 : right_shape[0] + 2, 2 : right_shape[1] + 2].astype(accum_dtype)
    if transpose_left:
        left_matrix = left_matrix.swapaxes(-2, -1)
    if transpose_right:
        right_matrix = right_matrix.swapaxes(-2, -1)
    expected[..., 3 : rows + 3, 3 : columns + 3] = left_matrix @ right_matrix + (0 if clear else 17)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("reduction", [0, 3])
def test_cpu_gemm_dynamic_clear(reduction, *, compile_tilelang):
    left = TensorView.create("left", [2, reduction], "float32")
    right = TensorView.create("right", [reduction, 2], "float32")
    output = TensorView.create("output", [2, 2], "float32")
    clear = ir.Var("clear", "bool")
    operation = parse(gemm_args(left.tensor, right.tensor, output.tensor, 2, 2, reduction, clear=clear))
    assert len(native("testing.GetOperatorAccessRegions", operation)[0]) == 3
    assert len(native("testing.GetReadBeforeWriteRegions", operation)) == 2
    function = tirx.Function(
        [left.tensor, right.tensor, output.tensor, clear], lower(operation, Target("c")), ret_type=ir.TupleType([])
    ).with_attr("global_symbol", "dynamic_clear")
    executable = compile_tilelang(function, target="c")
    for flag in (True, False):
        actual = runtime.tensor(np.full((2, 2), 7, dtype="float32"))
        executable(
            runtime.tensor(np.ones((2, reduction), dtype="float32")), runtime.tensor(np.ones((reduction, 2), dtype="float32")), actual, flag
        )
        np.testing.assert_array_equal(actual.numpy(), np.full((2, 2), reduction + (0 if flag else 7), dtype="float32"))


@pytest.mark.parametrize("name,flag", [("wgmma_gemm", "isWgmma"), ("tcgen05_gemm", "isTcgen05")])
def test_gemm_native_boolean_annotations(name, flag):
    left = TensorView.create("left", [64, 32], "float16", scope="shared")
    right = TensorView.create("right", [32, 64], "float16", scope="shared")
    output = TensorView.create("output", [64, 64], "float32", scope="local.fragment")
    operation = parse(gemm_args(left.tensor, right.tensor, output.tensor, 64, 64, 32), name=name)
    assert getattr(operation, flag)


@pytest.mark.parametrize("as_region", [False, True])
def test_gemm_preserves_barrier_tensor_element(as_region):
    left = TensorView.create("left", [64, 32], "float16", scope="shared")
    right = TensorView.create("right", [32, 64], "float16", scope="shared")
    output = TensorView.create("output", [64, 64], "float32", scope="shared.tmem")
    barrier = TensorView.create("barrier", [4], "uint64", scope="shared.barrier")
    arguments = gemm_args(left.tensor, right.tensor, output.tensor, 64, 64, 32)
    arguments[10] = barrier.region([ir.Range(2, 3)]) if as_region else barrier.load([ir.const(2)])
    operation = parse(arguments)
    ir.assert_structural_equal(operation.mbar, barrier.load([ir.const(2)]))


def test_gemm_rejects_multi_element_barrier():
    left = TensorView.create("left", [64, 32], "float16", scope="shared")
    right = TensorView.create("right", [32, 64], "float16", scope="shared")
    output = TensorView.create("output", [64, 64], "float32", scope="shared.tmem")
    barrier = TensorView.create("barrier", [4], "uint64", scope="shared.barrier")
    arguments = gemm_args(left.tensor, right.tensor, output.tensor, 64, 64, 32)
    arguments[10] = barrier.region([ir.Range(1, 3)])
    with pytest.raises(ValueError, match="single mbarrier element"):
        parse(arguments)


def test_blockscaled_gemm_never_falls_back_to_dense_cpu():
    left = TensorView.create("left", [16, 32], "float16")
    right = TensorView.create("right", [32, 16], "float16")
    output = TensorView.create("output", [16, 16], "float32")
    scales = TensorView.create("scales", [16, 1], "uint8")
    arguments = gemm_args(left.tensor, right.tensor, output.tensor, 16, 16, 32)
    arguments.extend([scales.tensor, scales.tensor, ir.const(0)])
    operation = parse(arguments, name="gemm_blockscaled")
    assert len(native("testing.GetOperatorAccessRegions", operation)[0]) == 5
    with pytest.raises(ValueError, match="Block-scaled GEMM is not supported"):
        lower(operation, Target("c"))


def test_sparse_gemm_retains_metadata_access():
    left = TensorView.create("left", [16, 16], "float16", scope="shared")
    metadata = TensorView.create("metadata", [16, 2], "uint32", scope="local.fragment")
    right = TensorView.create("right", [32, 16], "float16", scope="shared")
    output = TensorView.create("output", [16, 16], "float32", scope="local.fragment")
    arguments = [left.tensor, metadata.tensor, right.tensor, output.tensor]
    arguments.extend(ir.const(False, "bool") for _ in range(3))
    arguments.extend(ir.const(value) for value in [16, 16, 32, 0])
    arguments.append(ir.const(True, "bool"))
    operation = parse(arguments, name="gemm_sp", wg_wait=0, is_wgmma=False)
    reads, writes = native("testing.GetOperatorAccessRegions", operation)
    assert len(reads) == 3 and len(writes) == 1
    assert reads[1].source.same_as(metadata.tensor)


def test_native_gemm_instruction_selection():
    left = TensorView.create("left", [64, 32], "float16", scope="shared")
    right = TensorView.create("right", [32, 64], "float16", scope="shared")
    output = TensorView.create("output", [64, 64], "float32", scope="local.fragment")
    operation = parse(gemm_args(left.tensor, right.tensor, output.tensor, 64, 64, 32))
    for arch, expected in [("sm_80", "cuda.mma"), ("sm_90", "cuda.wgmma")]:
        assert native("GemmGetGemmInstructionKey", operation, 128, Target({"kind": "cuda", "arch": arch})) == expected
    assert native("GemmGetGemmInstructionKey", operation, 1, Target("c")) == "cpu.scalar"
