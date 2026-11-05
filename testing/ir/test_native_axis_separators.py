"""Source segmented storage maps to native addresses without changing the ABI."""

import itertools

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx

import tilelang
import tilelang.language as T
from tilelang.language._native import buffer
from tilelang.testing.packed import compile_packed_function


# Literal address coefficients from the original Buffer offset/physical-shape
# contract for shape (2, 3, 4). Explicit contiguous strides are intentionally
# distinct from omitted strides; the former use segmented stride*extent sizes.
CASES = [
    ((), (1,), (12, 4, 1)),
    ((), (2,), (12, 4, 1)),
    ((), (1, 2), (12, 4, 1)),
    ((), (0,), (12, 4, 1)),
    ((), (3,), (12, 4, 1)),
    ((), (1, 1), (12, 16, 4)),
    ((20, 4, 1), (1,), (240, 4, 1)),
    ((20, 4, 1), (2,), (80, 16, 1)),
    ((20, 4, 1), (1, 2), (960, 16, 1)),
    ((20, 4, 1), (0,), (20, 4, 1)),
    ((20, 4, 1), (1, 1), (2880, 48, 12)),
    ((12, 4, 1), (1,), (144, 4, 1)),
]


def lower(function):
    result = tilelang.transform.LowerTensorSpec()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    return result


def parameter_kernel(strides, separators):
    spec = buffer((2, 3, 4), "int32", strides=strides, axis_separators=separators, align=4)

    @T.prim_func
    def copy(A: spec, B: T.Tensor((2, 3, 4), "int32")):
        for i in T.serial(2):
            for j in T.serial(3):
                for k in T.serial(4):
                    B[i, j, k] = A[i, j, k]

    return copy


@pytest.mark.parametrize("strides,separators,expected", CASES)
def test_segmented_parameter_retains_source_abi_and_old_addresses(strides, separators, expected):
    function = parameter_kernel(strides, separators)
    result = lower(function)
    assert result.params[0].same_as(function.params[0])
    tensors = []
    tvm_ffi.structural_walk(result.body, (ir.TensorLoad, lambda load: tensors.append(load.source)))
    (tensor,) = tensors
    analyzer = sym.Analyzer()
    for point in itertools.product(range(2), range(3), range(4)):
        mapped = tensor.ty.layout.apply(*[ir.IntImm("int32", index) for index in point], shape=tensor.ty.shape)["m"]
        assert int(analyzer.simplify(mapped)) == sum(index * stride for index, stride in zip(point, expected, strict=True))
    # The resulting native function is self-contained under clone/serialization.
    restored = ir.load_json(ir.save_json(result))
    assert tirx.analysis.check_well_formed(restored)
    assert tvm_ffi.structural_equal(result, restored)


@pytest.mark.parametrize("strides,separators,expected", CASES)
@pytest.mark.parametrize("consumer", ["native_c", "tilelang_c"])
def test_segmented_parameter_executes_real_address_map(strides, separators, expected, consumer):
    function = parameter_kernel(strides, separators)
    if consumer == "native_c":
        executable = compile_packed_function(lower(function).with_attr("global_symbol", "main"), tilelang_owned=False)
    else:
        executable = tilelang.compile(function, target="c", target_host="c", execution_backend="tvm_ffi")
    capacity = 1 + expected[0] + 2 * expected[1] + 3 * expected[2]
    raw = np.arange(max(capacity, 40), dtype="int32") * 3 + 11
    source_strides = strides or (12, 4, 1)
    source = np.ndarray((2, 3, 4), dtype="int32", buffer=raw, strides=tuple(step * 4 for step in source_strides))
    output = tvm_tirx.runtime.tensor(np.zeros((2, 3, 4), dtype="int32"))
    executable(tvm_ffi.from_dlpack(source), output)
    oracle = np.empty((2, 3, 4), dtype="int32")
    for point in np.ndindex(oracle.shape):
        oracle[point] = raw[sum(index * stride for index, stride in zip(point, expected, strict=True))]
    np.testing.assert_array_equal(output.numpy(), oracle)


@pytest.mark.parametrize(
    "strides,separators,expected,capacity",
    [
        ((), (1,), (12, 4, 1), 24),
        ((20, 4, 1), (1,), (240, 4, 1), 480),
        ((20, 4, 1), (1, 1), (2880, 48, 12), 5760),
    ],
)
@pytest.mark.parametrize("consumer", ["native_c", "tilelang_c"])
def test_owned_segmented_storage_and_raw_alias_cover_declared_capacity(strides, separators, expected, capacity, consumer):
    @T.prim_func
    def write(B: T.Tensor((capacity,), "int32")):
        workspace = T.alloc_buffer((2, 3, 4), "int32", strides=strides, axis_separators=separators)
        raw = T.decl_buffer((capacity,), "int32", data=tirx.tensor_data_ptr(workspace))
        for offset in T.serial(capacity):
            raw[offset] = -1
        for i in T.serial(2):
            for j in T.serial(3):
                for k in T.serial(4):
                    workspace[i, j, k] = 100 * i + 10 * j + k
        for offset in T.serial(capacity):
            B[offset] = raw[offset]

    native = lower(write)
    allocations = []
    tvm_ffi.structural_walk(
        native.body,
        (
            ir.Bind,
            lambda node: (
                allocations.append(node) if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor" else None
            ),
        ),
    )
    assert len(allocations) == 1
    assert int(allocations[0].var.ty.shape[0]) >= capacity
    if consumer == "native_c":
        executable = compile_packed_function(native.with_attr("global_symbol", "main"), tilelang_owned=False)
    else:
        executable = tilelang.compile(write, target="c", target_host="c", execution_backend="tvm_ffi")
    output = tvm_tirx.runtime.tensor(np.zeros(capacity, dtype="int32"))
    executable(output)
    oracle = np.full(capacity, -1, dtype="int32")
    for i, j, k in itertools.product(range(2), range(3), range(4)):
        oracle[i * expected[0] + j * expected[1] + k * expected[2]] = 100 * i + 10 * j + k
    np.testing.assert_array_equal(output.numpy(), oracle)


@pytest.mark.parametrize("offset", [1, "symbolic"])
def test_segmented_nonzero_or_unknown_origin_rejects_at_consumer(offset):
    elem_offset = T.dynamic("offset") if offset == "symbolic" else offset
    spec = buffer((2, 3), "int32", elem_offset=elem_offset, axis_separators=(1,))

    @T.prim_func
    def invalid(A: spec):
        A[0, 0] = 1

    with pytest.raises(tilelang.errors.InternalError, match="one element offset for each output index"):
        lower(invalid)


def test_explicit_stride_terminal_separator_has_diagnostic():
    spec = buffer((2, 3), "int32", strides=(3, 1), axis_separators=(2,))

    @T.prim_func
    def invalid(A: spec):
        A[0, 0] = 1

    with pytest.raises(ValueError, match="at tensor rank has no physical storage extent"):
        lower(invalid)


@pytest.mark.parametrize("extent", [0, 1, 7])
def test_dynamic_segmented_parameter_executes_including_empty_domain(extent):
    n = T.dynamic("n", "int64")
    spec = buffer((n, 3), "int32", strides=(8, 1), axis_separators=(1,), align=4)

    @T.prim_func
    def copy(A: spec, B: T.Tensor((n, 3), "int32")):
        for i in T.serial(n):
            for j in T.serial(3):
                B[i, j] = A[i, j]

    executable = tilelang.compile(copy, target="c", target_host="c", execution_backend="tvm_ffi")
    raw = np.arange(max(24 * extent, 1), dtype="int32") + 17
    source = np.ndarray((extent, 3), dtype="int32", buffer=raw, strides=(32, 4))
    output = tvm_tirx.runtime.tensor(np.zeros((extent, 3), dtype="int32"))
    executable(tvm_ffi.from_dlpack(source), output)
    oracle = np.empty((extent, 3), dtype="int32")
    for i, j in np.ndindex(oracle.shape):
        oracle[i, j] = raw[i * 24 + j]
    np.testing.assert_array_equal(output.numpy(), oracle)


@pytest.mark.parametrize("separators", [(0,), (0, 0)])
def test_rank_zero_owned_storage_and_alias(separators):
    @T.prim_func
    def scalar(B: T.Tensor((1,), "int32")):
        value = T.alloc_buffer((), "int32", axis_separators=separators)
        value[()] = 41
        B[0] = value[()] + 1

    executable = tilelang.compile(scalar, target="c", target_host="c", execution_backend="tvm_ffi")
    output = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
    executable(output)
    np.testing.assert_array_equal(output.numpy(), np.array([42], dtype="int32"))


@pytest.mark.parametrize("scope", ["local", "shared"])
@pytest.mark.parametrize("explicit_layout", [False, True])
def test_segmented_storage_with_tile_copy_and_layout_remapping(scope, explicit_layout):
    layout = tilelang.layout.Layout((2, 3, 4), lambda i, j, k: i * 12 + j * 4 + k)

    @T.prim_func
    def copy(A: T.Tensor((2, 3, 4), "int32"), B: T.Tensor((2, 3, 4), "int32")):
        with T.Kernel(1):
            workspace = T.alloc_buffer((2, 3, 4), "int32", strides=(20, 4, 1), axis_separators=(1,), scope=scope)
            if explicit_layout:
                T.annotate_layout({workspace: layout})
            T.copy(A, workspace)
            T.copy(workspace, B)

    executable = tilelang.compile(copy, target="c", target_host="c", execution_backend="tvm_ffi")
    values = np.arange(24, dtype="int32").reshape(2, 3, 4) * 7 + 3
    source = tvm_tirx.runtime.tensor(values)
    output = tvm_tirx.runtime.tensor(np.zeros((2, 3, 4), dtype="int32"))
    executable(source, output)
    np.testing.assert_array_equal(output.numpy(), values)
