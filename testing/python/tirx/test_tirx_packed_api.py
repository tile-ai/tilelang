import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.fixture(autouse=True)
def storage_resolver():
    name = "tl.tvm_ffi.resolve_output_storage_dtype"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda dtype: dtype, override=True)
    yield
    if previous is None:
        tvm_ffi.registry.remove_global_func(name)
    else:
        tvm_ffi.register_global_func(name, previous, override=True)


def packed(function, backend="c", outputs=None):
    function = function.with_attr("target", Target({"kind": backend, "host": backend}))
    function = function.with_attr("global_symbol", "main").with_attr("tirx.is_entry_func", True)
    if outputs is not None:
        function = function.with_attr("tilelang_callee_allocated_outputs", True).with_attr("tilelang_out_idx", outputs)
    module = tvm_ffi.get_global_func("tl.transform.MakePackedAPI")()(ir.IRModule({"main": function}))
    for key in ("tirx.nullable_params", "tirx.required_metadata", "tirx.packed_storage_dtypes"):
        assert key not in module["main"].attrs
    assert tirx.analysis.check_well_formed(module["main"])
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    ir.assert_structural_equal(module, tvm_ffi.get_global_func("tl.transform.MakePackedAPI")()(module))
    return module


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_ordinary_packed_api_uses_native_argument_validation(backend):
    source = TensorView.create("source", [4], "int32")
    function = tirx.Function([source.tensor], ir.Return(source.load(0) + 1), ret_type=ir.PrimType("int32"))
    executable = tirx.compile(packed(function, backend), target=backend)
    assert executable(runtime.tensor(np.array([7, 0, 0, 0], dtype="int32"))) == 8
    with pytest.raises((ValueError, TypeError), match="dtype"):
        executable(runtime.tensor(np.zeros(4, dtype="float32")))
    with pytest.raises((ValueError, TypeError), match="shape"):
        executable(runtime.tensor(np.zeros(3, dtype="int32")))
    with pytest.raises(TypeError, match="Tensor"):
        executable(None)
    with pytest.raises(TypeError, match="Tensor"):
        executable(tvm_ffi.convert([1, 2, 3, 4]))


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (4, 0, 7)])
def test_compact_empty_tensors_accept_arbitrary_strides(backend, shape):
    import torch

    source = ir.Var("source", tirx.TensorType(shape, "int32"))
    function = tirx.Function([source], ir.Return(ir.const(7)), ret_type=ir.PrimType("int32"))
    executable = tirx.compile(packed(function, backend), target=backend)
    strides = tuple(13 + axis * 7 for axis in range(len(shape)))
    assert executable(torch.empty_strided(shape, strides, dtype=torch.int32)) == 7
    with pytest.raises((ValueError, TypeError), match="dtype"):
        executable(torch.empty(shape, dtype=torch.float32))


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_layout_lowering_keeps_symbolic_singleton_stride(backend):
    import torch

    stride = ir.Var("stride", "int64")
    source = TensorView.create("source", [1, 16], "int32", strides=[stride, 1])
    function = tirx.Function([source.tensor], ir.Return(stride), ret_type=stride.ty).with_attr("global_symbol", "main")
    executable = tirx.compile(function, target=backend)
    assert executable(torch.empty_strided((1, 16), (37, 1), dtype=torch.int32)) == 37


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("affine", [False, True])
def test_nullable_metadata_uses_any_available_shape_provider(backend, affine):
    extent = ir.Var("extent", "int64")
    first = TensorView.create("first", [extent * 2 + 1 if affine else extent], "int32")
    second = TensorView.create("second", [extent], "int32")
    function = tirx.Function([first.tensor, second.tensor], ir.Return(extent), ret_type=extent.ty)
    executable = tirx.compile(packed(function, backend), target=backend)
    first_value = runtime.tensor(np.zeros(9 if affine else 4, dtype="int32"))
    second_value = runtime.tensor(np.zeros(4, dtype="int32"))
    assert executable(first_value, second_value) == 4
    assert executable(first_value, None) == 4
    assert executable(None, second_value) == 4
    with pytest.raises(RuntimeError, match="at least one non-null buffer"):
        executable(None, None)
    with pytest.raises(ValueError, match="shape"):
        executable(first_value, runtime.tensor(np.zeros(5, dtype="int32")))
    with pytest.raises(TypeError, match="dtype"):
        executable(None, runtime.tensor(np.zeros(4, dtype="float32")))


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_nullable_single_shape_provider_has_zero_metadata_when_absent(backend):
    extent = ir.Var("extent", "int64")
    source = TensorView.create("source", [extent], "int32")
    function = tirx.Function([source.tensor], ir.Return(extent), ret_type=extent.ty)
    executable = tirx.compile(packed(function, backend), target=backend)
    assert executable(runtime.tensor(np.zeros(7, dtype="int32"))) == 7
    assert executable(None) == 0
    with pytest.raises(TypeError, match="Tensor"):
        executable(tvm_ffi.convert([1, 2, 3]))


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_nullable_unused_shapes_do_not_require_a_provider(backend):
    extent = ir.Var("extent", "int64")
    first = TensorView.create("first", [extent], "int32")
    second = TensorView.create("second", [extent], "int32")
    output = TensorView.create("output", [1], "int32")
    function = tirx.Function(
        [first.tensor, second.tensor, output.tensor],
        ir.Return(output.load(0)),
        ret_type=ir.PrimType("int32"),
    )
    executable = tirx.compile(packed(function, backend), target=backend)
    output_value = runtime.tensor(np.array([31], dtype="int32"))
    assert executable(None, None, output_value) == 31
    with pytest.raises(TypeError, match="Tensor"):
        executable(None, None, None)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_nullable_metadata_prefers_live_tensor_and_scalar_bindings(backend):
    extent = ir.Var("extent", "int64")
    optional = TensorView.create("optional", [extent], "int32")
    source = TensorView.create("source", [extent], "int32")
    function = tirx.Function(
        [optional.tensor, source.tensor],
        ir.Return(source.load(extent - 1)),
        ret_type=ir.PrimType("int32"),
    )
    executable = tirx.compile(packed(function, backend), target=backend)
    value = runtime.tensor(np.array([10, 20, 30], dtype="int32"))
    assert executable(None, value) == 30
    with pytest.raises(ValueError, match="shape"):
        executable(runtime.tensor(np.zeros(2, dtype="int32")), value)
    scalar_function = tirx.Function([optional.tensor, extent], ir.Return(extent), ret_type=extent.ty)
    scalar_executable = tirx.compile(packed(scalar_function, backend), target=backend)
    assert scalar_executable(None, 5) == 5
    assert scalar_executable(value, 3) == 3
    with pytest.raises(ValueError, match="shape"):
        scalar_executable(value, 4)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_nullable_tensor_validates_layout_when_present(backend):
    torch = pytest.importorskip("torch")
    extent = ir.Var("extent", "int64")
    stride = ir.Var("stride", "int64")
    source = TensorView.create("source", [2, extent], "int32", strides=[stride, 1])
    function = tirx.Function([source.tensor], ir.Return(extent * 1000 + stride), ret_type=extent.ty)
    executable = tirx.compile(packed(function, backend), target=backend)
    value = torch.zeros((2, 12), dtype=torch.int32)[:, :8]
    assert executable(value) == 8012
    assert executable(None) == 0
    with pytest.raises(ValueError, match="strides"):
        executable(value[:, ::2])


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize(
    "logical_dtype,storage_dtype",
    [
        ("float4_e2m1fn", "int8"),
        ("int4", "int8"),
        ("uint4", "uint8"),
    ],
)
def test_packed_input_shapes_and_strides_use_logical_elements(backend, logical_dtype, storage_dtype):
    torch = pytest.importorskip("torch")
    tvm_ffi.register_global_func(
        "tl.tvm_ffi.resolve_output_storage_dtype",
        lambda value: tvm_ffi.dtype(storage_dtype),
        override=True,
    )
    extent = ir.Var("extent", "int64")
    stride = ir.Var("stride", "int64")
    source = TensorView.create("source", [2, extent], logical_dtype, strides=[stride, 1])
    function = tirx.Function(
        [source.tensor],
        ir.Return(extent * 1000 + stride),
        ret_type=ir.PrimType("int64"),
    )
    executable = tirx.compile(packed(function, backend), target=backend)
    values = torch.zeros((2, 12), dtype=getattr(torch, storage_dtype))
    assert executable(values[:, :8]) == 16024
    assert executable(values[:, :8].contiguous()) == 16016
    with pytest.raises((ValueError, TypeError), match="dtype"):
        executable(values.float())
    with pytest.raises(ValueError, match="innermost stride"):
        executable(values[:, ::2])


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_packed_input_checks_static_logical_shape(backend):
    torch = pytest.importorskip("torch")
    tvm_ffi.register_global_func(
        "tl.tvm_ffi.resolve_output_storage_dtype",
        lambda value: tvm_ffi.dtype("int8"),
        override=True,
    )
    source = TensorView.create("source", [2, 16], "int4")
    function = tirx.Function([source.tensor], ir.Return(ir.const(7)), ret_type=ir.PrimType("int32"))
    executable = tirx.compile(packed(function, backend), target=backend)
    assert executable(torch.zeros((2, 8), dtype=torch.int8)) == 7
    with pytest.raises(ValueError, match="shape|strides"):
        executable(torch.zeros((2, 7), dtype=torch.int8))


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("size", [0, 1, 17])
def test_dynamic_output_allocation_and_torch_ownership(backend, size):
    torch = pytest.importorskip("torch")
    extent = ir.Var("extent", "int32")
    source = TensorView.create("source", [extent], "float32")
    output = TensorView.create("output", [extent], "float32")
    index = ir.Var("index", "int32")
    function = tirx.Function(
        [source.tensor, output.tensor], ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, output.store(index, source.load(index) + 1))
    )
    module = packed(function, backend, [-1])
    assert len(module["main"].params) == 4
    executable = tirx.compile(module, target=backend)
    source_values = torch.arange(size, dtype=torch.float32)
    for _ in range(3):
        result = executable(source_values, source_values)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, source_values + 1)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_multiple_outputs_preserve_requested_order(backend):
    torch = pytest.importorskip("torch")
    source = TensorView.create("source", [4], "int32")
    first = TensorView.create("first", [4], "int32")
    second = TensorView.create("second", [4], "int32")
    index = ir.Var("index", "int32")
    body = ir.For(
        index, 0, 4, 1, ir.ForKind.DEFAULT, [first.store(index, source.load(index) + 1), second.store(index, source.load(index) + 5)]
    )
    function = tirx.Function([first.tensor, source.tensor, second.tensor], body)
    executable = tirx.compile(packed(function, backend, [-1, 0]), target=backend)
    values = torch.arange(4, dtype=torch.int32)
    result = executable(values, values)
    assert len(result) == 2
    torch.testing.assert_close(result[0], values + 5)
    torch.testing.assert_close(result[1], values + 1)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_scalar_only_input_uses_allocator_anchor(backend):
    torch = pytest.importorskip("torch")
    extent = ir.Var("extent", "int32")
    output = TensorView.create("output", [extent], "int32")
    index = ir.Var("index", "int32")
    function = tirx.Function([extent, output.tensor], ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, output.store(index, index * 3)))
    executable = tirx.compile(packed(function, backend, [1]), target=backend)
    anchor = torch.empty(0)
    torch.testing.assert_close(executable(7, anchor), torch.arange(7, dtype=torch.int32) * 3)
    with pytest.raises(RuntimeError, match="anchor must be a Tensor"):
        executable(7, None)
    with pytest.raises(RuntimeError, match="dimensions must be nonnegative"):
        executable(-1, anchor)


@pytest.mark.parametrize("indices,pattern", [([], "non-empty"), ([1, -1], "duplicate"), ([3], "out-of-range"), ([0], "not a tensor")])
def test_invalid_output_indices_are_rejected(indices, pattern):
    scalar = ir.Var("scalar", "int32")
    output = TensorView.create("output", [1], "int32")
    function = tirx.Function([scalar, output.tensor], output.store(0, scalar))
    with pytest.raises((ValueError, TypeError), match=pattern):
        packed(function, outputs=indices)


def test_output_only_symbol_is_not_guessed():
    extent = ir.Var("extent", "int32")
    output = TensorView.create("output", [extent], "int32")
    function = tirx.Function([output.tensor], ir.Evaluate(0))
    with pytest.raises(ValueError, match="Cannot infer symbolic variable"):
        packed(function, outputs=[0])


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_dynamic_output_stride_is_bound_from_compact_allocation(backend):
    torch = pytest.importorskip("torch")
    stride = ir.Var("stride", "int32")
    offset = ir.Var("offset", "int32")
    output = TensorView.create("output", [2, 3], "int32", strides=[stride, 1], elem_offset=offset)
    function = tirx.Function([output.tensor], output.store([1, 2], 41))
    executable = tirx.compile(packed(function, backend, [0]), target=backend)
    result = executable(torch.empty(0))
    assert result.shape == (2, 3)
    assert result[1, 2].item() == 41


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_packed_output_storage_shape_and_alignment(backend):
    torch = pytest.importorskip("torch")
    tvm_ffi.register_global_func("tl.tvm_ffi.resolve_output_storage_dtype", lambda dtype: tvm_ffi.dtype("float32"), override=True)
    extent = ir.Var("extent", "int32")
    output = TensorView.create("output", [extent], "float16")
    function = tirx.Function([extent, output.tensor], ir.Evaluate(0))
    module = packed(function, backend, [1])
    executable = tirx.compile(module, target=backend)
    anchor = torch.empty(0)
    result = executable(8, anchor)
    assert result.dtype == torch.float32
    assert result.shape == (4,)
    with pytest.raises(RuntimeError, match="not storage-aligned"):
        executable(7, anchor)


def test_noncompact_output_fails_before_allocation():
    torch = pytest.importorskip("torch")
    output = TensorView.create("output", [2, 3], "int32", strides=[7, 1])
    executable = tirx.compile(packed(tirx.Function([output.tensor], ir.Evaluate(0)), outputs=[0]), target="c")
    with pytest.raises(RuntimeError, match="output strides must be compact"):
        executable(torch.empty(0))


def test_runtime_assumptions_retain_native_checks():
    scalar = ir.Var("scalar", "int32")
    marker = ir.RegionStmt(
        ir.Op.get("tirx.scope"),
        [],
        [],
        ir.DictAttrs({"tl.assume_requires_runtime_check": scalar > 0, "tl.assume_message": "scalar must be positive"}),
        ir.Return(scalar),
    )
    function = tirx.Function([scalar], marker, ret_type=ir.PrimType("int32"))
    module = packed(function)
    checks = []
    stmt_functor.post_order_visit(module["main"], lambda node: checks.append(node) if isinstance(node, ir.AssertStmt) else None)
    assert any("scalar must be positive" in str(node.message_parts) for node in checks)
    executable = tirx.compile(module, target="c")
    assert executable(3) == 3
    with pytest.raises(RuntimeError, match="scalar must be positive"):
        executable(0)
