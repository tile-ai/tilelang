"""Native TileLang bounds/padding passes executed through its packed C host."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.cpu.language as T
from tilelang.language import _native
from tilelang.testing.packed import compile_packed_function


def lower(function):
    module = tvm_tirx.IRModule({"main": function.with_attr("global_symbol", "main")})
    module = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, False, None, None, None)(module)
    for name in ("LowerTensorSpec", "LegalizeNegativeIndex", "LegalizeSafeMemoryAccess", "LowerAccessPtr"):
        module = tvm_ffi.get_global_func("tl.transform." + name)()(module)
    return module


def execute(module, *arrays):
    executable = compile_packed_function(module)
    tensors = [tvm_ffi.from_dlpack(array) for array in arrays]
    executable(*tensors)
    return executable


@pytest.mark.parametrize("extent", [0, 1, 17, 257])
@pytest.mark.parametrize("padding", [0.0, -3.5])
def test_symbolic_load_padding_and_store_bounds(extent, padding):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def padded(A: T.Buffer((n,), "float32", align=4), B: T.Buffer((n + 2,), "float32", align=4)):
        with T.Kernel(1):
            T.annotate_safe_value({A: padding})
            for i in T.serial(n + 4):
                B[i] = A[i - 1]

    source = np.arange(extent, dtype="float32")
    result = np.full(extent + 2, 123.0, dtype="float32")
    module = lower(padded)
    assert tirx.analysis.check_well_formed(module)
    execute(module, source, result)
    np.testing.assert_array_equal(result, np.pad(source, (1, 1), constant_values=padding))


def test_safe_value_scope_restores_enclosing_annotation():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def scoped(A: T.Buffer((n,), "float32", align=4), B: T.Buffer((4,), "float32", align=4)):
        with T.Kernel(1):
            T.annotate_safe_value({A: -2.0})
            B[0] = A[n]
            with _native.attr(None, "safe_value_map", {A: -5.0}):
                B[1] = A[n]
            B[2] = A[n]
            B[3] = A[0]

    source = np.array([7.0, 8.0], dtype="float32")
    result = np.empty(4, dtype="float32")
    execute(lower(scoped), source, result)
    np.testing.assert_array_equal(result, [-2, -5, -2, 7])


def test_provably_negative_index_wraps_once():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def last(A: T.Buffer((n,), "float32", align=4), B: T.Buffer((1,), "float32", align=4)):
        with T.Kernel(1):
            B[0] = A[-1]

    source = np.arange(17, dtype="float32")
    result = np.empty(1, dtype="float32")
    execute(lower(last), source, result)
    np.testing.assert_array_equal(result, [16])


def test_typed_strided_view_offset_is_applied_once():
    n = T.dynamic("n", "int64")
    stride = T.dynamic("stride", "int64")

    @T.prim_func
    def view(A: T.Buffer((n,), "float32", strides=(stride,), align=4), B: T.Buffer((n + 1,), "float32", align=4)):
        with T.Kernel(1):
            T.annotate_safe_value({A: -1.0})
            for i in T.serial(n + 1):
                B[i] = A[i]

    source = np.arange(40, dtype="float32")[3:33:3]
    result = np.empty(len(source) + 1, dtype="float32")
    execute(lower(view), source, result)
    np.testing.assert_array_equal(result, np.concatenate((source, np.array([-1], dtype="float32"))))


@pytest.mark.parametrize("padding", [0.0, -0.0, 3.0])
def test_async_copy_padding_retains_metadata_and_zero_bit_pattern(padding):
    import math

    n = T.dynamic("n", "int64")

    @T.prim_func
    def async_copy(A: T.Buffer((n,), "float32", align=4)):
        with T.sblock("root"):
            scratch = T.alloc_shared((8,), "float32")
            T.annotate_safe_value({A: padding})
            for i in T.serial(8):
                _native.call_intrin(
                    "void",
                    "tl.ptx_cp_async",
                    T.access_ptr(scratch[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    1,
                    annotations={"unit": "elements"},
                )

    module = tvm_tirx.IRModule({"main": async_copy})
    module = tvm_ffi.get_global_func("tl.transform.LegalizeSafeMemoryAccess")()(module)
    calls = []
    stores = []
    tvm_ffi.structural_walk(module["main"].body, (ir.Call, calls.append), order="pre")
    tvm_ffi.structural_walk(module["main"].body, (ir.TensorStore, stores.append), order="pre")
    copy = next(call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tl.ptx_cp_async")
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(copy.args[-1], copy.attrs["annotation_schema"])
    assert metadata["unit"] == "elements"
    assert int(copy.args[2]) == 1
    positive_zero = padding == 0.0 and not math.copysign(1, padding) < 0
    assert int(copy.attrs["operand_count"]) == (4 if positive_zero else 3)
    assert len(stores) == (0 if positive_zero else 1)
    assert tirx.analysis.check_well_formed(module)


def test_source_safe_value_requires_a_real_source_block():
    # The frozen old TVM builder also rejects root declarations: its implicit
    # root is only inserted after source construction has finished.
    with pytest.raises(ValueError, match="outside of any T.sblock"):

        @T.prim_func
        def invalid(A: T.Buffer((4,), "float32", align=4)):
            T.annotate_safe_value({A: -3.5})
            T.evaluate(0)
