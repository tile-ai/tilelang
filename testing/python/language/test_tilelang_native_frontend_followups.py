"""Concrete view, SIMD, and inline contracts found during frontend review."""

import json

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function
from tvm_tirx.script import tirx as N
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.ascend.language as A
import tilelang.language as T
from tilelang.language import _native


@pytest.mark.parametrize("signed_index", [False, True])
@pytest.mark.parametrize("source_dtype", ["uint32x64", "float32x64", "float16x128", "bfloat16x128"])
def test_vselr_reinterprets_values_without_changing_lane_bits(source_dtype, signed_index):
    index_suffix = "16x128" if source_dtype in ("float16x128", "bfloat16x128") else "32x64"
    index_dtype = ("int" if signed_index else "uint") + index_suffix
    source = ir.Var("source", source_dtype)
    index = ir.Var("index", index_dtype)
    result = A.simd.vselr(source, index)
    assert result.ty == ir.PrimType(source_dtype)
    operations = []
    tvm_ffi.structural_walk(result, (ir.Call, operations.append))
    (selection,) = [call for call in operations if call.op.name == "tl.simd.vselr"]
    assert int(selection.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.Opaque)
    assert selection.args[1].ty == ir.PrimType("uint" + index_suffix)
    if index_dtype.startswith("int"):
        assert selection.args[1].op.name == "tirx.reinterpret"
        assert selection.args[1].args[0].same_as(index)
    else:
        assert selection.args[1].same_as(index)
    if source_dtype == "float32x64":
        assert result.op.name == "tirx.reinterpret"
        assert result.args[0].same_as(selection)
        assert selection.ty == ir.PrimType("uint32x64")
        assert selection.args[0].op.name == "tirx.reinterpret"
        assert selection.args[0].args[0].same_as(source)
    else:
        assert selection.same_as(result)
        assert selection.args[0].same_as(source)
    for call in operations:
        call.validate()
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(result)), result, map_free_vars=True)


def test_vselr_reinterpret_nodes_use_the_user_source_location():
    @T.prim_func
    def source(values: T.float32x64, indices: T.int32x64):
        T.evaluate(A.simd.vselr(values, indices))

    calls = []
    tvm_ffi.structural_walk(source, (ir.Call, calls.append))
    reinterprets = [call for call in calls if call.op.name == "tirx.reinterpret"]
    assert len(reinterprets) == 3
    assert all(isinstance(call.loc, ir.SourceLoc) for call in reinterprets)
    assert len({call.loc.start_line for call in reinterprets}) == 1
    assert tirx.analysis.check_well_formed(source)


def test_nested_views_and_explicit_bind_keep_the_original_storage():
    captured = []

    def check_storage(tile, view, nested, bound):
        assert all(_native.tensor_storage_source(value).same_as(tile) for value in [view, nested, bound])
        captured.extend([tile, view, nested, bound])

    @A.prim_func
    def source():
        with T.sblock("metadata_owner"):
            tile = A.alloc_shared((2, 8), "float32")
            view = A.view(tile, shape=(16,))
            nested = A.view(view, shape=(4, 4))
            bound = T.bind(nested)
            with _native.attr(0, "test_view_scope", 1):
                check_storage(tile, view, nested, bound)
                A.annotate_buffer_versions({tile: (2, "counter"), bound: (2, "counter")})
            check_storage(tile, view, nested, bound)

    assert tirx.analysis.check_well_formed(source)
    assert len(captured) == 8
    restored = ir.load_json(ir.save_json(source))
    assert tirx.analysis.check_well_formed(restored)
    assert tvm_ffi.structural_equal(restored, source)


@pytest.mark.parametrize("conflicting_value", [(3, "counter"), (2, "iteration")])
def test_nested_view_version_conflicts_are_rejected(conflicting_value):
    with pytest.raises(ValueError, match="aliases of one buffer storage"):

        @A.prim_func
        def source():
            with T.sblock("metadata_owner"):
                tile = A.alloc_shared((2, 8), "float32")
                nested = A.view(A.view(tile, shape=(16,)), shape=(4, 4))
                A.annotate_buffer_versions({tile: (2, "counter"), nested: conflicting_value})


def test_separate_allocations_with_equal_types_keep_separate_storage():
    @A.prim_func
    def source():
        with T.sblock("metadata_owner"):
            first = A.alloc_shared((16,), "float32")
            second = A.alloc_shared((16,), "float32")
            first_view = A.view(first, shape=(4, 4))
            second_view = A.view(second, shape=(4, 4))
            A.annotate_buffer_versions({first_view: (2, "counter"), second_view: (3, "iteration")})

    assert tirx.analysis.check_well_formed(source)


def test_unknown_pointer_call_does_not_prove_a_view_storage_alias():
    captured = []

    def check_storage(tile, unknown, view):
        assert _native.tensor_storage_source(view).same_as(unknown)
        assert not _native.tensor_storage_source(view).same_as(tile)
        captured.append(True)

    @A.prim_func
    def source():
        with T.sblock("metadata_owner"):
            tile = A.alloc_shared((16,), "float32")
            unknown = T.call_extern("handle", "some_pointer_function", tile.data)
            view = T.Tensor((16,), "float32", data=unknown)
            check_storage(tile, unknown, view)
            A.annotate_buffer_versions({tile: (2, "counter"), view: (3, "iteration")})

    assert captured == [True]
    assert tirx.analysis.check_well_formed(source)


def test_view_alias_query_reads_the_active_native_builder_and_clears_with_it():
    with IRBuilder() as builder, _native.prim_func():
        T.func_name("view_scope")
        tile = A.alloc_shared((16,), "float32")
        view = A.view(tile, shape=(4, 4))
        assert _native.tensor_storage_source(view).same_as(tile)
    assert tirx.analysis.check_well_formed(builder.get())
    assert _native.tensor_storage_source(view).same_as(view)
    with IRBuilder(), _native.prim_func():
        T.func_name("different_builder")
        other = A.alloc_shared((16,), "float32")
        assert _native.tensor_storage_source(view).same_as(view)
        assert not _native.tensor_storage_source(view).same_as(other)


@pytest.mark.parametrize("frontend", ["tilelang", "native"])
def test_inline_meta_method_uses_its_active_parser_and_executes(frontend, tmp_path):
    @T.meta_class
    class Writer:
        def __init__(self, output):
            self.output = output

        @T.inline
        def store(self, value):
            if value > 0:
                self.output[0] = value + 7
            else:
                self.output[0] = -19

    if frontend == "tilelang":

        @T.prim_func
        def source(output: T.Tensor((1,), "int32"), value: T.int32):
            writer = Writer(output)
            writer.store(value)

    else:

        @N.function
        def source(output: N.Tensor((1,), "int32"), value: N.int32):
            writer = Writer(output)
            writer.store(value)

    assert tirx.analysis.check_well_formed(source)
    restored = ir.load_json(ir.save_json(source))
    assert tirx.analysis.check_well_formed(restored)
    executable = compile_packed_function(restored, tilelang_owned=False)
    results = []
    for argument, expected in [(5, 12), (0, -19), (-3, -19)]:
        output = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
        executable(output, argument)
        assert output.numpy()[0] == expected
        results.append({"input": argument, "actual": int(output.numpy()[0]), "expected": expected})
    (tmp_path / "inline-runtime.json").write_text(
        json.dumps({"frontend": frontend, "c_compilations": 1, "entry_calls": 3, "results": results}, indent=2)
    )


def test_native_inline_keeps_defaults_keywords_late_closure_and_nested_calls():
    factor = 2

    @T.meta_class
    class Writer:
        def __init__(self, output):
            self.output = output

        @T.inline
        def leaf(self, value=3, *, offset=1):
            self.output[0] = value * factor + offset

        @T.inline
        def outer(self, value=3):
            self.leaf(value, offset=4)

    def build():
        @N.function
        def source(output: N.Tensor((1,), "int32")):
            writer = Writer(output)
            writer.outer()

        stores = []
        tvm_ffi.structural_walk(source, (ir.TensorStore, stores.append))
        assert len(stores) == 1
        assert tirx.analysis.check_well_formed(source)
        return int(stores[0].value)

    factor = 5
    assert build() == 19
    factor = 7
    assert build() == 25


def test_native_inline_failure_aborts_its_builder_and_signature_errors_emit_nothing():
    observations = []

    def fail_during_emit(index):
        assert isinstance(index, ir.Var)
        raise ValueError("inline body failed")

    @T.meta_class
    class Writer:
        def __init__(self, output):
            self.output = output

        @T.inline
        def fail(self):
            for i in range(2):
                fail_during_emit(i)

        @T.inline
        def store(self, value):
            self.output[0] = value

    # Native statement frames finalize only on normal exit. A method-body
    # exception propagates out of this parse; the next parse owns a new builder.
    with pytest.raises(ValueError, match="inline body failed"):

        @N.function
        def invalid(output: N.Tensor((1,), "int32")):
            writer = Writer(output)
            writer.fail()

    assert not IRBuilder.is_in_scope()

    def guarded(writer):
        builder = IRBuilder.current()
        before = list(builder.frames)
        with pytest.raises(TypeError):
            writer.store()
        after = list(builder.frames)
        assert len(before) == len(after)
        assert all(left.same_as(right) for left, right in zip(before, after))
        observations.append(True)

    @N.function
    def source(output: N.Tensor((1,), "int32")):
        writer = Writer(output)
        guarded(writer)
        writer.store(31)

    assert observations == [True]
    assert not IRBuilder.is_in_scope()
    assert tirx.analysis.check_well_formed(source)
    stores = []
    tvm_ffi.structural_walk(source, (ir.TensorStore, stores.append))
    assert len(stores) == 1 and int(stores[0].value) == 31


def test_view_conflict_inside_nested_frame_does_not_leak_into_next_builder():
    with pytest.raises(ValueError, match="aliases of one buffer storage"):

        @A.prim_func
        def invalid():
            with T.sblock("metadata_owner"):
                tile = A.alloc_shared((16,), "float32")
                with _native.attr(0, "test_view_scope", 1):
                    view = A.view(tile, shape=(4, 4))
                    A.annotate_buffer_versions({tile: 2, view: 3})

    assert not IRBuilder.is_in_scope()

    @A.prim_func
    def valid():
        with T.sblock("metadata_owner"):
            first = A.alloc_shared((16,), "float32")
            second = A.alloc_shared((16,), "float32")
            A.annotate_buffer_versions({first: 2, second: 3})

    assert tirx.analysis.check_well_formed(valid)
