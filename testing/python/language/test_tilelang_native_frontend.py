"""Native IR contracts of the existing TileLang source language."""

import inspect

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function

import tilelang.language as T
import tilelang.cpu.language as C
import tilelang


def nodes(function, kind):
    result = []
    tvm_ffi.structural_walk(function, (kind, lambda node: result.append(node)), order="pre")
    return result


def execute(function, *values):
    kernel = compile_packed_function(function, tilelang_owned=False)
    args = [tvm_tirx.runtime.tensor(value, tvm_tirx.cpu()) if isinstance(value, np.ndarray) else value for value in values]
    kernel(*args)
    return [arg.numpy() if isinstance(value, np.ndarray) else value for arg, value in zip(args, values)]


def test_existing_prim_func_spelling_builds_native_function():
    @T.prim_func
    def add(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        for i in T.serial(8):
            B[i] = A[i] + 1.0

    assert isinstance(add, tirx.Function)
    assert isinstance(add.body, ir.SeqStmt)
    assert all(isinstance(parameter.ty, tirx.TensorType) for parameter in add.params)
    assert not hasattr(add, "buffer_map")
    assert tirx.analysis.check_well_formed(add)
    assert len(nodes(add, ir.TensorStore)) == 1
    values = np.arange(8, dtype="float32")
    _, actual = execute(add, values, np.zeros_like(values))
    np.testing.assert_array_equal(actual, values + 1)


@pytest.mark.parametrize("dialect,threads", [(C, None), (T, 8)])
def test_kernel_contract_binds_grid_and_threads(dialect, threads):
    X = dialect

    @X.prim_func
    def kernel(A: X.Tensor((16,), "float32"), B: X.Tensor((16,), "float32")):
        with X.Kernel(2, **({} if threads is None else {"threads": threads})) as bx:
            for i in X.Parallel(8):
                B[bx * 8 + i] = A[bx * 8 + i] + 1.0

    assert tirx.analysis.check_well_formed(kernel)
    launch = [node for node in nodes(kernel, ir.RegionStmt) if node.op.name == "tl.launch_kernel"]
    assert len(launch) == 1
    launch = launch[0]
    assert len(launch.args) == 5
    assert all(isinstance(value, ir.Tuple) for value in launch.args)
    assert len(launch.body_params) == 4
    assert bool(launch.attrs["threads_was_specified"]) == (threads is not None)


def test_dynamic_shape_keeps_symbol_identity():
    n = T.dynamic("n")

    @T.prim_func
    def add(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        for i in T.serial(n):
            B[i] = A[i] + 3.0

    assert add.params[0].ty.shape[0].same_as(n)
    assert add.params[1].ty.shape[0].same_as(n)
    assert tirx.analysis.check_well_formed(add)
    data = np.arange(13, dtype="float32")
    _, actual = execute(add, data, np.zeros_like(data))
    np.testing.assert_array_equal(actual, data + 3)


def test_scalar_annotations_are_native_primitive_types():
    @T.prim_func
    def scale(A: T.Tensor((4,), "int32"), value: T.int32):
        for i in T.serial(4):
            A[i] = value + i

    assert scale.params[1].ty == ir.PrimType("int32")
    assert tirx.analysis.check_well_formed(scale)
    (actual, _) = execute(scale, np.zeros(4, dtype="int32"), 11)
    np.testing.assert_array_equal(actual, np.arange(11, 15, dtype="int32"))


def test_dynamic_branch_and_cast_execute():
    @T.prim_func
    def sign(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "int32")):
        for i in T.serial(8):
            if A[i] > 0:
                B[i] = T.int32(A[i])
            else:
                B[i] = -1

    values = np.array([-3, -1, 0, 0.5, 1, 2, 5, 10], dtype="float32")
    _, actual = execute(sign, values, np.zeros(8, dtype="int32"))
    np.testing.assert_array_equal(actual, np.where(values > 0, values.astype("int32"), -1))


def test_ast_assertion_context_preserves_body_and_error():
    @T.prim_func
    def checked(A: T.Tensor((1,), "int32")):
        with T.Assert(A[0] >= 0, "negative checked input"):
            A[0] = A[0] + 1

    assert len(nodes(checked, ir.AssertStmt)) == 1
    assert len(nodes(checked, ir.TensorStore)) == 1
    (actual,) = execute(checked, np.array([4], dtype="int32"))
    np.testing.assert_array_equal(actual, [5])
    with pytest.raises(RuntimeError, match="negative checked input"):
        execute(checked, np.array([-1], dtype="int32"))


def test_eager_assertion_emits_native_check_before_store():
    @tilelang.jit
    def checked(A):
        A: T.Tensor((1,), "int32")
        assert A[0] >= 0, "negative eager input"
        A[0] = A[0] + 1

    function = checked.get_tir(np.array([4], dtype="int32"))
    assert len(nodes(function, ir.AssertStmt)) == 1
    assert len(nodes(function, ir.TensorStore)) == 1
    (actual,) = execute(function, np.array([4], dtype="int32"))
    np.testing.assert_array_equal(actual, [5])
    with pytest.raises(RuntimeError, match="negative eager input"):
        execute(function, np.array([-1], dtype="int32"))


def test_macro_preserves_source_and_scalar_semantics():
    @T.macro
    def plus_one(value):
        return value + 1

    @T.prim_func
    def add(A: T.Tensor((8,), "int32")):
        for i in T.serial(8):
            A[i] = plus_one(i)

    (actual,) = execute(add, np.zeros(8, dtype="int32"))
    np.testing.assert_array_equal(actual, np.arange(1, 9, dtype="int32"))
    assert tirx.analysis.check_well_formed(add)


@pytest.mark.parametrize("eager", [False, True])
def test_loop_break_preserves_nested_loop_control(eager):
    if eager:

        @tilelang.jit
        def kernel(A):
            A: T.Tensor((3,), "int32")
            for outer in T.serial(3):
                for inner in T.serial(5):
                    if inner == outer + 1:
                        T.loop_break()
                    A[outer] = A[outer] + 1

        function = kernel.get_tir(np.zeros(3, dtype="int32"))
    else:

        @T.prim_func
        def function(A: T.Tensor((3,), "int32")):
            for outer in T.serial(3):
                for inner in T.serial(5):
                    if inner == outer + 1:
                        T.loop_break()
                    A[outer] = A[outer] + 1

    assert len(nodes(function, ir.Break)) == 1
    assert tirx.analysis.check_well_formed(function)
    assert not [node for node in nodes(function, ir.Call) if isinstance(node.op, ir.Op) and node.op.name == "tl.loop_break"]
    (actual,) = execute(function, np.zeros(3, dtype="int32"))
    np.testing.assert_array_equal(actual, np.array([1, 2, 3], dtype="int32"))


def test_shared_allocation_is_a_typed_binding():
    @T.prim_func
    def local(A: T.Tensor((8,), "float32")):
        temp = T.alloc_shared((8,), "float32")
        for i in T.serial(8):
            temp[i] = A[i]

    allocations = [node for node in nodes(local, ir.Bind) if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor"]
    assert len(allocations) == 1
    assert isinstance(allocations[0].var.ty, tirx.TensorType)
    # alloc_shared has used shared.dyn by default since the source baseline.
    assert str(allocations[0].var.ty.scope) == "shared.dyn"
    assert tirx.analysis.check_well_formed(local)


def test_parallel_nest_keeps_dimension_order():
    @T.prim_func
    def matrix(A: T.Tensor((3, 5), "int32")):
        for i, j in T.Parallel(3, 5):
            A[i, j] = i * 10 + j

    assert tirx.analysis.check_well_formed(matrix)
    loops = nodes(matrix, ir.For)
    assert len(loops) == 2
    # Parallel-to-CPU lowering belongs to the backend pipeline. At the frontend
    # boundary, enumerate the exact indexing/value expression independently.
    store = nodes(matrix, ir.TensorStore)[0]
    analyzer = tvm_tirx.sym.Analyzer()
    for i in range(3):
        for j in range(5):
            replacements = {loops[0].var: ir.const(i), loops[1].var: ir.const(j)}
            value = tvm_ffi.structural_map(store.value, (ir.Var, lambda var, replacements=replacements: replacements.get(var, var)))
            assert int(analyzer.simplify(value)) == i * 10 + j


def test_source_location_retains_statement_line():
    def source(A: T.Tensor((8,), "float32")):
        for i in T.serial(8):
            A[i] = 1.0

    function = T.prim_func(source)
    _, first_line = inspect.getsourcelines(source)
    store = nodes(function, ir.TensorStore)[0]
    assert isinstance(store.loc, ir.SourceLoc)
    assert store.loc.start_line == first_line + 2


def test_compile_time_short_circuit_does_not_evaluate_dead_branch():
    def fail():
        raise AssertionError("dead branch executed while constructing the kernel")

    @T.prim_func
    def short_circuit(A: T.Tensor((1,), "int32")):
        if False and fail():  # noqa: SIM223 - exercise source short-circuit semantics
            A[0] = 0
        else:
            A[0] = 7

    (actual,) = execute(short_circuit, np.zeros(1, dtype="int32"))
    np.testing.assert_array_equal(actual, np.array([7], dtype="int32"))


def test_kernel_context_is_unwound_after_builder_failure():
    from tilelang.language.kernel import KernelLaunchFrame

    with pytest.raises(ValueError, match="frontend sentinel"):

        @T.prim_func
        def failure():
            with T.Kernel(1):
                raise ValueError("frontend sentinel")

    assert KernelLaunchFrame.Current() is None
    assert not T.IRBuilder.is_in_scope()


def test_mutable_local_scalar_preserves_loop_recurrence():
    @T.prim_func
    def recurrence(A: T.Tensor((1,), "int32")):
        value = T.alloc_var("int32", init=1)
        for i in T.serial(5):
            value = value * 2 + i
        A[0] = value

    expected = 1
    for i in range(5):
        expected = expected * 2 + i
    (actual,) = execute(recurrence, np.zeros(1, dtype="int32"))
    np.testing.assert_array_equal(actual, np.array([expected], dtype="int32"))


def test_view_alias_updates_original_storage():
    @T.prim_func
    def alias(A: T.Tensor((2, 4), "int32")):
        flat = T.view(A, shape=(8,))
        for i in T.serial(8):
            flat[i] = i * 3

    assert tirx.analysis.check_well_formed(alias)
    (actual,) = execute(alias, np.zeros((2, 4), dtype="int32"))
    np.testing.assert_array_equal(actual, (np.arange(8, dtype="int32") * 3).reshape(2, 4))


def test_cross_dtype_view_preserves_bits():
    @T.prim_func
    def bits(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "uint32")):
        encoded = T.view(A, dtype="uint32")
        for i in T.serial(4):
            B[i] = encoded[i]

    values = np.array([0.0, -0.0, 1.0, -3.0], dtype="float32")
    _, actual = execute(bits, values, np.zeros(4, dtype="uint32"))
    np.testing.assert_array_equal(actual, values.view("uint32"))


def test_layout_annotation_retains_tensor_and_symbol_identities():
    n = T.dynamic("n")

    @T.prim_func
    def annotated(A: T.Tensor((n,), "float32")):
        with T.sblock("metadata_owner"):
            T.annotate_layout({A: T.Layout((n,), lambda i: i)})
            for i in T.serial(n):
                A[i] = A[i] + 1.0

    annotations = [node for node in nodes(annotated, ir.Call) if node.op.same_as(ir.Op.get("tl.layout_map"))]
    assert len(annotations) == 1
    annotation = annotations[0]
    assert not annotation.args and "annotation_schema" not in annotation.attrs
    assert annotation.attrs["layout_map"][annotated.params[0]].get_input_shape()[0].same_as(n)
    assert tirx.analysis.check_well_formed(annotated)


def test_block_predicate_is_preserved_as_native_if():
    @T.prim_func
    def guarded(A: T.Tensor((8,), "int32")):
        for i in T.serial(8):
            with T.sblock("write"):
                vi = T.axis.spatial(8, i)
                T.where(vi < 4)
                A[vi] = 1

    assert tirx.analysis.check_well_formed(guarded)
    conditions = nodes(guarded, ir.If)
    assert len(conditions) == 1
    assert len(nodes(guarded, ir.Bind)) >= 1


def test_pipeline_metadata_carries_dynamic_layout_payload():
    n = T.dynamic("n")

    @T.prim_func
    def pipeline(A: T.Tensor((n,), "int32")):
        for i in T.Pipelined(n, num_stages=2, annotations={"tile_extent": n}):
            A[i] = i

    assert tirx.analysis.check_well_formed(pipeline)
    wrappers = [node for node in nodes(pipeline, ir.RegionStmt) if node.op.name == "tl.attr"]
    assert len(wrappers) == 1
    decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(wrappers[0].args[0], wrappers[0].attrs["annotation_schema"])
    assert decoded["key"] == "tl.loop.annotations"
    assert decoded["value"]["tile_extent"].same_as(n)


def test_cast_spelling_uses_native_primitive_conversion():
    @T.prim_func
    def narrowed(A: T.Tensor((6,), "uint32"), B: T.Tensor((6,), "uint32")):
        for i in T.serial(6):
            B[i] = T.cast(T.cast(A[i], "uint8"), "uint32")

    values = np.array([0, 255, 256, 257, 65535, 2**32 - 1], dtype="uint32")
    _, actual = execute(narrowed, values, np.zeros_like(values))
    np.testing.assert_array_equal(actual, values.astype("uint8").astype("uint32"))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"round": "bad"}, "Invalid round"),
        ({"round": "rs"}, "rbits is required"),
        ({"rbits": 0}, "rbits is only valid"),
        ({"sat": 1}, "Must be a bool"),
    ],
)
def test_cast_rejects_invalid_policy_before_constructing_ir(kwargs, error):
    with pytest.raises(ValueError, match=error):
        T.cast(ir.const(1.0), "float16", **kwargs)


def test_noncontiguous_source_requires_explicit_view_strides():
    with pytest.raises(ValueError, match="explicit strides"):

        @T.prim_func
        def invalid(A: T.StridedTensor((2, 4), (8, 2), "int32")):
            flat = T.view(A, shape=(8,))
            flat[0] = 1


def test_nonzero_source_offset_is_rejected_by_view():
    with pytest.raises(ValueError, match="non-zero elem_offset"):

        @T.prim_func
        def invalid(A: T.Buffer((8,), "int32", elem_offset=1)):
            flat = T.view(A, shape=(8,))
            flat[0] = 1


def test_jit_lazy_native_specializations_preserve_defaults():
    @tilelang.jit
    def factory(size: int, increment: int = 3):
        @T.prim_func
        def add(A: T.Tensor((size,), "int32")):
            for i in T.serial(size):
                A[i] = A[i] + increment

        return add

    first = factory.get_tir(8)
    assert isinstance(first, tirx.Function)
    assert factory.get_tir(size=8, increment=3) is first
    second = factory.get_tir(8, 5)
    assert not first.same_as(second)
    assert factory.func.parse_args(8)[0] == factory.func.parse_args(size=8, increment=3)[0]
    (actual,) = execute(second, np.arange(8, dtype="int32"))
    np.testing.assert_array_equal(actual, np.arange(8, dtype="int32") + 5)


def test_jit_eager_typed_template_preserves_output_order_and_constexpr():
    @tilelang.jit
    def add(A, increment: int = 4):
        n = T.const("n")
        A: T.Tensor((n,), "int32")
        B = T.empty((n,), dtype="int32")
        C = T.empty((n,), dtype="int32")
        for i in T.serial(n):
            B[i] = A[i] + increment
            C[i] = A[i] - increment
        return C, B

    values = np.arange(7, dtype="int32")
    function = add.get_tir(values)
    assert isinstance(function, tirx.Function)
    assert list(function.attrs["tilelang_out_idx"]) == [-1, -2]
    assert all(int(parameter.ty.shape[0]) == 7 for parameter in function.params)
    assert tirx.analysis.check_well_formed(function)
    _, plus, minus = execute(function, values, np.zeros_like(values), np.zeros_like(values))
    np.testing.assert_array_equal(plus, values + 4)
    np.testing.assert_array_equal(minus, values - 4)
    other = add.get_tir(np.zeros(11, dtype="int32"))
    assert all(int(parameter.ty.shape[0]) == 11 for parameter in other.params)
    assert all(int(parameter.ty.shape[0]) == 7 for parameter in function.params)


def test_body_annotations_keep_signature_order_and_dynamic_shape():
    @T.prim_func
    def add(A, B, increment: T.int32):
        n = T.dynamic("n")
        B: T.Tensor((n,), "int32")
        A: T.Tensor((n,), "int32")
        for i in T.serial(n):
            B[i] = A[i] + increment

    assert [parameter.name for parameter in add.params] == ["A", "B", "increment"]
    assert all(isinstance(parameter.ty, tirx.TensorType) for parameter in add.params[:2])
    assert tirx.analysis.check_well_formed(add)
    values = np.arange(9, dtype="int32")
    _, actual, _ = execute(add, values, np.zeros_like(values), 7)
    np.testing.assert_array_equal(actual, values + 7)


def test_body_pointer_annotation_matches_a_native_tensor_parameter():
    @T.prim_func
    def pointer_copy(A, B: T.Tensor((8,), "int32")):
        A: T.ptr("int32")
        view = T.Tensor.from_ptr(A, (8,), "int32")
        for i in T.serial(8):
            B[i] = view[i]

    # Tensor.from_ptr calls match_buffer: the final match determines the
    # native TensorType parameter ABI, including for an initially typed pointer.
    assert isinstance(pointer_copy.params[0].ty, tirx.TensorType)
    assert pointer_copy.params[0].ty.dtype == "int32"
    assert list(map(int, pointer_copy.params[0].ty.shape)) == [8]
    assert [parameter.name for parameter in pointer_copy.params] == ["A", "B"]
    assert all(load.source.same_as(pointer_copy.params[0]) for load in nodes(pointer_copy, ir.TensorLoad))
    assert tirx.analysis.check_well_formed(pointer_copy)


def test_unmatched_body_pointer_annotation_keeps_native_pointer_type():
    @T.prim_func
    def pointer_use(A):
        A: T.ptr("int32")
        T.evaluate(A)

    assert len(pointer_use.params) == 1
    assert pointer_use.params[0].ty == ir.PtrType("int32", "global")
    assert pointer_use.body[0].value.same_as(pointer_use.params[0])
    assert tirx.analysis.check_well_formed(pointer_use)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "metal", "webgpu", "ascend"])
def test_backend_language_kernel_constructs_native_regions(backend):
    from importlib import import_module

    X = import_module(f"tilelang.{backend}.language")
    options = {} if backend in {"cpu", "ascend"} else {"threads": 8}

    @X.prim_func
    def kernel(A: X.Tensor((16,), "float32"), B: X.Tensor((16,), "float32")):
        with X.Kernel(2, **options) as bx:
            for i in X.Parallel(8):
                B[bx * 8 + i] = A[bx * 8 + i]

    assert isinstance(kernel, tirx.Function)
    assert tirx.analysis.check_well_formed(kernel)
    launches = [region for region in nodes(kernel, ir.RegionStmt) if region.op.name == "tl.launch_kernel"]
    assert len(launches) == 1
    assert len(launches[0].body_params) == 4
    assert list(launches[0].attrs["grid_tags"]) == ["blockIdx.x"]


def test_ascend_execution_scopes_preserve_bindings_and_cleanup():
    import tilelang.ascend.language as A
    from tilelang.ascend.language.frame import inside_simdvf
    from tilelang.ascend.language.kernel import _get_current_simtvf

    @A.prim_func
    def kernel(output: A.Tensor((32,), "int32")):
        with A.Kernel(1):
            with A.Cube():
                output[0] = 1
            with A.Vector() as sid:
                with A.SimtVF(threads=32):
                    tx = A.get_thread_binding()
                    output[tx] = tx + sid
                with A.SimdVF():
                    output[sid] = 2

    assert tirx.analysis.check_well_formed(kernel)
    regions = nodes(kernel, ir.RegionStmt)
    tags = [str(region.args[0].value) for region in regions if region.op.name == "tirx.launch_thread"]
    assert tags == ["cthread", "threadIdx.x", "threadIdx.y", "threadIdx.z"]
    assert not inside_simdvf()
    assert _get_current_simtvf() is None


def test_ascend_execution_scope_unwinds_after_exception():
    import tilelang.ascend.language as A
    from tilelang.ascend.language.frame import inside_simdvf
    from tilelang.ascend.language.kernel import _get_current_simtvf

    with pytest.raises(ValueError, match="scope sentinel"):

        @A.prim_func
        def failure():
            with A.Kernel(1), A.SimtVF(threads=32), A.SimdVF():
                raise ValueError("scope sentinel")

    assert not inside_simdvf()
    assert _get_current_simtvf() is None
    assert not T.IRBuilder.is_in_scope()


def test_resource_annotations_retain_tensor_variables():
    @T.prim_func
    def metadata(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.sblock("metadata_owner"):
            T.annotate_restrict_buffers(A, B)
            first = T.alloc_barrier([32, 64])
            second = T.alloc_cluster_barrier(8)
            with T.annotate_ws_pipeline_depth({first: 2, second: 3}):
                B[0] = A[0]

    decode = tvm_ffi.get_global_func("tl.DecodeAnnotations")
    annotations = [node for node in nodes(metadata, ir.Call) if node.op.name == "tl.annotation"]
    decoded = [decode(node.args[0], node.attrs["annotation_schema"]) for node in annotations]
    assert all(tirx.is_tensor_var(value) for value in decoded[0]["tl.non_restrict_params"])
    barrier_maps = [mapping["barrier_init"] for mapping in decoded if "barrier_init" in mapping]
    assert len(barrier_maps) == 2
    first, second = [next(iter(mapping)) for mapping in barrier_maps]
    assert tirx.is_tensor_var(first) and tirx.is_tensor_var(second)
    assert not first.same_as(second)
    assert list(map(int, barrier_maps[0][first])) == [32, 64]
    assert list(map(int, barrier_maps[1][second])) == [8]
    depth_scopes = [
        decode(region.args[0], region.attrs["annotation_schema"])
        for region in nodes(metadata, ir.RegionStmt)
        if region.op.name == "tl.attr" and decode(region.args[0], region.attrs["annotation_schema"])["key"] == "tl.ws_pipeline_depth"
    ]
    assert len(depth_scopes) == 1
    depths = depth_scopes[0]
    assert depths["key"] == "tl.ws_pipeline_depth"
    assert int(depths["node"][first]) == 2
    assert int(depths["node"][second]) == 3
    assert tirx.analysis.check_well_formed(metadata)


def test_ascend_version_metadata_keeps_tensor_keys():
    import tilelang.ascend.language as A

    @A.prim_func
    def metadata():
        with T.sblock("metadata_owner"):
            tile = A.alloc_shared((2, 8), "float32")
            view = A.view(tile, shape=(16,))
            A.annotate_buffer_versions({tile: (2, "counter"), view: (2, "counter")})
            A.annotate_manual_multi_buffer(tile, {view: 2})

    decode = tvm_ffi.get_global_func("tl.DecodeAnnotations")
    payloads = [
        decode(node.args[0], node.attrs["annotation_schema"]) for node in nodes(metadata, ir.Call) if node.op.name == "tl.annotation"
    ]
    for payload in payloads:
        for key in ("tl.buffer_versions_map", "tl.buffer_version_mode", "tl.manual_multi_buffer"):
            if key in payload:
                assert len(payload[key]) == 2
                assert all(tirx.is_tensor_var(value) for value in payload[key])
    assert tirx.analysis.check_well_formed(metadata)


@pytest.mark.parametrize("second", [(3, "counter"), (2, "iteration")])
def test_ascend_conflicting_view_version_metadata_is_rejected(second):
    import tilelang.ascend.language as A

    with pytest.raises(ValueError, match="aliases of one buffer storage"):

        @A.prim_func
        def metadata():
            with T.sblock("metadata_owner"):
                tile = A.alloc_shared((2, 8), "float32")
                view = A.view(tile, shape=(16,))
                A.annotate_buffer_versions({tile: (2, "counter"), view: second})
