"""Native Carver templates and graph transforms preserve executable meaning."""

import itertools

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import carver
from tilelang.carver._native_analysis import NativeSchedule, specialize_shapes
from tilelang.carver._workload import WorkloadBuilder
from tilelang.carver.matmul_analysis import get_index_map, normalize_to_matmul
from tilelang.carver.roller import PrimFuncNode
from tilelang.carver.template.conv import ConvTemplate


def compile_cpu(function, allocate_output=True):
    tirx.analysis.check_well_formed(function)
    target = Target("c")
    return tilelang.compile(
        function, out_idx=[-1] if allocate_output else None, target=target, target_host=target, execution_backend="tvm_ffi"
    )


def values(shape, dtype=torch.int32):
    return ((torch.arange(int(np.prod(shape)), dtype=torch.int32) % 7) - 3).reshape(shape).to(dtype)


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 1, 3)])
def test_elementwise_template_cpu(shape):
    function = carver.ElementwiseTemplate(shape=list(shape), dtype="int32").equivalent_function()
    assert isinstance(function, tirx.Function)
    assert not hasattr(function, "buffer_map")
    assert isinstance(function.body, ir.SeqStmt)
    inp = values(shape)
    if shape:
        output = compile_cpu(function)(inp)
    else:
        # The existing JIT adapter intentionally rejects automatic rank-zero
        # output allocation. The native function accepts an explicit scalar.
        output = torch.empty((), dtype=inp.dtype)
        compile_cpu(function, allocate_output=False)(inp, output)
    torch.testing.assert_close(output, inp + 1, atol=0, rtol=0)


@pytest.mark.parametrize("structure,shape", [("SS", (2, 3)), ("SR", (2, 3)), ("RS", (3, 2)), ("RR", (2, 3)), ("SRS", (2, 3, 4))])
def test_general_reduction_template_cpu(structure, shape):
    function = carver.GeneralReductionTemplate(structure=structure, shape=list(shape), dtype="int32").equivalent_function()
    inp = values(shape)
    axes = tuple(index for index, kind in enumerate(structure) if kind == "R")
    expected = inp.sum(axes, dtype=torch.int32) if axes else inp
    if expected.ndim:
        output = compile_cpu(function)(inp)
    else:
        output = torch.empty((), dtype=inp.dtype)
        compile_cpu(function, allocate_output=False)(inp, output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("trans_a,trans_b,bias", list(itertools.product([False, True], repeat=3)))
def test_matmul_bias_cast_cpu(trans_a, trans_b, bias):
    template = carver.MatmulTemplate(
        3, 4, 5, trans_A=trans_a, trans_B=trans_b, with_bias=bias, in_dtype="int8", accum_dtype="int32", out_dtype="float32"
    )
    function = template.equivalent_function()
    a = values((5, 3) if trans_a else (3, 5), torch.int8)
    b = values((4, 5) if trans_b else (5, 4), torch.int8)
    args = [a, b]
    expected = (a.t() if trans_a else a).to(torch.int32) @ (b.t() if trans_b else b).to(torch.int32)
    if bias:
        offset = values((4,))
        args.append(offset)
        expected = expected + offset
    torch.testing.assert_close(compile_cpu(function)(*args), expected.to(torch.float32), atol=0, rtol=0)


def test_gemv_bias_cast_cpu():
    function = carver.GEMVTemplate(4, 5, with_bias=True, in_dtype="int8", accum_dtype="int32", out_dtype="float32").equivalent_function()
    a, b, offset = values((1, 5), torch.int8), values((4, 5), torch.int8), values((4,))
    expected = a.to(torch.int32) @ b.to(torch.int32).t() + offset
    torch.testing.assert_close(compile_cpu(function)(a, b, offset), expected.to(torch.float32), atol=0, rtol=0)


@pytest.mark.parametrize("stride,dilation", [(1, 1), (2, 1), (1, 2)])
def test_convolution_padding_cpu(stride, dilation):
    template = ConvTemplate(
        N=1,
        C=2,
        H=4,
        W=5,
        F=3,
        K=3,
        S=stride,
        D=dilation,
        P=1,
        with_bias=True,
        in_dtype="float32",
        accum_dtype="float32",
        out_dtype="float32",
    )
    a, b, offset = values((1, 4, 5, 2), torch.float32), values((3, 3, 2, 3), torch.float32), values((3,), torch.float32)
    expected = torch.nn.functional.conv2d(
        a.permute(0, 3, 1, 2), b.permute(3, 2, 0, 1), offset, stride=stride, padding=1, dilation=dilation
    ).permute(0, 2, 3, 1)
    torch.testing.assert_close(compile_cpu(template.equivalent_function())(a, b, offset), expected, atol=0, rtol=0)


@pytest.mark.parametrize("trans_a,trans_b", list(itertools.product([False, True], repeat=2)))
def test_native_matmul_reindex_layout_cpu(trans_a, trans_b):
    function = carver.MatmulTemplate(
        3, 4, 5, trans_A=trans_a, trans_B=trans_b, in_dtype="int32", accum_dtype="int32", out_dtype="int32"
    ).equivalent_function()
    schedule = NativeSchedule(function)
    main = schedule.get_child_blocks(schedule.root)[0]
    assert normalize_to_matmul(schedule, main, ["a", "a", "a"]) is schedule
    assert not tvm_ffi.structural_equal(function, schedule.function)
    for before, after in zip(function.params, schedule.function.params):
        assert before.same_as(after), "Public parameter identity/type is an unchanged ABI contract"
    stores = []
    tvm_ffi.structural_walk(schedule.function, (ir.TensorStore, stores.append))
    assert any(len(store.dest.ty.shape) == 3 for store in stores)
    a = values((5, 3) if trans_a else (3, 5))
    b = values((4, 5) if trans_b else (5, 4))
    expected = (a.t() if trans_a else a) @ (b.t() if trans_b else b)
    torch.testing.assert_close(compile_cpu(schedule.function)(a, b), expected, atol=0, rtol=0)
    torch.testing.assert_close(compile_cpu(function)(a, b), expected, atol=0, rtol=0)


def test_convolution_reindex_padding_cpu():
    function = ConvTemplate(
        N=1, C=2, H=4, W=5, F=3, K=3, S=1, D=1, P=1, in_dtype="float32", accum_dtype="float32", out_dtype="float32"
    ).equivalent_function()
    schedule = NativeSchedule(function)
    main = schedule.get_child_blocks(schedule.root)[0]
    assert get_index_map(schedule.get(main)) is not None
    normalize_to_matmul(schedule, main, ["a", "a", "a"])
    a, b = values((1, 4, 5, 2), torch.float32), values((3, 3, 2, 3), torch.float32)
    expected = torch.nn.functional.conv2d(a.permute(0, 3, 1, 2), b.permute(3, 2, 0, 1), padding=1).permute(0, 2, 3, 1)
    torch.testing.assert_close(compile_cpu(schedule.function)(a, b), expected, atol=0, rtol=0)


def test_compute_inline_transpose_identity_and_stale_handles():
    builder = WorkloadBuilder()
    a = builder.input((2, 3), name="data", dtype="int32")
    b = builder.compute((3, 2), lambda i, j: a[j, i] + 1, name="stage")
    c = builder.compute((3, 2), lambda i, j: b[i, j] * 2, name="stage")
    original = builder.to_function([a, c])
    schedule = NativeSchedule(original)
    producer, consumer = schedule.get_child_blocks(schedule.root)
    schedule.compute_inline(producer)
    with pytest.raises(ValueError, match="stale"):
        schedule.get(consumer)
    assert len(schedule.get_child_blocks(schedule.root)) == 1
    schedule.get(schedule.refresh(consumer))
    with pytest.raises(ValueError, match="deleted"):
        schedule.refresh(producer)
    inp = values((2, 3))
    expected = (inp.t() + 1) * 2
    torch.testing.assert_close(compile_cpu(schedule.function)(inp), expected, atol=0, rtol=0)
    torch.testing.assert_close(compile_cpu(original)(inp), expected, atol=0, rtol=0)


def test_reverse_inline_preserves_reduction_accumulator():
    builder = WorkloadBuilder()
    a = builder.input((2, 3), name="A", dtype="float32")
    r = builder.reduce_axis((0, 3), "r")
    b = builder.compute((2,), lambda i: builder.sum(a[i, r], r), name="sum")
    c = builder.compute((2,), lambda i: b[i], name="copy")
    original = builder.to_function([a, c])
    schedule = NativeSchedule(original)
    schedule.reverse_compute_inline(schedule.get_child_blocks(schedule.root)[1])
    inp = values((2, 3), torch.float32)
    torch.testing.assert_close(compile_cpu(schedule.function)(inp), inp.sum(1), atol=0, rtol=0)


def test_reverse_inline_cast_rejected_atomically():
    builder = WorkloadBuilder()
    a = builder.input((1, 3), name="A", dtype="float32")
    r = builder.reduce_axis((0, 3), "r")
    b = builder.compute((1,), lambda i: builder.sum(a[i, r], r), name="sum")
    c = builder.compute((1,), lambda i: b[i].astype("float16"), name="cast")
    original = builder.to_function([a, c])
    schedule = NativeSchedule(original)
    with pytest.raises(ValueError, match="reduction epilogue"):
        schedule.reverse_compute_inline(schedule.get_child_blocks(schedule.root)[1])
    assert schedule.function.same_as(original)
    inp = torch.tensor([[2048.0, 1.0, -2048.0]])
    torch.testing.assert_close(compile_cpu(schedule.function)(inp), torch.tensor([1.0], dtype=torch.float16), atol=0, rtol=0)


def test_layout_failure_transaction_restores_function_and_handles():
    function = carver.MatmulTemplate(3, 4, 5, in_dtype="int32", accum_dtype="int32", out_dtype="int32").equivalent_function()
    schedule = NativeSchedule(function)
    main = schedule.get_child_blocks(schedule.root)[0]
    with pytest.raises(RuntimeError, match="iterations do not traverse full iter space"), schedule.transaction():
        block = schedule.reindex(main, ("read", 0))
        lossy = tirx.IndexMap.from_func(lambda i, k: (i, k % 2), index_dtype="int64")
        schedule.transform_layout(block, ("write", 0), lossy)
    assert schedule.function.same_as(function)
    assert schedule.revision == 0
    assert schedule.get(main).name_hint == "C"
    assert len(schedule.get_child_blocks(schedule.root)) == 1


def test_symbolic_specialization_updates_types_and_uses():
    builder = WorkloadBuilder()
    n = ir.Var("n", ir.PrimType("int64"))
    a = builder.input((2 * n,), name="A", dtype="int32")
    c = builder.compute((2 * n,), lambda i: a[i] + 1, name="C")
    original = builder.to_function([n, a, c])
    specialized = specialize_shapes(original, {n: ir.const(3, "int64")})
    assert len(specialized.params) == 2
    assert not specialized.params[0].same_as(a)
    assert int(specialized.params[0].ty.shape[0]) == 6
    assert original.params[1].same_as(a)
    tensors = []
    tvm_ffi.structural_walk(specialized.body, (ir.TensorLoad, lambda load: tensors.append(load.source)))
    assert all(tensor.same_as(specialized.params[0]) for tensor in tensors)
    inp = values((6,))
    torch.testing.assert_close(compile_cpu(specialized)(inp), inp + 1, atol=0, rtol=0)
    node = PrimFuncNode(original.with_attr("opt_shapes", {"n": 3}))
    assert node.get_space_dim() == [6]


def test_native_dsl_block_init_predicate_and_scope():
    @T.prim_func
    def function(A: T.Tensor((2, 3), "int32"), C: T.Tensor((2,), "int32")):
        T.func_attr({"tirx.noalias": True})
        for i, k in T.grid(2, 3):
            with T.sblock("sum"):
                vi = T.axis.spatial(2, i)
                vk = T.axis.reduce(3, k)
                T.reads(A[vi, vk])
                T.writes(C[vi])
                T.where(vi < 1)
                with T.init():
                    tmp = T.alloc_local((1,), "int32")
                    tmp[0] = 7
                    C[vi] = tmp[0]
                C[vi] = C[vi] + A[vi, vk]

    target = Target("c")
    kernel = tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")
    inp = values((2, 3))
    out = torch.full((2,), -100, dtype=torch.int32)
    kernel(inp, out)
    torch.testing.assert_close(out, torch.tensor([7 + int(inp[0].sum()), -100], dtype=torch.int32), atol=0, rtol=0)


def test_unknown_finalizer_metadata_still_fails():
    from tilelang.carver._workload import annotation

    function = tirx.Function([], ir.SeqStmt([annotation({"tl.unhandled_carver_contract": 1})]), ir.NoneType())
    with pytest.raises(ValueError, match="Unconsumed TileLang annotation"):
        tilelang.transform.FinalizeAnnotations("")(ir.IRModule({"main": function}))


def test_hint_shapes_are_real_argument_shapes():
    function = carver.MatmulTemplate(32, 48, 16, in_dtype="float16", accum_dtype="float32", out_dtype="float32").equivalent_function()
    node = PrimFuncNode(function)
    assert node.input_param_ordinals == [0, 1]
    assert node.output_param_ordinals == [2]
    assert node.propagate_inputs([4, 8], {"k": 3}) == [[4, 3], [8, 3]]
    assert [tensor.name for tensor in node.input_buffers] == ["A", "B"]


def test_fixed_stride_offset_input_reindex_cpu():
    from tvm_tirx.tirx.layout import S

    source = ir.Var("A", tirx.TensorType((2, 3), "int32", align=4, layout=tirx.TileLayout(S[(2, 3) : (8, 2)])))
    builder = WorkloadBuilder()
    output = builder.compute((2, 3), lambda i, j: source[i, j] + 9, name="C")
    original = builder.to_function([source, output])
    schedule = NativeSchedule(original)
    main = schedule.get_child_blocks(schedule.root)[0]
    cache_stage = schedule.reindex(main, ("read", 0))
    schedule.transform_layout(cache_stage, ("write", 0), tirx.IndexMap.from_func(lambda i, j: (j, i), index_dtype="int64"))
    base = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8)
    inp = base[1:3, 1:7:2]
    assert inp.storage_offset() != 0 and not inp.is_contiguous()
    torch.testing.assert_close(compile_cpu(schedule.function)(inp), inp + 9, atol=0, rtol=0)


def test_constant_index_retains_nonsingleton_tensor_dimension():
    builder = WorkloadBuilder()
    source = builder.input((5, 4), name="A", dtype="int32")
    output = builder.compute((2, 4), lambda i, j: source[0, j], name="C")
    node = PrimFuncNode(builder.to_function([source, output]))
    assert node.propagate_inputs([2, 3]) == [[1, 3]]


def test_transform_does_not_reorder_shared_loop_stages():
    @T.prim_func
    def function(A: T.Tensor((4,), "int32"), C: T.Tensor((4,), "int32")):
        T.func_attr({"tirx.noalias": True})
        cache = T.alloc_local((1,), "int32")
        for i in T.serial(4):
            with T.sblock("first"):
                vi = T.axis.spatial(4, i)
                cache[0] = A[vi] + 1
            with T.sblock("second"):
                vi = T.axis.spatial(4, i)
                C[vi] = cache[0] * 2

    schedule = NativeSchedule(function)
    assert schedule.normalize() is None
    with pytest.raises(ValueError, match="complete, directly bound"):
        schedule.compute_inline(schedule.get_child_blocks(schedule.root)[0])
    assert schedule.function.same_as(function)
    inp = values((4,))
    torch.testing.assert_close(compile_cpu(function)(inp), (inp + 1) * 2, atol=0, rtol=0)


def test_non_noalias_function_cannot_duplicate_reads():
    builder = WorkloadBuilder()
    source = builder.input((3,), name="A", dtype="int32")
    cache = builder.compute((3,), lambda i: source[i] + 1, name="cache")
    output = builder.compute((3,), lambda i: cache[i] * 2, name="C")
    function = builder.to_function([source, output]).with_attr("tirx.noalias", False)
    schedule = NativeSchedule(function)
    with pytest.raises(ValueError, match="explicit tirx.noalias"):
        schedule.compute_inline(schedule.get_child_blocks(schedule.root)[0])
    assert schedule.function.same_as(function)


@pytest.mark.parametrize(
    "dtype,transpose,matrix", [("float16", False, "A"), ("float16", True, "B"), ("int8", False, "A"), ("int8", True, "B")]
)
def test_propagation_maps_use_tilelang_device_layout_oracle(dtype, transpose, matrix):
    from tilelang.carver.matmul_analysis import get_propagate_map
    from tilelang.cuda.intrinsics.layout import mma_layout

    forward, inverse = get_propagate_map(transpose, dtype, matrix)
    columns = 16 if dtype == "float16" else 32
    oracle = (
        (mma_layout.ldmatrix_trans_32x8_to_shared_16x16_layout if transpose else mma_layout.ldmatrix_32x8_to_shared_16x16_layout)
        if dtype == "float16"
        else (mma_layout.ldmatrix_32x16_to_shared_16x32_layout_a if matrix == "A" else mma_layout.ldmatrix_32x16_to_shared_16x32_layout_b)
    )
    for row, col in itertools.product(range(16), range(columns)):
        actual = tuple(int(value) for value in forward.map_indices([row, col]))
        expected = oracle(row * 2 + col // (columns // 2), col % (columns // 2))
        assert actual == expected
        assert tuple(int(value) for value in inverse.map_indices(actual)) == (row, col)


def test_auto_inline_chain_updates_actual_ir():
    from tilelang.carver.matmul_analysis import auto_inline_producers

    builder = WorkloadBuilder()
    source = builder.input((3,), name="A", dtype="int32")
    first = builder.compute((3,), lambda i: source[i] + 1, name="first")
    second = builder.compute((3,), lambda i: first[i] * 2, name="second")
    output = builder.compute((3,), lambda i: second[i] - 3, name="last")
    original = builder.to_function([source, output])
    schedule = NativeSchedule(original)
    main = schedule.get_child_blocks(schedule.root)[-1]
    auto_inline_producers(schedule, main)
    assert len(schedule.get_child_blocks(schedule.root)) == 1
    inp = values((3,))
    torch.testing.assert_close(compile_cpu(schedule.function)(inp), (inp + 1) * 2 - 3, atol=0, rtol=0)


def test_inline_preserves_snapshot_when_consumer_writes_source():
    @T.prim_func
    def function(A: T.Tensor((4,), "int32")):
        T.func_attr({"tirx.noalias": True})
        cache = T.alloc_local((4,), "int32")
        for i in T.serial(4):
            with T.sblock("snapshot"):
                vi = T.axis.spatial(4, i)
                cache[vi] = A[vi]
        for i in T.serial(3):
            with T.sblock("update"):
                vi = T.axis.spatial(3, i)
                A[vi + 1] = cache[vi] + 1

    schedule = NativeSchedule(function)
    with pytest.raises(ValueError, match="consumer.*write"):
        schedule.compute_inline(schedule.get_child_blocks(schedule.root)[0])
    assert schedule.function.same_as(function)
    inp = torch.tensor([1, 10, 100, 1000], dtype=torch.int32)
    compile_cpu(function, allocate_output=False)(inp)
    torch.testing.assert_close(inp, torch.tensor([1, 2, 11, 101], dtype=torch.int32), atol=0, rtol=0)


def test_reverse_inline_preserves_intervening_output_read():
    @T.prim_func
    def function(A: T.Tensor((3,), "int32"), C: T.Tensor((3,), "int32"), D: T.Tensor((3,), "int32")):
        T.func_attr({"tirx.noalias": True})
        cache = T.alloc_local((3,), "int32")
        for i in T.serial(3):
            with T.sblock("producer"):
                vi = T.axis.spatial(3, i)
                cache[vi] = A[vi] + 1
        for i in T.serial(3):
            with T.sblock("observe"):
                vi = T.axis.spatial(3, i)
                D[vi] = C[vi] + 5
        for i in T.serial(3):
            with T.sblock("consumer"):
                vi = T.axis.spatial(3, i)
                C[vi] = cache[vi] * 2

    schedule = NativeSchedule(function)
    with pytest.raises(ValueError, match="intervening.*access"):
        schedule.reverse_compute_inline(schedule.get_child_blocks(schedule.root)[-1])
    assert schedule.function.same_as(function)
    inp, output, observer = values((3,)), torch.full((3,), 41, dtype=torch.int32), torch.empty(3, dtype=torch.int32)
    compile_cpu(function, allocate_output=False)(inp, output, observer)
    torch.testing.assert_close(output, (inp + 1) * 2, atol=0, rtol=0)
    torch.testing.assert_close(observer, torch.full((3,), 46, dtype=torch.int32), atol=0, rtol=0)


def test_reindex_preserves_spatial_output_predicate():
    @T.prim_func
    def function(A: T.Tensor((4,), "int32"), C: T.Tensor((4,), "int32")):
        T.func_attr({"tirx.noalias": True})
        for i in T.serial(4):
            with T.sblock("partial"):
                vi = T.axis.spatial(4, i)
                T.where(vi < 2)
                C[vi] = A[vi] + 9

    schedule = NativeSchedule(function)
    schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("write", 0))
    inp = values((4,))
    expected = torch.tensor([int(inp[0]) + 9, int(inp[1]) + 9, -10, -10], dtype=torch.int32)
    for candidate in (function, schedule.function):
        output = torch.full((4,), -10, dtype=torch.int32)
        compile_cpu(candidate, allocate_output=False)(inp, output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_reindex_preserves_sequential_inplace_reads():
    @T.prim_func
    def function(A: T.Tensor((4,), "int32")):
        T.func_attr({"tirx.noalias": True})
        for i in T.serial(3):
            with T.sblock("scan"):
                vi = T.axis.spatial(3, i)
                A[vi + 1] = A[vi] + 1

    schedule = NativeSchedule(function)
    with pytest.raises(ValueError, match="written by the same stage"):
        schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("read", 0))
    assert schedule.function.same_as(function)
    inp = torch.tensor([1, 10, 100, 1000], dtype=torch.int32)
    compile_cpu(function, allocate_output=False)(inp)
    torch.testing.assert_close(inp, torch.tensor([1, 2, 3, 4], dtype=torch.int32), atol=0, rtol=0)


def test_cyclic_storage_graph_does_not_recurse_forever():
    from tilelang.carver.matmul_analysis import auto_inline_producers

    @T.prim_func
    def function(A: T.Tensor((3,), "int32")):
        T.func_attr({"tirx.noalias": True})
        for i in T.serial(3):
            with T.sblock("first"):
                vi = T.axis.spatial(3, i)
                A[vi] = A[vi] + 1
        for i in T.serial(3):
            with T.sblock("second"):
                vi = T.axis.spatial(3, i)
                A[vi] = A[vi] * 2

    schedule = NativeSchedule(function)
    main = schedule.get_child_blocks(schedule.root)[-1]
    auto_inline_producers(schedule, main)
    assert schedule.function.same_as(function)
    assert schedule.get(main).name_hint == "second"


@pytest.mark.parametrize("input_dtype,output_dtype", [("float16", "float16"), ("int8", "int32")])
def test_attention_template_pair_cpu(input_dtype, output_dtype):
    # The existing FlashAttentionTemplate models the QK and SV matrix products
    # for hint generation; it is not a softmax/causal-attention implementation.
    accumulation_dtype = "float32" if input_dtype == "float16" else "int32"
    template = carver.FlashAttentionTemplate(
        batch_size=1,
        num_heads=2,
        seq_length=16,
        seq_kv_length=32,
        head_dim=32,
        in_dtype=input_dtype,
        accum_dtype=accumulation_dtype,
        out_dtype=output_dtype,
    )
    functions = template.equivalent_function()
    assert len(functions) == 2
    for function in functions:
        a, b = [values(tuple(map(int, tensor.ty.shape)), getattr(torch, input_dtype)) for tensor in function.params[:2]]
        expected = (a.to(getattr(torch, accumulation_dtype)) @ b.to(getattr(torch, accumulation_dtype)).transpose(-1, -2)).to(
            getattr(torch, output_dtype)
        )
        torch.testing.assert_close(compile_cpu(function)(a, b), expected, atol=0, rtol=0)


def test_shape_specialization_rejects_ambiguous_variable_names():
    builder = WorkloadBuilder()
    n = ir.Var("n", ir.PrimType("int64"))
    other_n = ir.Var("n", ir.PrimType("int64"))
    source = builder.input((n, other_n), name="A", dtype="int32")
    output = builder.compute((n, other_n), lambda i, j: source[i, j] + 1, name="C")
    function = builder.to_function([n, other_n, source, output])
    with pytest.raises(ValueError, match="Ambiguous.*shape variable"):
        PrimFuncNode(function.with_attr("opt_shapes", {"n": 2}))
    tirx.analysis.check_well_formed(function)


@pytest.mark.parametrize("two_nodes", [False, True])
@tilelang.testing.requires_cuda
def test_tensorcore_policy_native_function_graph(two_nodes):
    # Model the existing SM80 hint policy independently of the device execution
    # tests. The current device supplies its resource limits; no SM80 kernel is
    # compiled or claimed to have executed by this analysis test.
    from tilelang.carver.arch import get_arch
    from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags
    from tilelang.carver.roller import Edge, OutputNode

    arch = get_arch(Target({"kind": "cuda", "arch": "sm_80"}))
    original = carver.MatmulTemplate(128, 128, 128, in_dtype="float16", accum_dtype="float16", out_dtype="float16").equivalent_function()
    function, tags = get_tensorized_func_and_tags(original, arch.target)
    assert tags["pipeline_stage"] == 2
    assert tags["use_async_copy"]
    assert tags["intrin_info"]["in_dtype"] == "float16"
    tirx.analysis.check_well_formed(function)
    first = PrimFuncNode(function, tags=tags, name="first")
    if two_nodes:
        second = PrimFuncNode(function, tags=tags, name="second")
        edge = Edge(first, second, 0, 0)
        first._out_edges.append(edge)
        second.set_inputs(0, edge)
        outputs = [OutputNode(second)]
    else:
        outputs = [OutputNode(first)]
    policy = carver.TensorCorePolicy.from_output_nodes(outputs, arch=arch, tags=tags)
    hints = policy.emit_config(topk=4)
    assert 0 < len(hints) <= 4
    for hint in hints:
        if two_nodes:
            assert set(hint) == {first, second}
            assert all(node_hint.use_tc for node_hint in hint.values())
        else:
            assert hint.use_tc


def test_parallel_stage_is_not_silently_serialized():
    function = carver.ElementwiseTemplate(shape=[4], dtype="int32").equivalent_function()
    parallel = tvm_ffi.structural_map(
        function, (ir.For, lambda loop: ir.For(loop.var, loop.begin, loop.end, loop.step, ir.ForKind.PARALLEL, loop.body, loc=loop.loc))
    )
    tirx.analysis.check_well_formed(parallel)
    schedule = NativeSchedule(parallel)
    assert schedule.normalize() is None
    with pytest.raises(ValueError, match="complete, directly bound"):
        schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("read", 0))
    assert schedule.function.same_as(parallel)
