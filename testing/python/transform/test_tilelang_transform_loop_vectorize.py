import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
_SM89_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
_SM90_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})


def _run_layout_inference(func):
    mod = tvm.IRModule({"main": func})
    with _TARGET:
        mod = tvm.tirx.transform.BindTarget(_TARGET)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        return tl.transform.LayoutInference()(mod)


def _run_vectorized_loop_legalizer(func):
    mod = tvm.IRModule({"main": func})
    with _TARGET:
        return tl.transform.LegalizeVectorizedLoop()(mod)


def _run_vectorize_loop(func, target=_SM90_TARGET):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.LowerAccessPtr()(mod)
        mod = tl.transform.FlattenBuffer()(mod)
        return tl.transform.VectorizeLoop()(mod)


def _atomic_op_names(func):
    names = []

    def collect(node):
        if isinstance(node, tvm.tirx.Call) and hasattr(node.op, "name") and node.op.name.startswith("tl.atomic_add"):
            names.append(node.op.name)

    post_order_visit(func.body, collect)
    return names


def _loop_kinds(func):
    kinds = []

    def collect(node):
        if isinstance(node, tvm.tirx.For):
            kinds.append(node.kind)

    post_order_visit(func.body, collect)
    return kinds


def _undefined_local_vars(func):
    defined_vars = [*func.params, *(buffer.data for buffer in func.buffer_map.values())]
    return tvm.tirx.analysis.undefined_vars(func.body, defined_vars)


def test_fp32_shared_atomic_stays_scalar_on_sm90():
    @T.prim_func
    def main(A: T.Tensor((4,), T.float32)):
        shared = T.alloc_buffer((4,), "float32", scope="shared")
        for i in T.vectorized(4):
            T.atomic_add(shared[i], A[i])

    transformed = _run_vectorize_loop(main)

    assert _atomic_op_names(transformed["main"]) == ["tl.atomic_add_elem_op"]
    assert _loop_kinds(transformed["main"]) == [tvm.tirx.ForKind.SERIAL]
    assert not _undefined_local_vars(transformed["main"])


def test_fp32_atomic_stays_scalar_on_sm89():
    @T.prim_func
    def main(A: T.Tensor((4,), T.float32), output: T.Tensor((4,), T.float32)):
        for i in T.vectorized(4):
            T.atomic_add(output[i], A[i])

    transformed = _run_vectorize_loop(main, _SM89_TARGET)

    assert _atomic_op_names(transformed["main"]) == ["tl.atomic_add_elem_op"]
    assert _loop_kinds(transformed["main"]) == [tvm.tirx.ForKind.SERIAL]
    assert not _undefined_local_vars(transformed["main"])


def test_fp32_global_atomic_keeps_x4_vectorization_on_sm90():
    @T.prim_func
    def main(A: T.Tensor((4,), T.float32), output: T.Tensor((4,), T.float32)):
        for i in T.vectorized(4):
            T.atomic_add(output[i], A[i])

    transformed = _run_vectorize_loop(main)

    assert _atomic_op_names(transformed["main"]) == ["tl.atomic_addx4_elem_op"]


def test_fp16_shared_atomic_keeps_x2_vectorization_on_sm90():
    @T.prim_func
    def main(A: T.Tensor((2,), T.float16)):
        shared = T.alloc_buffer((2,), "float16", scope="shared")
        for i in T.vectorized(2):
            T.atomic_add(shared[i], A[i])

    transformed = _run_vectorize_loop(main)

    assert _atomic_op_names(transformed["main"]) == ["tl.atomic_addx2_elem_op"]


def _vectorized_extents(func):
    extents = []

    def collect(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.VECTORIZED:
            extents.append(int(node.extent))

    post_order_visit(func.body, collect)
    return extents


def _plan_atomic_loop(index, pointer_form="access_ptr", dtype="float32", scope="global", shape=(64,), strides=None, elem_offset=0):
    """Check the planner before the execution vectorizer can scalarize anything."""
    tir = tvm.tirx
    i = tir.Var("i", "int32")
    offset = tir.Var("offset", "int64")
    source = tir.decl_buffer((8,), dtype, name="source")
    output = tir.decl_buffer(shape, dtype, name="output", scope=scope, strides=strides, elem_offset=elem_offset)
    indices = index(i, offset)
    if not isinstance(indices, list):
        indices = [indices]
    load = tir.BufferLoad(output, indices)
    if pointer_form == "access_ptr":
        ptr = tir.call_intrin("handle", tvm.ir.Op.get("tl.access_ptr"), load, 1, 3)
    elif pointer_form == "address_of":
        ptr = tir.call_intrin("handle", tvm.ir.Op.get("tirx.address_of"), load)
    else:
        assert len(indices) == 1
        ptr = tir.tvm_access_ptr(tir.type_annotation(dtype), output.data, indices[0], 1, 3)
    atomic = tir.call_intrin("handle", tvm.ir.Op.get("tl.atomic_add_elem_op"), ptr, source[i], 0)
    body = tir.For(i, 0, 8, tir.ForKind.VECTORIZED, tir.Evaluate(atomic))
    func = tir.PrimFunc([source, output, offset], body).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("pointer_form", ["access_ptr", "address_of", "tvm_access_ptr"])
@pytest.mark.parametrize(
    "index,width",
    [
        pytest.param(lambda i, _: i, 4, id="contiguous"),
        pytest.param(lambda i, _: i + 2, 2, id="two-aligned"),
        pytest.param(lambda i, _: i // 2, 1, id="repeated"),
        pytest.param(lambda i, _: (i // 2) * 2, 1, id="repeated-aligned"),
        pytest.param(lambda i, _: (i // 2) * 2 + 1, 1, id="repeated-odd"),
        pytest.param(lambda i, _: 0, 1, id="constant"),
        pytest.param(lambda i, _: i * 2, 1, id="strided"),
        pytest.param(lambda i, _: i % 2, 2, id="periodic"),
        pytest.param(lambda i, _: i // 4 * 6 + i % 4, 2, id="later-group-two-aligned"),
        pytest.param(lambda i, _: i // 4 * 5 + i % 4, 1, id="later-group-unaligned"),
        pytest.param(lambda i, offset: offset * 4 + i.astype("int64"), 4, id="dynamic-int64"),
        pytest.param(lambda i, offset: offset + i.astype("int64"), 1, id="unknown-alignment"),
    ],
)
def test_atomic_destination_width_is_planned(index, width, pointer_form):
    mod = _plan_atomic_loop(index, pointer_form)
    with _SM90_TARGET:
        planned = tl.transform.LegalizeVectorizedLoop()(mod)
        assert _vectorized_extents(planned["main"]) == ([width] if width > 1 else [])
        lowered = tl.transform.LowerAccessPtr()(planned)
        lowered = tl.transform.FlattenBuffer()(lowered)
        emitted = tl.transform.VectorizeLoop()(lowered)
    op = f"tl.atomic_addx{width}_elem_op" if width > 1 else "tl.atomic_add_elem_op"
    assert _atomic_op_names(emitted["main"]) == [op]
    assert not _undefined_local_vars(emitted["main"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("pointer_form", ["access_ptr", "address_of", "tvm_access_ptr"])
@pytest.mark.parametrize(
    "dtype,scope,target,width",
    [
        ("float32", "global", _SM90_TARGET, 4),
        ("float32", "", _SM90_TARGET, 4),
        ("float32", "shared", _SM90_TARGET, 1),
        ("float32", "global", _SM89_TARGET, 1),
        ("float16", "global", _SM90_TARGET, 2),
        ("float16", "shared", _SM90_TARGET, 2),
        ("bfloat16", "global", _SM90_TARGET, 2),
    ],
)
def test_atomic_capability_limits_the_plan(dtype, scope, target, width, pointer_form):
    mod = _plan_atomic_loop(lambda i, _: i, pointer_form=pointer_form, dtype=dtype, scope=scope)
    with target:
        planned = tl.transform.LegalizeVectorizedLoop()(mod)
        assert _vectorized_extents(planned["main"]) == ([width] if width > 1 else [])
        lowered = tl.transform.LowerAccessPtr()(planned)
        lowered = tl.transform.FlattenBuffer()(lowered)
        emitted = tl.transform.VectorizeLoop()(lowered)
    op = f"tl.atomic_addx{width}_elem_op" if width > 1 else "tl.atomic_add_elem_op"
    assert _atomic_op_names(emitted["main"]) == [op]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("strides,elem_offset,width", [(None, 0, 4), ((4, 2), 0, 1), (None, 2, 2)])
def test_atomic_plan_uses_physical_offset(strides, elem_offset, width):
    mod = _plan_atomic_loop(lambda i, _: [i // 2, i % 2], shape=(4, 2), strides=strides, elem_offset=elem_offset)
    with _SM90_TARGET:
        planned = tl.transform.LegalizeVectorizedLoop()(mod)
    assert _vectorized_extents(planned["main"]) == ([width] if width > 1 else [])


@pytest.mark.parametrize("access_kind", ["load", "store", "implicit-row-stride"])
def test_large_int32_affine_index_does_not_abort_layout_inference(access_kind):
    """Regression for issue #3027's host-side vectorization analysis ICE."""
    extent = 4
    stride = 600_000_000

    if access_kind == "load":

        @T.prim_func
        def main(
            A: T.Tensor((2_000_000_000,), T.float32),
            output: T.Tensor((extent,), T.float32),
        ):
            with T.Kernel(1, threads=extent):
                for i in T.Parallel(extent):
                    output[i] = A[i * stride]

    elif access_kind == "store":

        @T.prim_func
        def main(
            A: T.Tensor((extent,), T.float32),
            output: T.Tensor((2_000_000_000,), T.float32),
        ):
            with T.Kernel(1, threads=extent):
                for i in T.Parallel(extent):
                    output[i * stride] = A[i]

    else:

        @T.prim_func
        def main(
            A: T.Tensor((extent, stride), T.float32),
            output: T.Tensor((extent,), T.float32),
        ):
            with T.Kernel(1, threads=extent):
                for i in T.Parallel(extent):
                    output[i] = A[i, 0]

    _run_layout_inference(main)


@pytest.mark.parametrize("extent", [4, 8])
def test_large_int32_affine_index_with_bounded_residual_does_not_abort(extent):
    """The final index fits int32 even though canonical coefficients do not."""
    group_size = 4
    stride = 600_000_000

    @T.prim_func
    def main(
        A: T.Tensor((2_000_000_000,), T.float32),
        output: T.Tensor((extent,), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.Parallel(extent):
                output[i] = A[(i - (i // group_size) * group_size) * stride]

    _run_layout_inference(main)


def test_large_int32_bounded_residual_condition_does_not_abort():
    """Conditions must be widened before their first canonical simplification."""
    extent = 8
    group_size = 4
    stride = 600_000_000

    @T.prim_func
    def main(
        A: T.Tensor((extent,), T.float32),
        output: T.Tensor((extent,), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.Parallel(extent):
                if (i - (i // group_size) * group_size) * stride != 0:
                    output[i] = A[i]

    _run_layout_inference(main)


def test_large_int32_bounded_residual_access_ptr_does_not_abort():
    """tvm_access_ptr offsets must be widened before simplification."""
    extent = 8
    group_size = 4
    stride = 600_000_000

    @T.prim_func
    def main(A: T.Tensor((2_000_000_000,), T.float32)):
        with T.Kernel(1, threads=extent):
            for i in T.Parallel(extent):
                T.evaluate(
                    T.call_intrin(
                        "float32",
                        tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                        T.tvm_access_ptr(
                            T.type_annotation("float32"),
                            A.data,
                            (i - (i // group_size) * group_size) * stride,
                            1,
                            3,
                        ),
                        T.float32(1),
                        T.int32(0),
                    )
                )

    _run_layout_inference(main)


def test_overflow_promotion_preserves_let_var_binding():
    """Promoted Let bodies must reference the rewritten binder."""
    extent = 4
    stride = 600_000_000
    loop_var = tvm.tirx.Var("i", "int32")
    let_var = tvm.tirx.Var("offset", "int32")
    input_buffer = tvm.tirx.decl_buffer((2_000_000_000,), "float32", name="A")
    output_buffer = tvm.tirx.decl_buffer((extent,), "float32", name="output")
    scaled_offset = loop_var * stride
    index = tvm.tirx.Let(let_var, scaled_offset, let_var - scaled_offset)
    body = tvm.tirx.BufferStore(
        output_buffer,
        tvm.tirx.BufferLoad(input_buffer, [index]),
        [loop_var],
    )
    loop = tvm.tirx.For(
        loop_var,
        0,
        extent,
        tvm.tirx.ForKind.VECTORIZED,
        body,
    )
    func = tvm.tirx.PrimFunc(
        [input_buffer.data, output_buffer.data],
        loop,
        buffer_map={
            input_buffer.data: input_buffer,
            output_buffer.data: output_buffer,
        },
    )

    transformed = _run_vectorized_loop_legalizer(func)

    assert _vectorized_extents(transformed["main"]) == [extent]


def test_large_int32_affine_index_under_int64_cast_does_not_abort():
    """A wide cast must preserve the narrow arithmetic it encloses."""
    extent = 8
    group_size = 4
    stride = 600_000_000

    @T.prim_func
    def main(
        A: T.Tensor((2_000_000_000,), T.float32),
        output: T.Tensor((extent,), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.Parallel(extent):
                idx = T.cast(
                    (i - (i // group_size) * group_size) * stride,
                    "int64",
                )
                output[i] = A[idx]

    load_indices = []
    bindings = []

    def collect(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name == "A":
            load_indices.append(node.indices[0])
        elif isinstance(node, tvm.tirx.Bind):
            bindings.append((node.var, node.value))

    post_order_visit(main.body, collect)
    assert len(load_indices) == 1
    assert isinstance(load_indices[0], tvm.tirx.Var)
    cast_values = [value for var, value in bindings if var.same_as(load_indices[0])]
    assert len(cast_values) == 1
    assert isinstance(cast_values[0], tvm.tirx.Cast)
    assert str(cast_values[0].dtype) == "int64"
    assert str(cast_values[0].value.dtype) == "int32"

    _run_layout_inference(main)


def test_large_offset_unit_stride_remains_vectorizable():
    """Analysis-only int64 promotion must preserve unit-stride Ramp detection."""
    extent = 4
    offset = 600_000_000

    @T.prim_func
    def main(
        A: T.Tensor((2_000_000_000,), T.float32),
        output: T.Tensor((extent,), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.vectorized(extent):
                output[i] = A[offset + i]

    transformed = _run_vectorized_loop_legalizer(main)

    assert _vectorized_extents(transformed["main"]) == [extent]


def test_large_coefficient_boundary_invariant_index_remains_vectorizable():
    """A large coefficient must not turn a successful proof into scalarization."""
    extent = 8
    stride = 600_000_000
    vector_size = 4

    @T.prim_func
    def main(
        A: T.Tensor((2_000_000_000,), T.float32),
        output: T.Tensor((extent,), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.vectorized(extent):
                output[i] = A[(i // vector_size) * stride]

    transformed = _run_vectorized_loop_legalizer(main)

    assert _vectorized_extents(transformed["main"]) == [vector_size]


def test_overflow_promotion_preserves_atomic_vector_lanes():
    """Widening a dynamic row offset must not break x4 atomic lanes."""
    extent = 128
    lanes = 4
    row_size = 576

    @T.prim_func
    def main(
        A: T.Tensor((512, row_size), T.float32),
        indices: T.Tensor((extent,), T.int32),
        values: T.Tensor((extent, lanes), T.float32),
    ):
        with T.Kernel(1, threads=extent):
            for i in T.Parallel(extent):
                T.atomic_addx4(A[indices[i], i * lanes], values[i, 0])

    _run_layout_inference(main)
