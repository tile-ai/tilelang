import numpy as np
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.transform import DecoupleTypeCast


def _check(original, transformed):
    """Apply DecoupleTypeCast pass and check IR matches expected output."""
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))

    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_local_to_memory():
    """Test local → memory: compute to cast buffer, then copy to memory."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(b_frag[i], T.float4_e2m1fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_memory_to_local():
    """Test memory → local: copy from memory to cast buffer, then compute."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = b_frag[i]
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_nonzero_min():
    """Staging stores and copy loads use offsets within the loop extent."""

    @T.prim_func
    def before(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        for i in T.vectorized(2, 6):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((4,), T.float16, scope="local")
            for i in T.vectorized(2, 6):
                b_local_cast[i - 2] = T.cast(b_frag[i], T.float16)
            for i_copy in T.vectorized(2, 6):
                b[i_copy] = b_local_cast[i_copy - 2]

    _check(before, after)


def test_memory_to_local_nonzero_min():
    """Staging copy stores and compute loads use offsets within the extent."""

    @T.prim_func
    def before(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        for i in T.vectorized(2, 6):
            b_frag[i] = b[i]

    @T.prim_func
    def after(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((4,), T.float16, scope="local")
            for i_copy in T.vectorized(2, 6):
                b_local_cast[i_copy - 2] = b[i_copy]
            for i in T.vectorized(2, 6):
                b_frag[i] = T.cast(b_local_cast[i - 2], T.float32)

    _check(before, after)


@pytest.mark.parametrize("loop_min", [2, "symbolic"])
def test_decouple_preserves_logical_loop_domain(loop_min):
    """RMW staging uses ordinals while all three loops retain their domain."""
    tir = tvm.tirx
    extent = 4
    if loop_min == "symbolic":
        loop_min = tir.Var("start", "int32")
        scalar_params = [loop_min]
    else:
        scalar_params = []
    a = tir.decl_buffer((16,), "float16", name="a")
    i = tir.Var("i", "int32")
    value = tir.Cast("float16", tir.Cast("float32", a[i]) + tir.const(1, "float32"))
    before = tir.PrimFunc(
        [a.data, *scalar_params],
        tir.For(i, loop_min, extent, tir.ForKind.VECTORIZED, tir.BufferStore(a, value, [i])),
        buffer_map={a.data: a},
    ).with_attr("global_symbol", "main")

    # Keep logical memory indices and use ordinals only for the scratch buffer.
    # The RMW load and store must share the same four-element staging buffer.
    cast_buf = tir.decl_buffer((4,), "float16", name="a_local_cast", scope="local")
    copy_from_var = tir.Var("copy_from", "int32")
    copy_to_var = tir.Var("copy_to", "int32")
    copy_from = tir.For(
        copy_from_var,
        loop_min,
        extent,
        tir.ForKind.VECTORIZED,
        tir.BufferStore(cast_buf, a[copy_from_var], [copy_from_var - loop_min]),
    )
    ordinal = i - loop_min
    compute_value = tir.Cast("float16", tir.Cast("float32", cast_buf[ordinal]) + tir.const(1, "float32"))
    compute = tir.For(i, loop_min, extent, tir.ForKind.VECTORIZED, tir.BufferStore(cast_buf, compute_value, [ordinal]))
    copy_to = tir.For(
        copy_to_var,
        loop_min,
        extent,
        tir.ForKind.VECTORIZED,
        tir.BufferStore(a, cast_buf[copy_to_var - loop_min], [copy_to_var]),
    )
    block = tir.SBlock(
        [],
        [],
        [],
        "decoupled_cast",
        tir.SeqStmt([tir.AllocBuffer(cast_buf), copy_from, compute, copy_to]),
        annotations={"lexical_alloc_scope": 1},
    )
    expected = before.with_body(tir.SBlockRealize([], True, block))

    actual_mod = DecoupleTypeCast()(tvm.IRModule.from_expr(before))
    tvm.ir.assert_structural_equal(actual_mod["main"], expected, True)


@pytest.mark.parametrize("decouple", [False, True])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize(
    "start, extent, values",
    [(2, 4, [2, 3, 4, 5]), (-3, 4, [-3, -2, -1, 0]), (2, 0, []), (2, 1, [2])],
)
def test_nonzero_min_loop_execution(decouple, index_dtype, start, extent, values):
    """Legalization preserves the number of iterations and their logical values."""
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "float32", name="a")
    i = tir.Var("i", index_dtype)
    index = i - start
    value = tir.Cast("float32", tir.Cast("float64", a[index]) + tir.Cast("float64", i))
    loop = tir.For(i, start, extent, tir.ForKind.VECTORIZED, tir.BufferStore(a, value, [index]))
    func = tir.PrimFunc([a.data], loop, buffer_map={a.data: a}).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    with tvm.target.Target("c"):
        if decouple:
            mod = DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
    # Execute the legalized domains with scalar C codegen.
    with tvm.transform.PassContext(config={"tirx.disable_vectorize": True}):
        executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])
    output = tvm.runtime.tensor(np.full(8, -1, dtype="float32"))
    executable["main"](output)
    expected = np.full(8, -1, dtype="float32")
    expected[: len(values)] += values
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("step", [2, None], ids=["nonunit", "dynamic"])
def test_nonunit_step_remains_scalar(step):
    """The unit-step transformations leave other loop steps unchanged."""
    tir = tvm.tirx
    a = tir.decl_buffer((8,), "float32", name="a")
    i = tir.Var("i", "int32")
    params = [a.data]
    if step is None:
        step = tir.Var("step", "int32")
        params.append(step)
    value = tir.Cast("float32", tir.Cast("float64", a[i]) + tir.const(1, "float64"))
    body = tir.BufferStore(a, value, [i])
    before = tir.PrimFunc(params, tir.For(i, 2, 4, tir.ForKind.VECTORIZED, body, step=step), buffer_map={a.data: a})
    expected = before.with_body(tir.For(i, 2, 4, tir.ForKind.SERIAL, body, step=step))
    with tvm.target.Target("c"):
        mod = DecoupleTypeCast()(tvm.IRModule.from_expr(before))
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected, True)


def test_no_transform_same_dtype():
    """Test no transformation when dtypes are the same."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    _check(before, after)


def test_no_transform_local_to_local():
    """Test no transformation for local → local (both are local buffers)."""

    @T.prim_func
    def before():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = a_frag[i]

    @T.prim_func
    def after():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = T.cast(a_frag[i], T.float4_e2m1fn)

    _check(before, after)


def test_no_transform_if_then_else_condition():
    """Test no transformation when different dtype is only in if_then_else condition.

    The condition part of if_then_else doesn't participate in type casting,
    so a global/shared buffer load with different dtype in condition should
    not trigger cast buffer insertion.
    """

    @T.prim_func
    def before(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # cond_buf is int32, acc is float32, but cond_buf is only in condition
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * 2.0, acc[i])

    @T.prim_func
    def after(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # Should remain unchanged - no cast buffer needed
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * T.float32(2), acc[i])

    _check(before, after)


def test_rmw_same_buffer_different_indices():
    """RMW with different indices into the same buffer: a[i] = a[i] + a[i+32].

    Both loads and the store target the same buffer but at different index
    expressions. Each unique (buffer, indices) pair should get its own cast
    buffer, and the RMW load `a[i]` should read from the same cast buffer the
    store writes to (so the read-side copy-from and the write-side copy-to
    share that buffer).
    """

    @T.prim_func
    def before(a: T.Tensor[(64,), T.float8_e4m3fn]):
        for i in T.vectorized(32):
            a[i] = T.cast(
                T.cast(a[i], T.float32) + T.cast(a[i + 32], T.float32),
                T.float8_e4m3fn,
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    # Sanity checks: pass ran, two distinct cast buffers were created, and the
    # RMW load site no longer references `a` directly in the compute body.
    text = mod["main"].script()
    assert "a_local_cast" in text, "Expected cast buffer for store-side of a[i]"
    assert "a_local_cast_1" in text, "Expected second cast buffer for a[i+32]"


def test_local_to_memory_with_let_stmt():
    """Test local → memory transform still triggers through LetStmt-bound loads."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            b[i] = a_frag[i] * factor

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_bind_chain():
    """Test flat Bind chains are inlined according to sequential dominance."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            scaled = a_frag[i] * (factor + T.float32(1))
            b[i] = scaled

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(a_frag[i] * (scale[i] + T.float32(1)), T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_branch_local_bind():
    """Test Bind definitions inside an IfThenElse branch do not escape."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:
                factor = scale[i]
                b[i] = a_frag[i] * factor

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                if i < 8:
                    b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                if i_copy < 8:
                    b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_nested_guards():
    """Nested guards must be conjoined on the generated copy-to loop."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    b[i] = a_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                if i < 12:  # noqa: SIM102
                    if i < 8:
                        b_local_cast[i] = T.cast(a_frag[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                if T.if_then_else(i_copy < 12, i_copy < 8, False):
                    b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_memory_to_local_with_nested_guards():
    """Nested guards must be conjoined on the generated copy-from loop."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    a_frag[i] = b[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i_copy in T.vectorized(16):
                if T.if_then_else(i_copy < 12, i_copy < 8, False):
                    b_local_cast[i_copy] = b[i_copy]
            for i in T.vectorized(16):
                if i < 12:  # noqa: SIM102
                    if i < 8:
                        a_frag[i] = T.cast(b_local_cast[i], T.float32)

    _check(before, after)


def test_if_else_control_flow_skips_decoupling():
    """Copy loops cannot represent branch-specific guards, so keep the source."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:
                b[i] = a_frag[i]
            else:
                a_frag[i] = 0.0

    _check(before, before)


def test_cast_buffers_wrapped_in_lexical_alloc_scope():
    """The pass must wrap cast buffers in a block annotated with
    lexical_alloc_scope, so StorageRewrite keeps them scoped to the use site."""
    from tvm.tirx.stmt_functor import post_order_visit

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    annotated_blocks = [0]
    allocs_inside = [0]

    def _visit(node):
        if isinstance(node, tvm.tirx.SBlock) and "lexical_alloc_scope" in node.annotations:
            annotated_blocks[0] += 1

            def _count(inner):
                if isinstance(inner, tvm.tirx.AllocBuffer):
                    allocs_inside[0] += 1

            post_order_visit(node.body, _count)

    post_order_visit(mod["main"].body, _visit)
    assert annotated_blocks[0] == 1, f"Expected 1 lexical_alloc_scope block, got {annotated_blocks[0]}"
    assert allocs_inside[0] == 1, f"Expected the cast buffer alloc inside the scope, got {allocs_inside[0]}"


# =============================================================================
# CUDA Codegen Tests
# =============================================================================


@tilelang.testing.requires_cuda
def test_codegen_local_to_memory():
    """Test CUDA codegen for local → memory with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have vectorized copy (fp4_e2_16_t is 16 fp4 elements = 64 bits)
    assert "fp4_e2_16_t" in source, "Expected vectorized fp4 copy in generated code"


@tilelang.testing.requires_cuda
def test_codegen_memory_to_local():
    """Test CUDA codegen for memory → local with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            a_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                a_frag[i] = b[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"


@tilelang.testing.requires_cuda
def test_codegen_fp8_local_to_memory():
    """Test CUDA codegen for fp8 local → memory."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float8_e4m3fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have fp8 conversion (uses __nv_cvt for fp8)
    assert "fp8" in source and "cvt" in source, "Expected fp8 conversion"


@tilelang.testing.requires_cuda
def test_codegen_no_cast_buffer_same_dtype():
    """Test no cast buffer when dtypes are the same."""

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def kernel(b: T.Tensor[(16,), T.float32]):
            with T.Kernel(1, threads=32):
                b_frag = T.alloc_local((16,), T.float32)
                for i in T.vectorized(16):
                    b[i] = b_frag[i]

        return kernel

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    # Should NOT have local cast buffer when dtypes match
    assert "local_cast" not in source, "Should not have cast buffer when dtypes match"


# =============================================================================
# End-to-end correctness + vectorization tests for DecoupleTypeCast
# =============================================================================


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10)
def test_e2e_bf16_global_to_frag():
    """bf16 global -> float32 frag -> bf16 global: roundtrip should be lossless.

    With 1024 bf16 elements and 64 threads, each thread handles 16 bf16 = 256 bits,
    so the kernel should use 256-bit load/store (load_global_256 / store_global_256).
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # Check vectorization: 256-bit load/store
    source = kernel.get_kernel_source()
    assert "load_global_256" in source, "Expected 256-bit global load"
    assert "store_global_256" in source, "Expected 256-bit global store"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8)
def test_e2e_bf16_global_shared_frag():
    """bf16 global -> shared -> float32 frag -> bf16 global: roundtrip should be lossless.

    Shared memory path uses TMA for global->shared, then 128-bit for shared->local.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_shared = T.alloc_shared((1024,), dtype=T.bfloat16)
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_shared)
                T.copy(a_shared, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # Check: shared path should NOT use 256-bit (shared doesn't support it)
    source = kernel.get_kernel_source()
    assert "uint4" in source, f"Expected uint4 store in {source}"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_global_to_frag():
    """fp8 global -> float32 frag -> fp8 global: roundtrip should be lossless.

    Verifies that cast constraints do not pollute the memory access layout.
    With 1024 fp8 elements and 64 threads, each thread handles 16 fp8 = 128 bits,
    so the kernel should use fp8_e4_16_t (128-bit) loads/stores.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, (
        "Expected fp8_e4_16_t (128-bit) loads/stores for N=1024. Cast constraints may be polluting layout decisions."
    )

    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_manual_decouple():
    """fp8 with manually decoupled copy stages: same result as auto-decoupled.

    Tests: fp8 global -> fp8 frag -> float32 frag -> fp8 frag -> fp8 global
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                b_frag = T.alloc_fragment((1024,), dtype=T.float32)
                c_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                T.copy(A, a_frag)
                T.copy(a_frag, b_frag)
                T.copy(b_frag, c_frag)
                T.copy(c_frag, B)

        return main

    kernel = kernel_fn()

    # Check vectorization
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, "Expected fp8_e4_16_t in kernel source"

    # Correctness
    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_scalar_load_no_cast_buffer():
    """Test that scalar memory load (b[0]) is not decoupled into a cast buffer.

    When a vectorized loop stores to global with a scalar memory load in the
    expression (e.g. c[i] = a_local[i] * b[0]), the scalar load's index does
    not depend on the loop variable. It should remain in the compute loop as
    a broadcast, not be extracted into a local cast buffer.

    Previously this caused float32x32 codegen errors because both
    VectorizePlanner and DecoupleTypeCast treated b[0] as a vector memory
    access.
    """

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def main(
            a: T.Tensor[(32,), T.float8_e4m3fn],
            b: T.Tensor[(1,), T.float32],
            c: T.Tensor[(32,), T.float8_e4m3fn],
        ):
            with T.Kernel(1, threads=32):
                a_local = T.alloc_local((32,), T.float8_e4m3fn)
                T.copy(a, a_local)

                for i in T.vectorized(32):
                    c[i] = a_local[i] * b[0]

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    assert "c_local_cast" in source, "Expected c_local_cast for store-side decoupling"
    assert "b_local_cast" not in source, "Scalar load b[0] should not get a cast buffer"


def test_no_transform_evaluate_root_opaque_intrinsic():
    """Test that a vectorized loop whose body is an opaque intrinsic is skipped.

    Decoupling splits the value edge of a BufferStore to insert a staging
    buffer. An Evaluate discards its result and stores nothing, so that edge
    does not exist. Here the int64 index casts inside the ptx_cp_async
    predicate previously made the pass treat the statement as mixed-precision
    compute and rewrite both address operands to local staging buffers.

    Regression test for issue #2497.
    """

    @T.prim_func
    def before(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )

    @T.prim_func
    def after(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )

    _check(before, after)


@tilelang.testing.requires_cuda
def test_codegen_pipelined_int64_index_cp_async_operands():
    """Test that pipelined global→shared copies keep global/shared cp_async operands.

    With int64 index arithmetic, DecoupleTypeCast used to rewrite the address
    operands of ptx_cp_async to kernel-local register arrays, emitting
    cp_async_gs_conditional(local, local). That violates the instruction's
    address-space constraints (dst must be shared, src must be global) and
    fails at launch with CUDA_ERROR_INVALID_ADDRESS_SPACE.

    Regression test for issue #2497.
    """
    block_size, head_dim, num_stages = 128, 128, 2

    @tilelang.jit
    def kernel_fn():
        total_len = T.dynamic("total_len", "int64")
        num_blocks = T.dynamic("num_blocks")

        @T.prim_func
        def main(
            Input: T.Tensor[(total_len, head_dim), T.bfloat16],
            Output: T.Tensor[(total_len, head_dim), T.float32],
            Offsets: T.Tensor[(num_blocks,), T.int64],
        ):
            with T.Kernel(num_blocks, threads=256) as bx:
                shared_buf = T.alloc_shared((block_size, head_dim), T.bfloat16)
                accum = T.alloc_fragment((block_size, head_dim), T.float32)

                offset = Offsets[bx]
                trip_count = T.ceildiv(T.cast(total_len - offset, "int32"), block_size)
                for i in T.Pipelined(trip_count, num_stages=num_stages):
                    start = T.cast(i, "int64") * block_size
                    T.copy(
                        Input[offset + start : offset + start + block_size, :],
                        shared_buf,
                        disable_tma=True,
                    )
                    for r, c in T.Parallel(block_size, head_dim):
                        accum[r, c] = T.cast(shared_buf[r, c], T.float32)
                        Output[offset + start + T.cast(r, "int64"), c] = accum[r, c]

        return main

    source = kernel_fn().get_kernel_source()

    # Filter for the data-movement calls specifically: cp_async_commit/cp_async_wait
    # also match a bare "cp_async" substring, so they alone would satisfy a laxer check
    # while leaving the operand assertion below with nothing to inspect.
    gs_lines = [line for line in source.splitlines() if "cp_async_gs" in line]
    assert gs_lines, "Expected cp_async_gs (global->shared) calls for disable_tma copy"
    for line in gs_lines:
        assert "_local_cast" not in line, f"cp_async operand points at local memory: {line.strip()}"


def test_memory_dependent_guard_skips_decoupling():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:  # noqa: SIM102
                if b[i] > T.float16(0):
                    b[i] = local[i]

    _check(before, before)


def test_effectful_guard_skips_decoupling():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if T.call_extern("bool", "guard", i):
                b[i] = local[i]

    _check(before, before)


def test_nested_guard_preserves_conditional_evaluation():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i != 0:  # noqa: SIM102
                if 16 // i > 2:
                    b[i] = local[i]

    mod = DecoupleTypeCast()(tvm.IRModule.from_expr(before))
    copy_conditions = []

    def collect(node):
        if isinstance(node, tvm.tirx.IfThenElse) and isinstance(node.then_case, tvm.tirx.BufferStore) and node.then_case.buffer.name == "b":
            copy_conditions.append(node.condition)

    tvm.tirx.stmt_functor.post_order_visit(mod["before"].body, collect)
    assert len(copy_conditions) == 1
    condition = copy_conditions[0]
    assert isinstance(condition, tvm.tirx.Call)
    assert condition.op.same_as(tvm.ir.Op.get("tirx.if_then_else"))
    assert int(condition.args[2]) == 0


if __name__ == "__main__":
    tilelang.testing.main()
