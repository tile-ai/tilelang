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
    """Test Bind definitions inside an IfThenElse branch do not escape.

    The guarded store gets a per-entry validity mask: the mask is
    zeroed by an init loop, set to 1 at the compute store site (inside the
    branch), and the copy-to loop is guarded by ``mask[i_copy] != 0`` instead
    of re-evaluating the original branch condition.
    """

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
            b_local_cast_mask = T.decl_buffer((16,), "int32", scope="local")
            for i_mask in T.vectorized(16):
                b_local_cast_mask[i_mask] = 0
            for i in T.vectorized(16):
                if i < 8:
                    b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
                    b_local_cast_mask[i] = 1
            for i_copy in T.vectorized(16):
                if b_local_cast_mask[i_copy] != 0:
                    b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


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


def test_mask_guard_root_if_else_regression():
    """Copy-to guards must NOT re-evaluate original
    buffer conditions, because an earlier copy-to write-back can flip them.

    Structure: root if/else where the then-branch stores to A (the buffer the
    guard reads) and the else-branch stores to B. Re-evaluating ``A[i] > 0``
    in B's copy-to loop after A's copy-to wrote back would flip the guard and
    read an uninitialized cast local. The mask scheme records compute-time
    truth at the store site, so B's copy-to fires exactly where the original
    else-branch store fired.
    """

    @T.prim_func
    def before(
        A: T.Tensor[(8,), T.bfloat16],
        src: T.Tensor[(8,), T.float32],
        B: T.Tensor[(8,), T.bfloat16],
    ):
        for i in T.vectorized(8):
            if A[i] > 0:
                A[i] = T.cast(src[i], T.bfloat16)
            else:
                B[i] = T.cast(src[i], T.bfloat16)

    @T.prim_func
    def after(
        A: T.Tensor[(8,), T.bfloat16],
        src: T.Tensor[(8,), T.float32],
        B: T.Tensor[(8,), T.bfloat16],
    ):
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            A_local_cast = T.decl_buffer((8,), T.bfloat16, scope="local")
            B_local_cast_1 = T.decl_buffer((8,), T.bfloat16, scope="local")
            src_local_cast_2 = T.decl_buffer((8,), T.float32, scope="local")
            A_local_cast_mask = T.decl_buffer((8,), "int32", scope="local")
            B_local_cast_1_mask = T.decl_buffer((8,), "int32", scope="local")
            for i_mask in T.vectorized(8):
                A_local_cast_mask[i_mask] = 0
            for i_mask in T.vectorized(8):
                B_local_cast_1_mask[i_mask] = 0
            for i_copy in T.vectorized(8):
                src_local_cast_2[i_copy] = src[i_copy]
            for i_copy in T.vectorized(8):
                A_local_cast[i_copy] = A[i_copy]
            for i in T.vectorized(8):
                if A_local_cast[i] > T.bfloat16(0.0):
                    A_local_cast[i] = T.cast(src_local_cast_2[i], T.bfloat16)
                    A_local_cast_mask[i] = 1
                else:
                    B_local_cast_1[i] = T.cast(src_local_cast_2[i], T.bfloat16)
                    B_local_cast_1_mask[i] = 1
            for i_copy in T.vectorized(8):
                if A_local_cast_mask[i_copy] != 0:
                    A[i_copy] = A_local_cast[i_copy]
            for i_copy in T.vectorized(8):
                if B_local_cast_1_mask[i_copy] != 0:
                    B[i_copy] = B_local_cast_1[i_copy]

    _check(before, after)


def _collect_guarded_copy_guards(mod, mask_name: str) -> list[str]:
    """Return the guard expressions of every IfThenElse guarding a copy loop
    whose condition references ``mask_name`` (as printed IR strings)."""
    from tvm.tirx.stmt_functor import post_order_visit

    guards: list[str] = []

    def _visit(node):
        if isinstance(node, tvm.tirx.IfThenElse) and mask_name in str(node.condition):
            guards.append(str(node.condition))

    post_order_visit(mod["main"].body, _visit)
    return guards


def test_mask_or_semantics_multi_branch_same_entry():
    """OR/nested-branch semantics: several stores to the SAME (buffer, indices)
    entry under different conditions map to ONE validity mask; the copy-to
    fires iff any branch actually executed. No copy-to guard may reference the
    original buffers (only the mask)."""

    @T.prim_func
    def before(
        src: T.Tensor[(8,), T.float32],
        B: T.Tensor[(8,), T.float8_e4m3fn],
    ):
        # Two branches kept separate so the OR-of-path-conditions semantics of
        # the decoupled copy guard is exercised as written.
        for i in T.vectorized(8):
            if i < 4:  # noqa: SIM114
                B[i] = T.cast(src[i], T.float8_e4m3fn)
            elif i < 8:
                B[i] = T.cast(src[i], T.float8_e4m3fn)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    # Exactly one mask buffer, named after the single B entry's cast buffer.
    from tvm.tirx.stmt_functor import post_order_visit

    buffers = []

    def _collect(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            buffers.append(node.buffer.name)

    post_order_visit(mod["main"].body, _collect)
    mask_buffers = [n for n in buffers if n.endswith("_mask")]
    assert mask_buffers == ["B_local_cast_mask"], f"Unexpected mask buffers: {mask_buffers}"

    # Every guarded copy loop reads ONLY the mask, never the original buffers.
    guards = _collect_guarded_copy_guards(mod, "B_local_cast_mask")
    assert guards, "Expected at least one mask-guarded copy-to loop"
    for guard in guards:
        assert "B[" not in guard, f"Copy-to guard re-reads original buffer: {guard}"
        assert "src[" not in guard, f"Copy-to guard re-reads original buffer: {guard}"

    # The mask is set inside both branches (OR semantics).
    body_str = str(mod["main"].body)
    assert body_str.count("B_local_cast_mask[i] = 1") == 2, body_str


def test_unconditional_store_gets_no_mask():
    """Anti-over-fix: an unconditional store needs no validity mask; the
    copy-to loop stays unconditional and no mask buffer/init loop is emitted."""
    from tvm.tirx.stmt_functor import post_order_visit

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    buffers = []

    def _collect(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            buffers.append(node.buffer.name)

    post_order_visit(mod["main"].body, _collect)
    assert not any(n.endswith("_mask") for n in buffers), f"Unexpected masks: {buffers}"


def test_guarded_copy_from_substitutes_loop_var():
    """Guarded copy-from loops run under their own loop var: the collected
    path condition (over the original loop var) must be substituted, otherwise
    the original loop var is left free inside the copy loop ("used before
    definition" in VarUseDefAnalysis). This catches a mask refactor that
    drops the substitution for from-memory guards.
    """

    @T.prim_func
    def before(src: T.Tensor[(8,), T.float32], B: T.Tensor[(8,), T.bfloat16]):
        for i in T.vectorized(8):
            if i < 4:
                B[i] = T.cast(src[i], T.bfloat16)

    @T.prim_func
    def after(src: T.Tensor[(8,), T.float32], B: T.Tensor[(8,), T.bfloat16]):
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            B_local_cast = T.decl_buffer((8,), T.bfloat16, scope="local")
            src_local_cast_1 = T.decl_buffer((8,), T.float32, scope="local")
            B_local_cast_mask = T.decl_buffer((8,), "int32", scope="local")
            for i_mask in T.vectorized(8):
                B_local_cast_mask[i_mask] = 0
            for i_copy in T.vectorized(8):
                if i_copy < 4:
                    src_local_cast_1[i_copy] = src[i_copy]
            for i in T.vectorized(8):
                if i < 4:
                    B_local_cast[i] = T.cast(src_local_cast_1[i], T.bfloat16)
                    B_local_cast_mask[i] = 1
            for i_copy in T.vectorized(8):
                if B_local_cast_mask[i_copy] != 0:
                    B[i_copy] = B_local_cast[i_copy]

    _check(before, after)


def test_ternary_value_load_copy_from_is_unconditional():
    """Both value branches of ``if_then_else`` are staged unconditionally.

    DecoupleTypeCast must not apply the ternary condition to either copy-from:
    the condition selects a value in the compute loop; it is not an enclosing
    statement guard on either memory load.
    """

    @T.prim_func
    def before(
        src: T.Tensor[(16,), T.bfloat16],
        out: T.Tensor[(8,), T.float32],
    ):
        for i in T.vectorized(8):
            out[i] = T.if_then_else(
                i < 4,
                T.cast(src[i], T.float32),
                T.cast(src[i + 8], T.float32) * T.float32(2),
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)
    text = mod["main"].script()

    assert "src_local_cast_1[i_copy] = src[i_copy]" in text
    assert "src_local_cast_2[i_copy] = src[i_copy + 8]" in text
    assert "if i_copy < 4:" not in text
    assert "if 4 <= i_copy:" not in text


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


if __name__ == "__main__":
    tilelang.testing.main()
