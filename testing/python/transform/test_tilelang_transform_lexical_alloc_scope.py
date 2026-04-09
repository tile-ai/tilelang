"""Tests for the lexical_alloc_scope feature.

Verifies that:
1. LowerOpaqueBlock inserts AttrStmt("lexical_alloc_scope") for blocks
   with allocations that are nested inside a loop.
2. Top-level blocks (not inside any loop) do NOT receive the marker.
3. StorageRewrite does not hoist allocations past the scope boundary.
4. CUDA codegen emits { ... } for the scoped block.
"""

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tvm.tir.stmt_functor import post_order_visit
import tilelang.testing


def _count_attrs(func, attr_key):
    """Count occurrences of a specific AttrStmt key in the function body."""
    count = [0]

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            count[0] += 1

    post_order_visit(func.body, _visit)
    return count[0]


def _count_allocate_inside_attr(func, attr_key):
    """Count Allocate nodes that are (transitively) nested inside the given AttrStmt."""
    count = [0]
    inside = [False]

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            old = inside[0]
            inside[0] = True
            post_order_visit(node.body, _visit)
            inside[0] = old
        elif isinstance(node, tvm.tir.Allocate) and inside[0]:
            count[0] += 1

    post_order_visit(func.body, _visit)
    return count[0]


# ---------------------------------------------------------------------------
# Test 1: LowerOpaqueBlock inserts the lexical_alloc_scope marker for
#          blocks with allocations inside a loop.
# ---------------------------------------------------------------------------
def test_lower_opaque_block_inserts_lexical_alloc_scope():
    """A block with alloc_buffers inside a loop should produce a lexical_alloc_scope."""
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.block():
                S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                S[tx] = A[tx]
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n >= 1, f"Expected at least 1 lexical_alloc_scope AttrStmt, got {n}"

    # The Allocate for S should be inside the scope
    n_alloc = _count_allocate_inside_attr(lowered, "lexical_alloc_scope")
    assert n_alloc >= 1, f"Expected Allocate inside lexical_alloc_scope, got {n_alloc}"


# ---------------------------------------------------------------------------
# Test 2: Block without alloc_buffers should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_empty_alloc():
    """A block without alloc_buffers should not produce a lexical_alloc_scope."""
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.block():
                B[tx] = A[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope AttrStmt for empty block, got {n}"


# ---------------------------------------------------------------------------
# Test 3: local.var-only block inside loop should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_local_var_only_alloc():
    """A block that allocates only local.var should not get lexical_alloc_scope."""
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.block():
                idx = T.alloc_var(T.int32)
                idx = tx
                B[tx] = A[idx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope for local.var-only block, got {n}"


# ---------------------------------------------------------------------------
# Test 4: Top-level block (not inside a loop) should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_top_level():
    """A top-level block with alloc_buffers should NOT get lexical_alloc_scope."""
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        with T.block():
            S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
            S[tx] = A[tx]
            B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope for top-level block, got {n}"


# ---------------------------------------------------------------------------
# Test 5: StorageRewrite preserves lexical_alloc_scope
# ---------------------------------------------------------------------------
def test_storage_rewrite_preserves_scope():
    """lexical_alloc_scope should survive StorageRewrite without crashing."""
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.block():
                S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                S[tx] = A[tx]
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    mod = tl.transform.StorageRewrite()(mod)
    lowered = mod["main"]

    # The scope marker should still be present after StorageRewrite
    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n >= 1, f"Expected lexical_alloc_scope to survive StorageRewrite, got {n}"


# ---------------------------------------------------------------------------
# Test 6: CUDA codegen emits { } for the scope
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
def test_codegen_emits_braces():
    """The generated CUDA source should contain scoped { } blocks for loop-local allocs."""

    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            for k in T.serial(4):
                S = T.alloc_local((128,), T.float32)
                S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                B[T.get_thread_binding(), k] = S[T.get_thread_binding()]

    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== lexical_alloc_scope codegen ===")
    print(src)
    import re

    standalone_open_braces = re.findall(r"^\s*\{\s*$", src, re.MULTILINE)
    assert len(standalone_open_braces) >= 1, f"Expected at least 1 standalone '{{' for lexical scope, found {len(standalone_open_braces)}"


if __name__ == "__main__":
    test_lower_opaque_block_inserts_lexical_alloc_scope()
    test_lower_opaque_block_skips_empty_alloc()
    test_lower_opaque_block_skips_local_var_only_alloc()
    test_lower_opaque_block_skips_top_level()
    test_storage_rewrite_preserves_scope()
    test_codegen_emits_braces()
    print("All tests passed!")
