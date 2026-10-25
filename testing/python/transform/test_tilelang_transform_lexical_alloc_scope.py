"""Tests for the lexical_alloc_scope feature.

Verifies that:
1. LowerOpaqueBlock preserves native scope attributes on explicitly marked scopes.
2. Unmarked scopes do NOT receive the marker by heuristic inference.
3. gemm_py-produced blocks are explicitly marked and survive as scopes.
4. StorageRewrite does not hoist allocations past the scope boundary.
5. CUDA codegen emits { ... } for the scoped block.
"""

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
from tilelang._compat.stmt_functor import post_order_visit
import tilelang.testing
from _native_ir import allocations


def _count_attrs(func, attr_key):
    """Count native regions carrying a scope attribute."""
    count = [0]

    def _visit(node):
        if isinstance(node, tvm.ir.RegionStmt) and attr_key in node.attrs:
            count[0] += 1

    post_order_visit(func.body, _visit)
    return count[0]


def _count_allocate_inside_attr(func, attr_key):
    """Count unique native allocations inside the marked lexical regions."""
    tensors = set()

    def _visit(node):
        if isinstance(node, tvm.ir.RegionStmt) and attr_key in node.attrs:
            tensors.update(allocations(node.body))

    post_order_visit(func.body, _visit)
    return len(tensors)


def _apply_lower_opaque_pipeline(func, target, pass_configs=None):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    pass_configs = pass_configs or {}
    with target, tvm.transform.PassContext(config=pass_configs):
        mod = CUDAPassPipelineBodyPrologue(mod, target)
        mod = tl.cuda.transform.LowerSharedTmem()(mod)
        mod = tl.transform.IfStmtBinding()(mod)
        mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.cuda.transform.LowerSharedBarrier()(mod)
        mod = tl.transform.HoistGlobalBufferAllocations()(mod)
        mod = tl.transform.LowerOpaqueBlock()(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: LowerOpaqueBlock inserts the lexical_alloc_scope marker for an
#         explicitly annotated block.
# ---------------------------------------------------------------------------
def test_lower_opaque_block_inserts_lexical_alloc_scope_for_explicit_block():
    """An explicitly annotated block should produce a lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
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
# Test 2: An unmarked block with local alloc should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_unmarked_local_alloc():
    """An unmarked local-alloc block should not produce a lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.scope():
                S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                S[tx] = A[tx]
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope AttrStmt for unmarked local block, got {n}"


# ---------------------------------------------------------------------------
# Test 3: Block without alloc_buffers should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_empty_alloc():
    """A block without alloc_buffers should not produce a lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.scope():
                B[tx] = A[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope AttrStmt for empty block, got {n}"


# ---------------------------------------------------------------------------
# Test 4: GEMM descriptor allocs inside loop should get the marker
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
def test_lower_opaque_block_inserts_scope_for_gemm_descriptor_alloc():
    """Lowered WGMMA descriptor buffers inside a loop should trigger lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})

    @T.prim_func
    def func(
        A: T.Tensor((64, 16), T.bfloat16),
        B: T.Tensor((64, 16), T.bfloat16),
        C: T.Tensor((64, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.bfloat16)
            B_shared = T.alloc_shared((64, 16), T.bfloat16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for _ in T.serial(2):
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0:64, 0:64])

    mod = _apply_lower_opaque_pipeline(func, target)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n >= 1, f"Expected lexical_alloc_scope for lowered GEMM descriptor alloc, got {n}"


# ---------------------------------------------------------------------------
# Test 5: local.var-only block inside loop should NOT get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_local_var_only_alloc():
    """A block that allocates only local.var should not get lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.scope():
                idx = T.alloc_var(T.int32)
                idx = tx
                B[tx] = A[idx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected 0 lexical_alloc_scope for local.var-only block, got {n}"


# ---------------------------------------------------------------------------
# Test 6: top-level explicitly annotated local alloc should get the marker
# ---------------------------------------------------------------------------
def test_lower_opaque_block_marks_explicit_top_level_local_alloc():
    """A top-level explicitly annotated local alloc should get lexical_alloc_scope."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
            S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
            S[tx] = A[tx]
            B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n >= 1, f"Expected lexical_alloc_scope for top-level local block, got {n}"


# ---------------------------------------------------------------------------
# Test 7: top-level fragment alloc should not force a lexical scope
# ---------------------------------------------------------------------------
def test_lower_opaque_block_skips_fragment_alloc():
    """A fragment alloc should not force lexical_alloc_scope by itself."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        with T.scope():
            S = T.alloc_buffer((128,), dtype=T.float32, scope="local.fragment")
            S[tx] = A[tx]
            B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(func)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    lowered = mod["main"]

    n = _count_attrs(lowered, "lexical_alloc_scope")
    assert n == 0, f"Expected no lexical_alloc_scope for fragment-only block, got {n}"


# ---------------------------------------------------------------------------
# Test 8: disable-ws pipelined GEMM should not wrap the fragment root block
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
def test_lower_opaque_block_skips_fragment_root_in_disable_ws_pipeline():
    """A fragment root block should not force lexical_alloc_scope in disable-ws pipeline."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    pass_configs = {tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED.value: True}

    @T.prim_func
    def func(
        A: T.Tensor((256, 256), T.bfloat16),
        B: T.Tensor((128, 256), T.bfloat16),
        C: T.Tensor((256, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((256, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_local = T.alloc_fragment((256, 128), T.float32)
            C_shared = T.alloc_shared((256, 128), T.bfloat16)
            T.clear(C_local)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[0, k * 128], A_shared)
                T.copy(B[0, k * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[0:256, 0:128])

    mod = _apply_lower_opaque_pipeline(func, target, pass_configs=pass_configs)
    lowered = mod["main"]
    (fragment,) = [tensor for tensor in allocations(lowered) if tensor.name == "C_local"]
    regions = []
    post_order_visit(lowered.body, lambda node: regions.append(node) if isinstance(node, tvm.ir.RegionStmt) else None)
    assert all(
        not any(tensor.same_as(fragment) for tensor in allocations(region.body))
        for region in regions
        if "lexical_alloc_scope" in region.attrs
    )
    assert tvm.tirx.analysis.check_well_formed(lowered)


# ---------------------------------------------------------------------------
# Test 9: StorageRewrite preserves lexical_alloc_scope
# ---------------------------------------------------------------------------
def test_storage_rewrite_preserves_scope():
    """lexical_alloc_scope should survive StorageRewrite without crashing."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})

    @T.prim_func
    def func(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        for _ in T.serial(4):
            with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
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
# Test 10: CUDA codegen emits { } for the scope
# ---------------------------------------------------------------------------
@tilelang.testing.requires_cuda
def test_codegen_emits_braces():
    """The generated CUDA source should contain scoped { } blocks for explicitly marked allocs."""

    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            for k in T.serial(4):
                with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    B[T.get_thread_binding(), k] = S[T.get_thread_binding()]
                # Sibling statement keeps the scope from being the sole loop
                # body, so its braces are not elided as redundant.
                B[T.get_thread_binding(), k] = B[T.get_thread_binding(), k] + T.float32(1)

    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== lexical_alloc_scope codegen ===")
    print(src)
    import re

    standalone_open_braces = re.findall(r"^\s*\{\s*$", src, re.MULTILINE)
    assert len(standalone_open_braces) >= 1, f"Expected at least 1 standalone '{{' for lexical scope, found {len(standalone_open_braces)}"


@tilelang.testing.requires_cuda
def test_codegen_unwraps_scope_spanning_entire_loop_body():
    """Native lexical wrappers must keep the temporary inside its loop."""

    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            for k in T.serial(4):
                with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    B[T.get_thread_binding(), k] = S[T.get_thread_binding()]

    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    import re

    # The local alloc must still be declared inside the loop body, not hoisted
    # to the kernel entry.
    loop_pos = src.find("for (")
    decl_pos = src.find("float S_ptr[")
    assert loop_pos >= 0 and decl_pos > loop_pos, f"Expected S declared inside the loop body:\n{src}"
    import torch

    source = torch.randn(128, 4, device="cuda")
    torch.testing.assert_close(kernel(source), source)


@tilelang.testing.requires_cuda
def test_codegen_skips_redundant_top_level_braces():
    """Native function-local and inner lexical allocations stay distinct."""

    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            C = T.alloc_fragment((128,), T.float32)
            T.clear(C)
            for k in T.serial(4):
                with T.scope(attrs={**{"lexical_alloc_scope": 1}}):
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    C[T.get_thread_binding()] = S[T.get_thread_binding()]
                # Sibling statement keeps the inner scope's braces from being
                # elided as redundant with the loop body.
                C[T.get_thread_binding()] = C[T.get_thread_binding()] + T.float32(1)
            for k in T.serial(4):
                B[T.get_thread_binding(), k] = C[T.get_thread_binding()]

    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== top-level lexical_alloc_scope codegen ===")
    print(src)
    import re

    assert re.search(r"^\s*(?:alignas\(\d+\)\s+)?float C_ptr\[\d+\];\s*$", src, re.MULTILINE), "Expected function-local fragment allocation"
    kernel_match = re.search(
        r'extern "C" __global__ void(?: __launch_bounds__\([^)]*\))? [A-Za-z_]\w*_kernel\(',
        src,
    )
    assert kernel_match, "Expected generated CUDA source to contain a kernel signature"
    body_open_idx = src.find("{", kernel_match.start())
    assert body_open_idx >= 0, "Expected generated CUDA kernel body"
    fragment_position = src.find("float C_ptr[", body_open_idx)
    loop_position = src.find("for (", body_open_idx)
    assert body_open_idx < fragment_position < loop_position
    standalone_open_braces = re.findall(r"^\s*\{\s*$", src, re.MULTILINE)
    assert len(standalone_open_braces) >= 1, "Expected inner lexical scopes to still emit standalone braces"
    import torch

    source = torch.randn(128, 4, device="cuda")
    expected = (source[:, -1:] + 1).expand_as(source)
    torch.testing.assert_close(kernel(source), expected)


if __name__ == "__main__":
    test_lower_opaque_block_inserts_lexical_alloc_scope_for_explicit_block()
    test_lower_opaque_block_skips_unmarked_local_alloc()
    test_lower_opaque_block_skips_empty_alloc()
    test_lower_opaque_block_inserts_scope_for_gemm_descriptor_alloc()
    test_lower_opaque_block_skips_local_var_only_alloc()
    test_lower_opaque_block_marks_explicit_top_level_local_alloc()
    test_lower_opaque_block_skips_fragment_alloc()
    test_lower_opaque_block_skips_fragment_root_in_disable_ws_pipeline()
    test_storage_rewrite_preserves_scope()
    test_codegen_emits_braces()
    test_codegen_unwraps_scope_spanning_entire_loop_body()
    test_codegen_skips_redundant_top_level_braces()
    print("All tests passed!")
