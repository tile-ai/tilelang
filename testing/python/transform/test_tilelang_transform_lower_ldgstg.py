"""Tests for LowerLDGSTG pass that converts Ramp-based global memory
load/store to ldg/stg intrinsics.

Pass configurations:
- tl.enable_lower_ldgstg: Enable non-predicated ldg/stg lowering (default: OFF)
- tl.enable_lower_ldgstg_predicated: Enable predicated ldg/stg lowering (default: OFF)
"""

import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
from tilelang.transform import PassConfigKey
from tvm import tirx


def _apply_passes(mod, enable_non_predicated=False, enable_predicated=False):
    """Apply the LowerLDGSTG pass and related lowering passes."""
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target("cuda"))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: enable_non_predicated,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: enable_predicated,
        }
    ):
        mod = tl.cuda.transform.LowerLDGSTG()(mod)
    return mod


requires_ldgstg = pytest.mark.skipif(
    tvm.get_global_func("tl.cuda.transform.LowerLDGSTG", allow_missing=True) is None,
    reason="LowerLDGSTG is not compiled into this build",
)


def _assert_global_access(func, direction, bits, buffer_index, predicate=None):
    """Check this pass's pointer, packed value and predicate contract."""
    op_name = f"tl.{direction}{bits}"
    assert_call_count(func, op=op_name, count=1)
    call = collect_calls(func, op=op_name)[0]
    is_load = direction == "ldg"
    assert len(call.args) == (1 if is_load else 2) + (predicate is not None)
    packed_dtype = "uint32" if bits == 32 else f"uint32x{bits // 32}"
    assert (call.dtype if is_load else call.args[1].dtype) == packed_dtype
    ptr = call.args[0]
    assert isinstance(ptr, tirx.Call) and ptr.op.same_as(tvm.ir.Op.get("tirx.tvm_access_ptr"))
    buffer = func.buffer_map[func.params[buffer_index]]
    assert ptr.args[1].same_as(buffer.data)
    assert ptr.args[4].value == (1 if is_load else 2)
    if predicate is not None:
        tvm.ir.assert_structural_equal(call.args[-1], predicate)
    return call


def _assert_no_global_intrinsics(func):
    for direction in ("ldg", "stg"):
        for bits in (32, 64, 128, 256):
            assert_call_count(func, op=f"tl.{direction}{bits}", count=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("enable_non_predicated", [False, True])
@pytest.mark.parametrize("enclosing_store", [False, True])
def test_nested_predicated_load_preserves_store_guard(lanes, enable_non_predicated, enclosing_store):
    """A nested load must retain both predicates, for scalar and vector widths."""
    iterator = T.serial if lanes == 1 else T.vectorized

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), outer: T.int32, inner: T.int32):
        for i in T.thread_binding(128 // lanes, "threadIdx.x"):
            for j in iterator(lanes):
                if enclosing_store:
                    with T.If(outer > 0), T.Then():
                        B[i * lanes + j] = T.if_then_else(inner > 0, A[i * lanes + j], T.float32(0))
                else:
                    B[i * lanes + j] = T.if_then_else(inner > 0, A[i * lanes + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = _apply_passes(mod, enable_non_predicated=enable_non_predicated, enable_predicated=True)
    loads, stores = [], []

    def visitor(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
            if node.op.name == f"tl.ldg{32 * lanes}":
                loads.append(node)
            elif node.op.name == f"tl.stg{32 * lanes}":
                stores.append(node)

    tirx.stmt_functor.post_order_visit(lowered["main"].body, visitor)
    assert len(loads) == 1
    outer, inner = func.params[2:]
    expected = tirx.And(outer > 0, inner > 0) if enclosing_store else inner > 0
    tvm.ir.assert_structural_equal(loads[0].args[-1], expected)
    if enclosing_store:
        assert len(stores) == 1
        tvm.ir.assert_structural_equal(stores[0].args[-1], outer > 0)


@tilelang.testing.requires_cuda
def test_nested_predicated_load_does_not_leak_store_guard():
    """A subsequent load uses its own condition, not a preceding store guard."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), outer: T.int32, inner: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            with T.If(outer > 0), T.Then():
                B[i] = T.if_then_else(inner > 0, A[i], T.float32(0))
            B[i] = T.if_then_else(inner > 0, A[i], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = _apply_passes(mod, enable_predicated=True)
    predicates = []

    def visitor(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name") and node.op.name == "tl.ldg32":
            predicates.append(node.args[-1])

    tirx.stmt_functor.post_order_visit(lowered["main"].body, visitor)
    assert len(predicates) == 2
    outer, inner = func.params[2:]
    tvm.ir.assert_structural_equal(predicates[0], tirx.And(outer > 0, inner > 0))
    tvm.ir.assert_structural_equal(predicates[1], inner > 0)


@requires_ldgstg
def test_lower_ldg32_default_off():
    """Test that non-predicated ldg/stg lowering is OFF by default."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(128, "threadIdx.x"):
            B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod)  # Default: enable_non_predicated=False
    print("=== test_lower_ldg32_default_off ===")
    print(mod)
    # By default, non-predicated lowering is OFF
    _assert_no_global_intrinsics(mod["main"])


@requires_ldgstg
def test_lower_ldg32_enabled():
    """Test that ldg32/stg32 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(128, "threadIdx.x"):
            B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_non_predicated=True)
    print("=== test_lower_ldg32_enabled ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 32, 0)
    _assert_global_access(mod["main"], "stg", 32, 1)


@requires_ldgstg
def test_lower_ldg64_enabled():
    """Test that ldg64/stg64 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(64, "threadIdx.x"):
            for j in T.vectorized(2):
                B[i * 2 + j] = A[i * 2 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_non_predicated=True)
    print("=== test_lower_ldg64_enabled ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 64, 0)
    _assert_global_access(mod["main"], "stg", 64, 1)


@requires_ldgstg
def test_lower_ldg128_enabled():
    """Test that ldg128/stg128 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_non_predicated=True)
    print("=== test_lower_ldg128_enabled ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 128, 0)
    _assert_global_access(mod["main"], "stg", 128, 1)


@requires_ldgstg
def test_lower_ldg256_enabled():
    """Test that ldg256/stg256 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(8):
                B[i * 8 + j] = A[i * 8 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_non_predicated=True)
    print("=== test_lower_ldg256_enabled ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 256, 0)
    _assert_global_access(mod["main"], "stg", 256, 1)


@requires_ldgstg
def test_lower_ldg32_predicated():
    """Test predicated ldg32 for single element load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            B[i] = T.if_then_else(pred > 0, A[i], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_lower_ldg32_predicated ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 32, 0, predicate=mod["main"].params[-1] > 0)


@requires_ldgstg
def test_lower_stg32_predicated():
    """Test predicated stg32 for single element store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            with T.If(pred > 0), T.Then():
                B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_lower_stg32_predicated ===")
    print(mod)
    _assert_global_access(mod["main"], "stg", 32, 1, predicate=mod["main"].params[-1] > 0)


@requires_ldgstg
def test_lower_ldg128_predicated():
    """Test predicated ldg128 for vectorized load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                B[i * 4 + j] = T.if_then_else(pred > 0, A[i * 4 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_lower_ldg128_predicated ===")
    print(mod)
    _assert_global_access(mod["main"], "ldg", 128, 0, predicate=mod["main"].params[-1] > 0)


@requires_ldgstg
def test_lower_stg128_predicated():
    """Test predicated stg128 for vectorized store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_lower_stg128_predicated ===")
    print(mod)
    _assert_global_access(mod["main"], "stg", 128, 1, predicate=mod["main"].params[-1] > 0)


@requires_ldgstg
def test_predicated_store_with_load():
    """Test that when a predicated store contains a load, the load also gets predicated.

    This tests the pattern: if (pred) { B[i] = A[i] }
    Both the store and the load should use predicated versions to avoid
    out-of-bounds memory access when pred is false.
    """

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_predicated_store_with_load ===")
    print(mod)
    # Both load and store should be predicated
    load = _assert_global_access(mod["main"], "ldg", 128, 0, predicate=mod["main"].params[-1] > 0)
    store = _assert_global_access(mod["main"], "stg", 128, 1, predicate=mod["main"].params[-1] > 0)
    stored_loads = collect_calls(store.args[1], op="tl.ldg128")
    assert len(stored_loads) == 1 and stored_loads[0].same_as(load)


@requires_ldgstg
def test_predicated_store_with_shared_load_keeps_explicit_guard():
    """Do not hoist shared-memory loads out of a predicated global store."""

    @T.prim_func
    def func(B: T.Buffer((128,), "float32"), pred: T.int32):
        S = T.alloc_buffer((128,), dtype=T.float32, scope="shared")
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = S[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)
    print("=== test_predicated_store_with_shared_load_keeps_explicit_guard ===")
    print(mod)

    func = mod["main"]
    guards = collect_nodes(func, tirx.IfThenElse)
    assert len(guards) == 1
    guard = guards[0]
    tvm.ir.assert_structural_equal(guard.condition, func.params[-1] > 0)
    shared_loads = [load for load in collect_nodes(func, tirx.BufferLoad) if load.buffer.scope() == "shared"]
    assert shared_loads, "Expected a shared-memory load"
    guarded_loads = collect_nodes(guard.then_case, tirx.BufferLoad)
    assert all(any(load.same_as(guarded) for guarded in guarded_loads) for load in shared_loads)
    stores = collect_nodes(func, tirx.BufferStore)
    guarded_stores = collect_nodes(guard.then_case, tirx.BufferStore)
    assert len(stores) == len(guarded_stores) == 1
    assert stores[0].same_as(guarded_stores[0])
    # FlattenBuffer can create a new Buffer handle for the same storage.
    assert stores[0].buffer.data.same_as(func.buffer_map[func.params[0]].data)
    _assert_no_global_intrinsics(mod["main"])


@requires_ldgstg
def test_predicated_disabled():
    """Test that predicated lowering can be disabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), N: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                idx = i * 4 + j
                B[idx] = T.if_then_else(idx < N, A[idx], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=False)
    _assert_no_global_intrinsics(mod["main"])
    assert collect_nodes(mod["main"], tirx.BufferLoad)
    assert collect_nodes(mod["main"], tirx.BufferStore)


@requires_ldgstg
def test_predicated_option_controls_eligible_load():
    """Disabling the option must change an otherwise eligible lowering."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                B[i * 4 + j] = T.if_then_else(pred > 0, A[i * 4 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    # Establish that this same input is eligible when the option is enabled.
    enabled = _apply_passes(mod, enable_predicated=True)
    _assert_global_access(enabled["main"], "ldg", 128, 0, predicate=enabled["main"].params[-1] > 0)
    mod = _apply_passes(mod, enable_predicated=False)
    # Both lowering options are disabled; the original memory operations remain.
    _assert_no_global_intrinsics(mod["main"])
    assert collect_nodes(mod["main"], tirx.BufferLoad)
    assert collect_nodes(mod["main"], tirx.BufferStore)


@requires_ldgstg
def test_non_cuda_target_skip():
    """Test that the pass is skipped for non-CUDA targets."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                B[i * 4 + j] = A[i * 4 + j]

    # Use a CPU target
    cpu_target = tvm.target.Target("llvm")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(cpu_target)(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True}):
        mod = tl.cuda.transform.LowerLDGSTG()(mod)
    print("=== test_non_cuda_target_skip ===")
    print(mod)
    # The load should NOT be lowered to ldg because target is not CUDA
    _assert_no_global_intrinsics(mod["main"])


@tilelang.testing.requires_cuda
def test_e2e_load_global_store_global():
    """End-to-end test that ldg/stg intrinsics work correctly when enabled."""
    import torch

    @tilelang.jit(pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True})
    def copy_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            for j in T.vectorized(4):
                Y[pid * 4 + j] = X[pid * 4 + j]

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    copy_kernel(X, Y)

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)

    # Verify codegen contains ldg/stg
    src = copy_kernel.get_kernel_source(N=128)
    print("=== Generated kernel source ===")
    print(src)
    assert "load_global_128" in src or "store_global_128" in src, "Expected load_global_128/store_global_128 in generated source"


@tilelang.testing.requires_cuda
def test_e2e_load_global_store_global_predicated():
    """End-to-end test that load_global/store_global intrinsics work correctly when enabled."""
    import torch

    @tilelang.jit(pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True, PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True})
    def copy_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            for j in T.vectorized(4):
                Y[pid * 4 + j] = T.if_then_else(pid < N // 8, X[pid * 4 + j], T.float32(0))

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    copy_kernel(X, Y)

    # Verify correctness
    Y_ref = torch.zeros(128, dtype=torch.float32, device="cuda")
    for i in range(128):
        if i < 64:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)

    # Verify codegen contains load_global/store_global
    src = copy_kernel.get_kernel_source(N=128)
    print("=== Generated kernel source ===")
    print(src)
    assert "load_global_128_conditional" in src or "store_global_128_conditional" in src, (
        "Expected load_global_128_conditional/store_global_128_conditional in generated source"
    )


if __name__ == "__main__":
    tilelang.testing.main()
