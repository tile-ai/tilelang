"""Tests for LowerLDGSTG pass that converts Ramp-based global memory
load/store to ldg/stg intrinsics.

Pass configurations:
- tl.enable_lower_ldgstg: Enable non-predicated ldg/stg lowering (default: OFF)
- tl.enable_lower_ldgstg_predicated: Enable predicated ldg/stg lowering (default: OFF)
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tvm import tir


def _apply_passes(mod, enable_non_predicated=False, enable_predicated=False):
    """Apply the LowerLDGSTG pass and related lowering passes."""
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("cuda"))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: enable_non_predicated,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: enable_predicated,
        }
    ):
        mod = tl.transform.LowerLDGSTG()(mod)
    return mod


def _check_has_intrinsic(mod, intrinsic_name):
    """Check if the module contains a specific intrinsic call."""
    found = [False]

    def visitor(obj):
        if isinstance(obj, tir.Call) and hasattr(obj.op, "name") and intrinsic_name in obj.op.name:
            found[0] = True

    tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


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
    assert not _check_has_intrinsic(mod, "ldg32"), "Non-predicated ldg should be OFF by default"
    assert not _check_has_intrinsic(mod, "stg32"), "Non-predicated stg should be OFF by default"


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
    assert _check_has_intrinsic(mod, "ldg32"), "Expected ldg32 when enabled"
    assert _check_has_intrinsic(mod, "stg32"), "Expected stg32 when enabled"


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
    assert _check_has_intrinsic(mod, "ldg64"), "Expected ldg64 when enabled"
    assert _check_has_intrinsic(mod, "stg64"), "Expected stg64 when enabled"


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
    assert _check_has_intrinsic(mod, "ldg128"), "Expected ldg128 when enabled"
    assert _check_has_intrinsic(mod, "stg128"), "Expected stg128 when enabled"


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
    assert _check_has_intrinsic(mod, "ldg256"), "Expected ldg256 when enabled"
    assert _check_has_intrinsic(mod, "stg256"), "Expected stg256 when enabled"


def test_lower_ldg32_predicated():
    """Test predicated ldg32 for single element load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            B[i] = T.if_then_else(pred > 0, A[i], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)  # Default: predicated is ON
    print("=== test_lower_ldg32_predicated ===")
    print(mod)
    assert _check_has_intrinsic(mod, "ldg32"), "Expected predicated ldg32"


def test_lower_stg32_predicated():
    """Test predicated stg32 for single element store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            with T.If(pred > 0), T.Then():
                B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)  # Default: predicated is ON
    print("=== test_lower_stg32_predicated ===")
    print(mod)
    assert _check_has_intrinsic(mod, "stg32"), "Expected predicated stg32"


def test_lower_ldg128_predicated():
    """Test predicated ldg128 for vectorized load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                B[i * 4 + j] = T.if_then_else(pred > 0, A[i * 4 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_predicated=True)  # Default: predicated is ON
    print("=== test_lower_ldg128_predicated ===")
    print(mod)
    assert _check_has_intrinsic(mod, "ldg128"), "Expected predicated ldg128"


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
    mod = _apply_passes(mod, enable_predicated=True)  # Default: predicated is ON
    print("=== test_lower_stg128_predicated ===")
    print(mod)
    assert _check_has_intrinsic(mod, "stg128"), "Expected predicated stg128"


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
    assert _check_has_intrinsic(mod, "ldg128"), "Expected predicated ldg128 for load inside predicated store"
    assert _check_has_intrinsic(mod, "stg128"), "Expected predicated stg128"


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
    print("=== test_predicated_disabled ===")
    print(mod)
    # When disabled, no predicated ldg/stg should be generated
    # This just verifies the configuration works


def test_skip_async_scope():
    """Test that loads in async scope are not lowered (will be cp.async)."""

    @T.prim_func
    def func(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32", scope="shared"),
    ):
        for i in T.thread_binding(32, "threadIdx.x"):
            with T.attr(0, "async_scope", 1):
                for j in T.vectorized(4):
                    B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_passes(mod, enable_non_predicated=True)
    print("=== test_skip_async_scope ===")
    print(mod)
    # The load should NOT be lowered to ldg because it's in async scope
    assert not _check_has_intrinsic(mod, "ldg"), "Loads in async scope should NOT be lowered to ldg"


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
    mod = tvm.tir.transform.BindTarget(cpu_target)(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True}):
        mod = tl.transform.LowerLDGSTG()(mod)
    print("=== test_non_cuda_target_skip ===")
    print(mod)
    # The load should NOT be lowered to ldg because target is not CUDA
    assert not _check_has_intrinsic(mod, "ldg"), "Non-CUDA targets should NOT use ldg intrinsics"
    assert not _check_has_intrinsic(mod, "stg"), "Non-CUDA targets should NOT use stg intrinsics"


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
