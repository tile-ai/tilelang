"""
Tests for HIP/AMD codegen fixes in TileLang.

Covers three fixes made to src/target/codegen_hip.cc:
  1. T.sync_warp() is lowered to a no-op on HIP (AMD wavefronts execute in
     lockstep so no explicit reconvergence barrier is needed).
  2. T.alloc_var(dtype, init=value) emits a properly initialised scalar
     declaration on HIP (previously the init value was silently dropped).
  3. local.var buffers are accessed as plain scalars in GetBufferRef (no [0]
     subscript), consistent with the scalar declaration emitted for them.
"""

import torch
import tilelang
import tilelang.testing
import tilelang.language as T


# ---------------------------------------------------------------------------
# Fix 1: T.sync_warp() → no-op on HIP
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_sync_warp_codegen():
    """Minimal kernel that exercises T.sync_warp()."""

    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((32,), "float32")
            A_shared[tx] = A[tx]
            T.sync_warp()
            B[tx] = A_shared[tx] * 2.0

    return main


@tilelang.testing.requires_rocm
def test_sync_warp_no_syncwarp_in_hip_source():
    """__syncwarp must NOT appear in the HIP-generated kernel source."""
    kernel = _kernel_sync_warp_codegen()
    src = kernel.get_kernel_source()
    assert "__syncwarp" not in src, (
        "T.sync_warp() should be a no-op on HIP (AMD wavefronts are lockstep), "
        f"but __syncwarp was found in the generated source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_sync_warp_correctness():
    """Kernel using T.sync_warp() should produce correct results on HIP."""
    kernel = _kernel_sync_warp_codegen()
    A = torch.arange(32, dtype=torch.float32, device="cuda")
    B = torch.zeros(32, dtype=torch.float32, device="cuda")
    kernel(A, B)
    torch.testing.assert_close(B, A * 2.0)


# ---------------------------------------------------------------------------
# Fix 2: T.alloc_var(init=...) initialisation on HIP
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_init():
    """Kernel that initialises a local int32 variable and writes it to output."""

    @T.prim_func
    def main(Out: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            counter = T.alloc_var(T.int32, init=7)
            Out[tx] = counter

    return main


@tilelang.testing.requires_rocm
def test_alloc_var_init_in_hip_source():
    """Init value must appear in the generated HIP source for T.alloc_var."""
    kernel = _kernel_alloc_var_init()
    src = kernel.get_kernel_source()
    assert "= 7;" in src, (
        "T.alloc_var(T.int32, init=7) should generate '= 7;' in HIP source, "
        f"but it was not found.\nGenerated source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_no_array_subscript_in_hip_source():
    """local.var should be declared as a scalar, not as an array (no [0])."""
    kernel = _kernel_alloc_var_init()
    src = kernel.get_kernel_source()
    # The kernel source should contain 'counter = 7' not 'counter[1]' or 'counter[0]'
    assert "counter[" not in src, (
        "local.var should be emitted as a scalar (e.g. 'int counter = 7'), "
        f"but array-style access was found in the HIP source:\n{src}"
    )


@tilelang.testing.requires_rocm
def test_alloc_var_init_correctness():
    """Kernel should read back the initialised value correctly on HIP."""
    kernel = _kernel_alloc_var_init()
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 7), (
        f"Expected all 7, got: {out}"
    )


# ---------------------------------------------------------------------------
# Fix 2b: multiple T.alloc_var with distinct init values
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_multi_alloc_var_init():
    """Two local variables with different init values, summed into output."""

    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            a = T.alloc_var(T.int32, init=3)
            b = T.alloc_var(T.int32, init=4)
            Out[tx] = a + b

    return main


@tilelang.testing.requires_rocm
def test_multi_alloc_var_init_in_hip_source():
    """Both init values must appear in the HIP source."""
    kernel = _kernel_multi_alloc_var_init()
    src = kernel.get_kernel_source()
    assert src.count("= 3;") >= 1, f"Init value 3 not found in HIP source:\n{src}"
    assert src.count("= 4;") >= 1, f"Init value 4 not found in HIP source:\n{src}"


@tilelang.testing.requires_rocm
def test_multi_alloc_var_init_correctness():
    """Sum of two initialised local variables should equal 7 on HIP."""
    kernel = _kernel_multi_alloc_var_init()
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 7), f"Expected all 7 (3+4), got: {out}"


# ---------------------------------------------------------------------------
# Fix 2c: T.alloc_var(init=0) — the default zero-init case
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_alloc_var_count():
    """
    Accumulates a count by incrementing a local variable in a loop.
    Relies on the variable being zero-initialised (init=0 default).
    """

    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            count = T.alloc_var(T.int32, init=0)
            for _ in T.unroll(5):
                count += 1
            Out[tx] = count

    return main


@tilelang.testing.requires_rocm
def test_alloc_var_zero_init_correctness():
    """Variable initialised to 0, incremented 5 times, should equal 5."""
    kernel = _kernel_alloc_var_count()
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    kernel(out)
    assert torch.all(out == 5), f"Expected all 5, got: {out}"


# ---------------------------------------------------------------------------
# Fix 3: T.sync_grid() codegen on HIP (codegen only; runtime not yet complete)
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def _kernel_sync_grid_codegen():
    """Kernel that calls T.sync_grid() to trigger cooperative groups codegen."""

    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.sync_grid()
            A[tx] = T.float32(tx)

    return main


@tilelang.testing.requires_rocm
def test_sync_grid_cooperative_groups_in_hip_source():
    """
    T.sync_grid() should generate cooperative_groups::this_grid().sync()
    and include <hip/hip_cooperative_groups.h> in the HIP source.

    Note: runtime execution of this kernel is not yet supported because
    rocm_module.cc does not yet call hipModuleLaunchCooperativeKernel.
    This test validates codegen only.
    """
    kernel = _kernel_sync_grid_codegen()
    src = kernel.get_kernel_source()
    assert "this_grid().sync()" in src, (
        "T.sync_grid() should generate 'this_grid().sync()' in HIP source, "
        f"but it was not found:\n{src}"
    )
    assert "cooperative_groups" in src, (
        "T.sync_grid() should include cooperative_groups in HIP source, "
        f"but it was not found:\n{src}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
