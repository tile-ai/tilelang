"""Argument binding of the torch Metal adapter.

Host/device splitting orders device parameters by its own rules and drops
unused ones, so the MSL buffer order differs from the declared signature. The
adapter must bind every MSL slot from the host call site, keep the declared
call order, pack runtime scalars like the code generator lays them out, honor
storage offsets, launch multi-kernel programs in program order, and reject
signatures it cannot represent instead of launching them.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.jit.adapter.torch.metal import MetalLaunch, MetalLaunchPlanError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="PyTorch MPS device is required")


def compile_metal(func, out_idx=None):
    return tilelang.compile(func, out_idx=out_idx, target="metal", execution_backend="torch")


def affine(size: int):
    @T.prim_func
    def main(Z: T.Tensor((size,), "float32"), A: T.Tensor((size,), "float32")):
        with T.Kernel(T.ceildiv(size, 64), threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            if i < size:
                A[i] = Z[i] * 2 + 1

    return main


def with_unused(size: int):
    @T.prim_func
    def main(
        A: T.Tensor((size,), "float32"),
        B: T.Tensor((size,), "float32"),
        C: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(1, threads=size):
            i = T.get_thread_binding(0)
            C[i] = A[i] + 2

    return main


def scaled(size: int):
    @T.prim_func
    def main(
        A: T.Tensor((size,), "float32"),
        S: T.int32,
        F: T.float32,
        B: T.Tensor((size,), "float32"),
        L: T.int64,
    ):
        with T.Kernel(1, threads=size):
            i = T.get_thread_binding(0)
            # Scale the int64 down so its contribution survives float32 rounding
            # while the runtime value itself stays above 32 bits.
            B[i] = A[i] * T.cast(S, "float32") + F + T.cast(L // (1 << 30), "float32")

    return main


def two_stages(size: int):
    @T.prim_func
    def main(
        Y: T.Tensor((size,), "float32"),
        X: T.Tensor((size,), "float32"),
        M: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(1, threads=size):
            i = T.get_thread_binding(0)
            M[i] = X[i] + 1
        with T.Kernel(2, threads=size // 2):
            j = T.get_thread_binding(0) + T.get_block_binding(0) * (size // 2)
            Y[j] = M[j] * 3

    return main


def test_reordered_parameters_bind_by_declared_order():
    size = 129
    kernel = compile_metal(affine(size))
    assert [launch.buffers for launch in kernel.adapter.launches] == [(1, 0)]
    source = torch.arange(size, dtype=torch.float32, device="mps")
    output = torch.full((size,), -1.0, device="mps")
    kernel(source, output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), torch.arange(size, dtype=torch.float32) * 2 + 1)
    torch.testing.assert_close(source.cpu(), torch.arange(size, dtype=torch.float32))


def test_unused_parameter_is_not_bound_and_stays_untouched():
    size = 64
    kernel = compile_metal(with_unused(size))
    assert kernel.adapter.launches[0].buffers == (0, 2)
    a = torch.arange(size, dtype=torch.float32, device="mps")
    b = torch.full((size,), -7.0, device="mps")
    c = torch.zeros(size, device="mps")
    kernel(a, b, c)
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), torch.arange(size, dtype=torch.float32) + 2)
    assert bool((b.cpu() == -7).all())


def test_storage_offsets_are_honored_and_guards_preserved():
    size = 129
    kernel = compile_metal(affine(size))
    backing = torch.arange(512, dtype=torch.float32, device="mps")
    output = torch.full((512,), -99.0, device="mps")
    kernel(backing[13 : 13 + size], output[29 : 29 + size])
    torch.mps.synchronize()
    expected = torch.arange(13, 13 + size, dtype=torch.float32) * 2 + 1
    torch.testing.assert_close(output[29 : 29 + size].cpu(), expected)
    assert bool((output[:29].cpu() == -99).all()) and bool((output[29 + size :].cpu() == -99).all())
    torch.testing.assert_close(backing.cpu(), torch.arange(512, dtype=torch.float32))


def test_runtime_scalars_are_packed_in_struct_order():
    size = 32
    kernel = compile_metal(scaled(size))
    (launch,) = kernel.adapter.launches
    assert launch.buffers == (0, 3)
    # Struct member order is the device parameter order, not the declared one.
    assert launch.scalars == (2, 4, 1)
    a = torch.arange(size, dtype=torch.float32, device="mps")
    b = torch.zeros(size, device="mps")
    big = (1 << 33) + (5 << 30)
    kernel(a, 3, 0.5, b, big)
    torch.mps.synchronize()
    torch.testing.assert_close(b.cpu(), torch.arange(size, dtype=torch.float32) * 3 + 0.5 + float(big >> 30))


def test_scalar_output_is_rejected():
    with pytest.raises(MetalLaunchPlanError):
        compile_metal(scaled(32), out_idx=[1])


def test_multi_kernel_program_launches_in_order_with_per_kernel_bindings():
    size = 64
    kernel = compile_metal(two_stages(size))
    launches = kernel.adapter.launches
    assert len(launches) == 2
    assert launches[0].grid == (1, 1, 1) and launches[0].block == (size, 1, 1)
    assert launches[1].grid == (2, 1, 1) and launches[1].block == (size // 2, 1, 1)
    assert set(launches[0].buffers) == {1, 2} and set(launches[1].buffers) == {0, 2}
    x = torch.arange(size, dtype=torch.float32, device="mps")
    y = torch.zeros(size, device="mps")
    m = torch.zeros(size, device="mps")
    kernel(y, x, m)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), (torch.arange(size, dtype=torch.float32) + 1) * 3)


def test_out_idx_allocates_static_outputs():
    size = 64
    kernel = compile_metal(with_unused(size), out_idx=[2])
    a = torch.arange(size, dtype=torch.float32, device="mps")
    result = kernel(a, torch.zeros(size, device="mps"))
    torch.mps.synchronize()
    assert result.device.type == "mps" and result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), torch.arange(size, dtype=torch.float32) + 2)


def test_jit_decorator_uses_the_same_binding():
    @tilelang.jit(target="metal", execution_backend="torch")
    def build(size):
        return affine(size)

    kernel = build(70)
    source = torch.arange(70, dtype=torch.float32, device="mps")
    output = torch.zeros(70, device="mps")
    kernel(source, output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), torch.arange(70, dtype=torch.float32) * 2 + 1)


def test_argument_validation_rejects_wrong_count_device_dtype_and_shape():
    size = 64
    kernel = compile_metal(affine(size))
    ok = torch.zeros(size, device="mps")
    with pytest.raises(TypeError, match="expects 2 arguments"):
        kernel(ok)
    with pytest.raises(ValueError, match="mps device"):
        kernel(torch.zeros(size), ok)
    with pytest.raises(TypeError, match="dtype"):
        kernel(torch.zeros(size, dtype=torch.float16, device="mps"), ok)
    with pytest.raises(ValueError, match="shape"):
        kernel(torch.zeros(size + 1, device="mps"), ok)
    with pytest.raises(ValueError, match="contiguous"):
        kernel(torch.zeros(size * 2, device="mps")[::2], ok)
    with pytest.raises(TypeError, match="scalar"):
        compile_metal(scaled(32))(ok[:32], ok[:32], 0.5, ok[:32], 1)


def test_pipeline_limits_are_exposed_and_enforced():
    kernel = compile_metal(affine(64))
    adapter = kernel.adapter
    assert adapter.thread_execution_width in (32, 64)
    assert adapter.max_total_threads_per_threadgroup >= 64

    @T.prim_func
    def oversized(A: T.Tensor((2048,), "float32"), B: T.Tensor((2048,), "float32")):
        with T.Kernel(1, threads=2048):
            i = T.get_thread_binding(0)
            B[i] = A[i]

    # Pipeline creation itself rejects the declared threadgroup size on every
    # Apple GPU; the adapter's own limit check covers pipelines that compile.
    with pytest.raises((MetalLaunchPlanError, RuntimeError), match="threads per threadgroup"):
        compile_metal(oversized)


def test_dynamic_shape_signature_is_rejected_clearly():
    n = T.dynamic("n")

    @T.prim_func
    def dynamic(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 64), threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            if i < n:
                B[i] = A[i]

    with pytest.raises(MetalLaunchPlanError):
        compile_metal(dynamic)


def test_launch_plan_is_plain_data():
    launch = compile_metal(affine(64)).adapter.launches[0]
    assert launch == MetalLaunch(symbol="main_kernel", buffers=(1, 0), scalars=(), grid=(1, 1, 1), block=(64, 1, 1))
    assert launch.threads == (64, 1, 1)


if __name__ == "__main__":
    tilelang.testing.main()
