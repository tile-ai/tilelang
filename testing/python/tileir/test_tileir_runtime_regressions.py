"""Numerical regressions for packed arguments, indexing, and persistent loops."""

import pytest

import tilelang
import tilelang.testing
from tilelang.tileir import language as T
from tileir_test_utils import _enable_tileir_runtime, _skip_if_tileir_toolchain_unavailable


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("shape", [(16, 64), (32, 128), (64, 256)])
def test_packed_fp4_dynamic_input_and_outputs(monkeypatch, shape):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("packed FP4 requires torch >= 2.8")

    @tilelang.jit(out_idx=[1, 2], target="tileir", execution_backend="tileir")
    def kernel():
        m, n = T.dynamic("m, n")

        @T.prim_func
        def main(A: T.Tensor((m, n), T.float4_e2m1fn), B: T.Tensor((m, n), T.float32), C: T.Tensor((m, n), T.float4_e2m1fn)):
            with T.Kernel(T.ceildiv(m, 16), T.ceildiv(n, 64), threads=128) as (bx, by):
                packed = T.alloc_shared((16, 64), T.float4_e2m1fn)
                decoded = T.alloc_fragment((16, 64), T.float32)
                T.copy(A[bx * 16, by * 64], packed)
                T.copy(packed, decoded)
                T.copy(decoded, B[bx * 16, by * 64])
                T.copy(packed, C[bx * 16, by * 64])

        return main

    storage = torch.arange(shape[0] * shape[1] // 2, device="cuda").to(torch.uint8).reshape(shape[0], shape[1] // 2)
    packed_input = storage.view(torch.float4_e2m1fn_x2)
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    bits = torch.stack((storage & 15, storage >> 4), dim=-1).reshape(shape).long()
    expected = magnitudes[bits & 7] * torch.where(bits < 8, 1, -1)
    compiled = kernel()
    decoded, copied = compiled(packed_input)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert copied.shape == packed_input.shape
    torch.testing.assert_close(copied.view(torch.uint8), storage, rtol=0, atol=0)
    explicit_decoded = torch.empty_like(decoded)
    explicit_copied = torch.empty_like(copied)
    compiled(packed_input, explicit_decoded, explicit_copied)
    torch.testing.assert_close(explicit_decoded, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit_copied.view(torch.uint8), storage, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_flattened_offset_slice_and_scalar_scale(monkeypatch):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)

    @tilelang.jit(out_idx=-1, target="tileir", execution_backend="tileir")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((24,), T.float32), S: T.Tensor((3,), T.float32), B: T.Tensor((16,), T.float32)):
            with T.Kernel(1, threads=96):
                fragment = T.alloc_fragment((24,), T.float32)
                shared = T.alloc_shared((24,), T.float32)
                for i in T.Parallel(24):
                    fragment[i] = 0
                    for _repeat in T.serial(2):
                        fragment[i] += A[i]
                T.copy(fragment, shared)
                for j, k in T.Parallel(4, 4):
                    B[j * 4 + k] = shared[j * 4 + k + 8] * S[2] + S[0] - S[1]

        return main

    a = torch.arange(24, device="cuda", dtype=torch.float32)
    scales = torch.tensor([11, -3, 2], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel()(a, scales), 2 * a[8:24] * scales[2] + scales[0] - scales[1], rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_while_mutable_state_nested_loops_and_early_exit(monkeypatch):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)

    @tilelang.jit(out_idx=-1, target="tileir", execution_backend="tileir")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((4, 32), T.float32), Limits: T.Tensor((4,), T.int32), B: T.Tensor((4, 32), T.float32)):
            with T.Kernel(4, threads=128) as bx:
                count = T.alloc_var(T.int32)
                count = 0
                accum = T.alloc_fragment((32,), T.float32)
                loaded = T.alloc_shared((32,), T.float32)
                T.clear(accum)
                while count < Limits[bx]:
                    count += 1
                    if count == 2:
                        continue
                    if count == 4:
                        T.loop_break()
                    for _inner in T.serial(2):
                        T.copy(A[bx, :], loaded)
                        for i in T.Parallel(32):
                            accum[i] += loaded[i]
                for i in T.Parallel(32):
                    B[bx, i] = accum[i] + T.Cast(T.float32, count)

        return main

    a = torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 32)
    limits = torch.tensor([0, 1, 3, 8], device="cuda", dtype=torch.int32)
    factors = torch.tensor([0, 2, 4, 4], device="cuda").reshape(4, 1)
    counts = torch.tensor([0, 1, 3, 4], device="cuda").reshape(4, 1)
    torch.testing.assert_close(kernel()(a, limits), a * factors + counts, rtol=0, atol=0)
