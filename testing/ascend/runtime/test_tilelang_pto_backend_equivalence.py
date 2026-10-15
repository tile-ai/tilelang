"""Bitwise equivalence of the tvm_ffi and legacy pto execution backends.

The same program is compiled twice with ``target="pto"`` — once with
``execution_backend="tvm_ffi"`` and once with the legacy Cython launcher
(``execution_backend="pto"``) — and both run on identical inputs.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytestmark = pytest.mark.pto


def vector_add(n: int, tile: int):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        c: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // tile) as bx, T.SimtVF(threads=2048):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                c[idx] = a[idx] + b[idx]

    return main


def dynamic_shared_copy():
    n = T.dynamic("n")

    @T.prim_func
    def copy(src: T.Tensor((n,), T.float32), dst: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((T.ceildiv(n, 64) * 64,), T.float32)
            T.copy(src, ub[:n])
            T.copy(ub[:n], dst)

    return copy


def two_kernels(n: int, tile: int):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        tmp: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // tile) as bx, T.SimtVF(threads=2048):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                tmp[idx] = a[idx] + b[idx]

        with T.Kernel(n // tile) as bx, T.SimtVF(threads=2048):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                out[idx] = tmp[idx] * T.float32(2.0)

    return main


def test_pto_backend_equivalence_vecadd(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    n, tile = 64 * 2048, 2048
    func = vector_add(n, tile)
    ffi_kernel = tilelang.compile(func, target="pto", out_idx=-1, execution_backend="tvm_ffi")
    legacy_kernel = tilelang.compile(func, target="pto", out_idx=-1, execution_backend="pto")

    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    out_ffi = ffi_kernel(a, b)
    out_legacy = legacy_kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(out_ffi.cpu(), out_legacy.cpu())


def test_pto_backend_equivalence_dynamic_shared_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    func = dynamic_shared_copy()
    ffi_kernel = tilelang.compile(func, target="pto", out_idx=-1, execution_backend="tvm_ffi")
    legacy_kernel = tilelang.compile(func, target="pto", out_idx=-1, execution_backend="pto")

    for size in (1, 65, 1000):
        source = torch.arange(size, dtype=torch.float32, device="npu")
        out_ffi = ffi_kernel(source)
        out_legacy = legacy_kernel(source)
        torch.npu.synchronize()
        assert torch.equal(out_ffi.cpu(), out_legacy.cpu())


def test_pto_backend_equivalence_multiple_kernels(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    n, tile = 64 * 2048, 2048
    func = two_kernels(n, tile)
    ffi_kernel = tilelang.compile(func, target="pto", out_idx=[2, 3], execution_backend="tvm_ffi")
    legacy_kernel = tilelang.compile(func, target="pto", out_idx=[2, 3], execution_backend="pto")

    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    tmp_ffi, out_ffi = ffi_kernel(a, b)
    tmp_legacy, out_legacy = legacy_kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(tmp_ffi.cpu(), tmp_legacy.cpu())
    assert torch.equal(out_ffi.cpu(), out_legacy.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
