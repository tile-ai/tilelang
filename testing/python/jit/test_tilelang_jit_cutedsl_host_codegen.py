"""Tests for CuTeDSL host codegen integration."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    """Skip when the CuTeDSL Python stack is unavailable."""
    try:
        from tilelang.cuda.cutedsl_backend import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@pytest.mark.parametrize("out_idx", [None, -1])
@pytest.mark.parametrize("compiler", ["nvcc", "nvrtc", "cutedsl"])
@tilelang.testing.requires_cuda
def test_cutedsl_tvm_ffi_host_execution(out_idx, compiler, tmp_path):
    if compiler == "cutedsl":
        _require_cutedsl()
    N = T.dynamic("N")
    M = T.dynamic("N")  # Distinct Vars must not be bound by their printed name.

    @T.prim_func
    def main(
        A: T.Tensor((N * 2,), T.float32),
        shape: T.Tensor((M,), T.float32),
        scale: T.int32,
        repeats: T.int32,
        B: T.Tensor((N * 2,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N * 2, 128), threads=128) as bx:
            i = bx * 128 + T.get_thread_binding()
            if i < N * 2:
                B[i] = A[i]
        if scale > 0:
            for step in T.serial(repeats):
                offset = T.bind(scale * 3 + step + M)
                with T.Kernel(T.ceildiv(N * 2, 128), threads=128) as bx:
                    i = bx * 128 + T.get_thread_binding()
                    if i < N * 2:
                        B[i] = B[i] + T.float32(offset)

    kernel = tilelang.compile(
        main,
        target="cutedsl" if compiler == "cutedsl" else "cuda",
        out_idx=out_idx,
        execution_backend="tvm_ffi",
        pass_configs={"tl.cuda_compiler": "nvrtc"} if compiler == "nvrtc" else None,
    )
    assert kernel.execution_backend == "tvm_ffi"
    stream = torch.cuda.Stream()
    for size, scale, repeats in ((17, 3, 4), (128, -1, 3), (257, 7, 0)):
        with torch.cuda.stream(stream):
            a = torch.arange(size * 2, device="cuda", dtype=torch.float32)
            shape = torch.empty(3, device="cuda")
            if out_idx is None:
                b = torch.empty_like(a)
                assert kernel(a, shape, scale, repeats, b) == []
            else:
                b = kernel(a, shape, scale, repeats)
            expected = a + (repeats * (scale * 3 + 3) + repeats * (repeats - 1) / 2 if scale > 0 else 0)
        stream.synchronize()
        torch.testing.assert_close(b, expected)
    library = str(tmp_path / "kernel.so")
    kernel.export_library(library)
    loaded = tilelang.tvm.runtime.load_module(library)
    if out_idx is None:
        b.zero_()
        loaded(a, shape, 2, 1, b)
    else:
        b = loaded(a, shape, 2, 1, a)  # allocator anchor
    torch.testing.assert_close(b, a + 9)
    if out_idx is None:
        kernel(a, None, 2, 1, b)  # A nullable shape carrier binds its extent to zero.
    else:
        b = kernel(a, None, 2, 1)
    torch.testing.assert_close(b, a + 6)
    for invalid in (a[:-1], a.reshape(-1, 2), a.double(), a.repeat_interleave(2)[::2]):
        with pytest.raises((ValueError, RuntimeError)):
            if out_idx is None:
                kernel(invalid, shape, 2, 1, b)
            else:
                kernel(invalid, shape, 2, 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("feature", ["tma", "cluster", "cooperative"])
def test_cutedsl_tvm_ffi_device_extensions(feature, tmp_path):
    _require_cutedsl()
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90 or newer")
    cluster = (2, 1, 1) if feature == "cluster" else None

    @T.prim_func
    def main(A: T.Tensor((2, 128), T.float32), B: T.Tensor((2, 128), T.float32), C: T.Tensor((2, 128), T.float32)):
        with T.Kernel(2, threads=128, cluster_dims=cluster) as bx:
            i = T.get_thread_binding()
            if feature == "tma":
                shared = T.alloc_shared((1, 128), T.float32)
                barrier = T.alloc_barrier(128)
                T.tma_copy(A[bx : bx + 1, :], shared, barrier=barrier)
                T.barrier_arrive(barrier)
                T.barrier_wait(barrier, 0)
                T.tma_copy(shared, B[bx : bx + 1, :])
                T.tma_store_wait(0)
            else:
                B[bx, i] = A[bx, i]
                if feature == "cluster":
                    T.cluster_sync()
                else:
                    T.sync_grid()
                C[bx, i] = B[1 - bx, i]
        if feature == "tma":
            with T.Kernel(2, threads=128) as bx:
                i = T.get_thread_binding()
                C[bx, i] = B[bx, i]

    kernel = tilelang.compile(main, target="cutedsl", execution_backend="tvm_ffi", out_idx=[1, 2])
    for _ in range(2):
        a = torch.randn(2, 128, device="cuda")
        b, c = kernel(a)
        torch.testing.assert_close(b, a)
        torch.testing.assert_close(c, a if feature == "tma" else a.flip(0))
    library = str(tmp_path / "extensions.so")
    kernel.export_library(library)
    b, c = tilelang.tvm.runtime.load_module(library)(a, a)
    torch.testing.assert_close(b, a)
    torch.testing.assert_close(c, a if feature == "tma" else a.flip(0))


if __name__ == "__main__":
    tilelang.testing.main()
