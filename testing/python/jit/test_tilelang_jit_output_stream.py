"""Outputs created for a kernel launched on a caller-supplied stream must be
allocated on that stream (#3648).

The caching allocator orders reuse of a block only after earlier work on the
stream it was allocated on. If the output were allocated on the current stream,
it could be a block that still has queued work there, which then overwrites the
kernel's result."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

N = 1 << 20


def copy_kernel(n):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for i in T.Parallel(256):
                B[bx * 256 + i] = A[bx * 256 + i]

    return main


def _compile(backend):
    if backend == "nvrtc":
        from tilelang.jit.adapter.nvrtc import is_nvrtc_available

        if not is_nvrtc_available:
            pytest.skip("NVRTC is not available")
    if backend == "cutedsl":
        try:
            from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

            check_cutedsl_available()
        except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
            pytest.skip(f"CuTeDSL is not available: {err}")
    target = "cutedsl" if backend == "cutedsl" else "cuda"
    return tilelang.compile(copy_kernel(N), out_idx=[1], target=target, execution_backend=backend)


BACKENDS = ["cython", "nvrtc", "cutedsl"]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_output_is_not_reused_from_pending_current_stream_work(backend):
    kernel = _compile(backend)
    a = torch.randn(N, device="cuda")
    side = torch.cuda.Stream()
    for _ in range(3):
        torch.cuda.synchronize()
        pending = torch.empty(N, device="cuda")
        torch.cuda._sleep(200_000_000)
        pending.fill_(7.0)
        del pending
        out = kernel(a, stream=side.cuda_stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_output_on_current_and_explicit_current_stream(backend):
    kernel = _compile(backend)
    a = torch.randn(N, device="cuda")
    torch.testing.assert_close(kernel(a), a, rtol=0, atol=0)
    torch.testing.assert_close(kernel(a, stream=torch.cuda.current_stream().cuda_stream), a, rtol=0, atol=0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = kernel(a)
    side.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_output_allocation_stream_selection():
    from tilelang.jit.adapter.base import output_allocation_stream

    device = torch.device("cuda", torch.cuda.current_device())
    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert output_allocation_stream(side.cuda_stream, device) is None
        assert output_allocation_stream(other.cuda_stream, device).cuda_stream == other.cuda_stream
        # A zero handle is the default stream, not a fresh pooled stream.
        assert output_allocation_stream(0, device) == torch.cuda.default_stream(device)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_output_on_explicit_default_stream_from_side_stream(backend):
    kernel = _compile(backend)
    a = torch.randn(N, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        out = kernel(a, stream=0)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
