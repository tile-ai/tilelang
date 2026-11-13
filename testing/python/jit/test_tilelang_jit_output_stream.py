"""Outputs created for a kernel launched on a caller-supplied stream must be
allocated on that stream (#3648).

The caching allocator orders reuse of a block only after earlier work on the
stream it was allocated on. If the output were allocated on the current stream,
it could be a block that still has queued work there, which then overwrites the
kernel's result.

A kernel whose inputs are on a non-current device must also launch on that
device (#3408)."""

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


def fill_kernel(n):
    # 64 KB of shared memory, over the 48 KB a launch gets without opting in on
    # its device.
    smem = 16384

    @T.prim_func
    def main(value: T.float32, B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            S = T.alloc_shared((smem,), "float32")
            for i in T.Parallel(smem):
                S[i] = value
            for i in T.Parallel(256):
                B[bx * 256 + i] = S[i * (smem // 256)]

    return main


def _compile(backend, program=None, in_place=False):
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
    program = copy_kernel(N) if program is None else program
    out_idx = None if in_place else [len(program.params) - 1]
    return tilelang.compile(program, out_idx=out_idx, target=target, execution_backend=backend)


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


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_launch_on_non_current_input_device(backend):
    """Inputs on a non-current device run on that device's current stream, and
    the call leaves the current device unchanged (#3408)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    kernel = _compile(backend)
    for device in ("cuda:1", "cuda:1", "cuda:0", "cuda:1"):
        a = torch.randn(N, device=device)
        with torch.cuda.device(0):
            out = kernel(a)
            assert torch.cuda.current_device() == 0
        torch.cuda.synchronize(device)
        assert out.device == a.device
        torch.testing.assert_close(out, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_launch_on_the_explicit_streams_device(backend):
    """A caller-supplied stream launches on the device that owns it, also when
    another device is current, and outputs are allocated there (#3408)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from tilelang.jit.adapter.base import stream_device_index

    kernel = _compile(backend)
    side = torch.cuda.Stream(device="cuda:1")
    assert stream_device_index(side.cuda_stream) == 1
    assert stream_device_index(0) is None
    a = torch.randn(N, device="cuda:1")
    side.wait_stream(torch.cuda.current_stream("cuda:1"))
    with torch.cuda.device(0):
        out = kernel(a, stream=side.cuda_stream)
        assert torch.cuda.current_device() == 0
    side.synchronize()
    assert out.device == a.device
    torch.testing.assert_close(out, a, rtol=0, atol=0)

    # A default-stream handle runs with the inputs, not on the current device.
    a = torch.randn(N, device="cuda:1")
    with torch.cuda.device(0):
        out = kernel(a, stream=0)
    torch.cuda.synchronize("cuda:1")
    assert out.device == a.device
    torch.testing.assert_close(out, a, rtol=0, atol=0)

    # An in-place call with a default-stream handle runs with the inputs too.
    in_place = _compile(backend, in_place=True)
    a = torch.randn(N, device="cuda:1")
    b = torch.empty(N, device="cuda:1")
    with torch.cuda.device(0):
        in_place(a, b, stream=0)
    torch.cuda.synchronize("cuda:1")
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_launch_without_inputs_on_the_explicit_streams_device(backend):
    """Without a tensor input to infer the device from, the output and the shared-memory
    opt-in follow the stream (#3408). Skipped where the devices cannot opt in to
    64 KB of dynamic shared memory."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from tilelang.utils.device import get_max_dynamic_shared_memory

    if min(get_max_dynamic_shared_memory(0), get_max_dynamic_shared_memory(1)) < 64 << 10:
        pytest.skip("requires 64 KB of dynamic shared memory on both devices")

    kernel = _compile(backend, fill_kernel(N))
    side = torch.cuda.Stream(device="cuda:1")
    with torch.cuda.device(0):
        out = kernel(1.0, stream=side.cuda_stream)
    side.synchronize()
    assert out.device == torch.device("cuda:1")
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_stream_device_without_cu_stream_get_device(monkeypatch):
    """Drivers before CUDA 12.8 lack cuStreamGetDevice; the stream's device is
    then read from its context."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from tilelang.jit.adapter import base

    driver = base._cuda_driver()

    class OldDriver:
        def __getattr__(self, name):
            if name == "cuStreamGetDevice":
                raise AttributeError(name)
            return getattr(driver, name)

    monkeypatch.setattr(base, "_cuda_driver", OldDriver)
    for index in (0, 1):
        stream = torch.cuda.Stream(device=f"cuda:{index}")
        with torch.cuda.device(1 - index):
            assert base.stream_device_index(stream.cuda_stream) == index
            assert torch.cuda.current_device() == 1 - index


if __name__ == "__main__":
    tilelang.testing.main()
