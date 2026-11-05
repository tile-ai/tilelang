"""Direct GPU access to explicit, mapped host tensor parameters."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

BACKENDS = ["cython", "tvm_ffi"]


@tilelang.testing.requires_cuda
def test_host_mapped_cuda_stub_pointer_query():
    import ctypes

    from tilelang.libinfo import find_lib_path

    try:
        stub_path = find_lib_path("stub_cuda")
    except RuntimeError:
        pytest.skip("requires a CUDA stub build")

    # Call the stub explicitly: ELF symbol interposition can otherwise route
    # the runtime helper directly to an already-loaded CUDA driver library.
    stub = ctypes.CDLL(stub_path)
    get_pointer = stub.cuMemHostGetDevicePointer_v2
    get_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
    get_pointer.restype = ctypes.c_int

    torch.empty(1, device="cuda")  # Ensure a CUDA context is current.
    storage = torch.empty(128, pin_memory=True)
    base_pointer = ctypes.c_uint64()
    status = get_pointer(ctypes.byref(base_pointer), ctypes.c_void_p(storage.data_ptr()), 0)
    assert status == 0, f"CUDA stub mapping failed with error {status}"
    assert base_pointer.value != 0

    view = storage[4:]
    view_pointer = ctypes.c_uint64()
    status = get_pointer(ctypes.byref(view_pointer), ctypes.c_void_p(view.data_ptr()), 0)
    assert status == 0, f"CUDA stub view mapping failed with error {status}"
    assert view_pointer.value == base_pointer.value + 4 * storage.element_size()


def _affine(n=4099, stride=1, annotated=True):
    @T.prim_func
    def kernel(src: T.Tensor((n,), T.float32), dst: T.StridedTensor((n,), (stride,), T.float32)):
        if annotated:
            T.annotate_host_mapped_buffers(dst)
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < n:
                    dst[i] = src[i] * 2 + 1

    return kernel


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stride", [1, 2])
@tilelang.testing.requires_cuda
def test_host_mapped_eager_and_graph(backend, stride):
    n = 4099
    kernel = tilelang.compile(_affine(n, stride), target="cuda", execution_backend=backend)
    base = torch.arange(n, dtype=torch.float32)
    src = base.cuda()
    sentinel = -777.0
    storage = torch.full((n * stride + 8,), sentinel, pin_memory=True)
    dst = storage[4 : 4 + n * stride : stride]
    guard = torch.ones_like(storage, dtype=torch.bool)
    guard[4 : 4 + n * stride : stride] = False

    def check(expected):
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        assert bool((storage[guard] == sentinel).all())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(src, dst)
    stream.synchronize()
    check(base * 2 + 1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        kernel(src, dst)
    for replay in range(3):
        values = base + (replay + 1) * 17
        src.copy_(values)
        torch.cuda.synchronize()
        storage.fill_(sentinel)
        graph.replay()
        torch.cuda.synchronize()
        check(values * 2 + 1)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_mapped_preserves_validation(backend):
    n = 33
    kernel = tilelang.compile(_affine(n), target="cuda", execution_backend=backend)
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    good = torch.empty(n, pin_memory=True)
    invalid_outputs = [
        (torch.empty(n), "registered|pinned|mapped"),
        (torch.empty(n, device="cuda"), "device|host|Host"),
        (torch.empty(n, dtype=torch.int32, pin_memory=True), "dtype"),
        (torch.empty(n - 1, pin_memory=True), "shape"),
        (torch.empty((1, n), pin_memory=True), "shape|ndim|dimensions"),
        (torch.empty(n * 2, pin_memory=True)[::2], "stride|contiguous"),
    ]
    for dst, message in invalid_outputs:
        with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
            kernel(src, dst)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="device"):
        kernel(src.cpu(), good)

    # The opt-in applies only to the annotated parameter of this function.
    normal = tilelang.compile(_affine(n, annotated=False), target="cuda", execution_backend=backend)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="device"):
        normal(src, good)
    device_dst = torch.empty_like(src)
    normal(src, device_dst)
    torch.testing.assert_close(device_dst, src * 2 + 1, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_mapped_first_parameter_and_host_only(backend):
    @T.prim_func
    def host_first(dst: T.Tensor((33,), T.float32), src: T.Tensor((33,), T.float32)):
        T.annotate_host_mapped_buffers(dst)
        with T.Kernel(1, threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 1

    @T.prim_func
    def host_only(dst: T.Tensor((33,), T.float32), src: T.Tensor((33,), T.float32)):
        T.annotate_host_mapped_buffers(dst, src)
        with T.Kernel(1, threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 2

    src = torch.arange(33, dtype=torch.float32, pin_memory=True)
    dst = torch.empty_like(src, pin_memory=True)
    first = tilelang.compile(host_first, target="cuda", execution_backend=backend)
    device_src = src.cuda()
    first(dst, device_src)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src + 1, rtol=0, atol=0)
    only = tilelang.compile(host_only, target="cuda", execution_backend=backend)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        only(dst, src)
    stream.synchronize()
    torch.testing.assert_close(dst, src + 2, rtol=0, atol=0)

    # With no CUDA tensor argument, the launcher must still use Torch's stream.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        only(dst, src)
    torch.cuda.synchronize()
    src.add_(17)
    dst.fill_(-777)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src + 2, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_mapped_eager_jit(backend):
    @tilelang.jit(target="cuda", execution_backend=backend)
    def kernel(src, dst):
        src: T.Tensor((33,), T.float32)
        dst: T.Tensor((33,), T.float32)
        T.annotate_host_mapped_buffers(dst)
        with T.Kernel(1, threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 3

    src = torch.arange(33, dtype=torch.float32, device="cuda")
    dst = torch.empty(33, pin_memory=True)
    kernel(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src.cpu() + 3, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_input_with_automatic_device_output(backend):
    @T.prim_func
    def kernel(src: T.Tensor((33,), T.float32), dst: T.Tensor((33,), T.float32)):
        T.annotate_host_mapped_buffers(src)
        with T.Kernel(1, threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 4

    compiled = tilelang.compile(kernel, out_idx=[1], target="cuda", execution_backend=backend)
    src = torch.arange(33, dtype=torch.float32, pin_memory=True)
    dst = compiled(src)
    assert dst.device.type == "cuda"
    torch.testing.assert_close(dst.cpu(), src + 4, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_mapping_uses_execution_device(backend):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(0):
        dst = torch.empty(33, pin_memory=True)
    with torch.cuda.device(1):
        kernel = tilelang.compile(_affine(33), target="cuda", execution_backend=backend)
        src = torch.arange(33, dtype=torch.float32, device="cuda:1")
        kernel(src, dst)
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, src.cpu() * 2 + 1, rtol=0, atol=0)
    with torch.cuda.device(0), pytest.raises((ValueError, RuntimeError), match="device"):
        kernel(src, dst)


@pytest.mark.parametrize("backend", BACKENDS)
@tilelang.testing.requires_cuda
def test_host_mapped_disk_cache(backend, tmp_path, monkeypatch):
    from tilelang.backend import create_backend_context
    from tilelang.cache import _dispatch_map
    from tilelang.env import env
    from tilelang import tvm
    from tilelang.jit.kernel import JITKernel

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    func = _affine(33)
    kernel = JITKernel(func, target="cuda", execution_backend=backend)
    cache = _dispatch_map[backend]
    key = "host-mapped-round-trip"
    cache._save_kernel_to_disk(key, kernel)
    restored_func = tvm.ir.load_json(tvm.ir.save_json(func))
    restored = cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("cuda", execution_backend=backend),
        out_idx=[],
        pass_configs=None,
        compile_flags=None,
        func=restored_func,
    )
    assert restored is not None
    src = torch.arange(33, dtype=torch.float32, device="cuda")
    dst = torch.empty(33, pin_memory=True)
    restored(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src.cpu() * 2 + 1, rtol=0, atol=0)
    with pytest.raises((ValueError, RuntimeError), match="registered|pinned|mapped"):
        restored(src, torch.empty(33))
