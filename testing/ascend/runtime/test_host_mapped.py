"""Explicit mapped CPU parameters use both Ascend execution backends."""

import ctypes

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

pytest.importorskip("torch_npu")
pytestmark = tilelang.testing.requires_ascend.marks()
BACKENDS = ["cython", "tvm_ffi"]


def _host_device_connect_type():
    torch.npu.init()
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    acl.aclrtGetDeviceInfo.restype = ctypes.c_int
    connect_type = ctypes.c_int64(-1)
    # ACL_DEV_ATTR_HD_CONNECT_TYPE = 408; PCIe = 0, HCCS = 1, UB = 2.
    status = acl.aclrtGetDeviceInfo(torch.npu.current_device(), 408, ctypes.byref(connect_type))
    assert status == 0, f"Ascend host-device connection query failed: {status}"
    assert connect_type.value in (0, 1, 2), f"Unknown Ascend host-device connection type: {connect_type.value}"
    return connect_type.value


@pytest.fixture
def simt_host_mapping():
    # PCIe Through supports DMA transfers, but not SIMT accesses to mapped
    # host memory. A successful registration/pointer query does not prove that
    # the kernel's memory instructions can access it. Gate before any launch
    # to avoid a device fault poisoning subsequent tests in the same worker.
    if _host_device_connect_type() == 0:
        pytest.skip("PCIe Through does not support SIMT host-memory accesses; SIMD/DMA coverage runs separately")


@pytest.fixture
def mapped_storage():
    # Do not change allocator environment variables after other tests may have
    # initialized torch_npu. Register only allocations not already mapped by it.
    torch.npu.init()
    acl = ctypes.CDLL("libascendcl.so")
    for name in ("aclrtHostRegister", "aclrtHostGetDevicePointer", "aclrtHostUnregister"):
        if not hasattr(acl, name):
            pytest.skip(f"CANN runtime does not provide {name}")
    acl.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    acl.aclrtHostGetDevicePointer.restype = ctypes.c_int
    acl.aclrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtHostRegister.restype = ctypes.c_int
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtHostUnregister.restype = ctypes.c_int
    acl.aclrtGetLastError.argtypes = [ctypes.c_int]
    acl.aclrtGetLastError.restype = ctypes.c_int
    allocations = []

    def allocate(shape, dtype=torch.float32):
        tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
        pointer = ctypes.c_void_p()
        registered = acl.aclrtHostGetDevicePointer(tensor.data_ptr(), ctypes.byref(pointer), 0) == 0
        if not registered:
            # The probe's failure must not leak into the next kernel launch.
            acl.aclrtGetLastError(0)
            # ACL_HOST_REGISTER_MAPPED = 0. Lifetime stays with the test.
            status = acl.aclrtHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, ctypes.byref(pointer))
            assert status == 0, f"Ascend host registration failed: {status}"
        allocations.append((tensor, not registered))
        return tensor

    yield allocate
    torch.npu.synchronize()
    for tensor, owned_registration in allocations:
        if owned_registration:
            assert acl.aclrtHostUnregister(tensor.data_ptr()) == 0


def _affine(n=4099, stride=1, annotated=True):
    @T.prim_func
    def kernel(src: T.Tensor((n,), T.float32), dst: T.StridedTensor((n,), (stride,), T.float32)):
        if annotated:
            T.annotate_host_mapped_buffers(dst)
        with T.Kernel(T.ceildiv(n, 128)) as block, T.SimtVF(threads=128):
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < n:
                    dst[i] = src[i] * 2 + 1

    return kernel


def _double_dma(host_input=True, host_output=True):
    @T.prim_func
    def kernel(src: T.Tensor((64,), T.float32), dst: T.Tensor((64,), T.float32)):
        if host_input:
            T.annotate_host_mapped_buffers(src)
        if host_output:
            T.annotate_host_mapped_buffers(dst)
        with T.Kernel(1):
            values = T.alloc_shared((64,), T.float32)
            result = T.alloc_shared((64,), T.float32)
            T.copy(src, values)
            with T.SimdVF():
                mask = T.simd.pset(32)
                x = T.simd.vld(values[0])
                T.simd.vsts(result[0], T.simd.vadd(x, x, mask), mask)
            T.copy(result, dst)

    return kernel


def test_host_mapped_stub_pointer_query(mapped_storage):
    from tilelang.libinfo import find_lib_path

    try:
        stub_path = find_lib_path("stub_ascendcl")
    except RuntimeError:
        pytest.skip("requires an Ascend stub build")
    # Query the stub directly so ELF interposition cannot bypass its loader.
    stub = ctypes.CDLL(stub_path)
    query = stub.aclrtHostGetDevicePointer
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    query.restype = ctypes.c_int
    storage = mapped_storage(128)
    base_pointer = ctypes.c_void_p()
    view_pointer = ctypes.c_void_p()
    assert query(storage.data_ptr(), ctypes.byref(base_pointer), 0) == 0
    assert query(storage[4:].data_ptr(), ctypes.byref(view_pointer), 0) == 0
    assert view_pointer.value == base_pointer.value + 4 * storage.element_size()
    ordinary = torch.empty(128)
    assert query(ordinary.data_ptr(), ctypes.byref(view_pointer), 0) != 0
    stub.aclrtGetLastError(0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.usefixtures("simt_host_mapping")
def test_host_mapped_eager_and_graph(backend, stride, mapped_storage):
    n = 4099
    kernel = tilelang.compile(_affine(n, stride), target="ascend", execution_backend=backend)
    base = torch.arange(n, dtype=torch.float32)
    src = base.npu()
    storage = mapped_storage(n * stride + 8).fill_(-777)
    dst = storage[4 : 4 + n * stride : stride]
    guard = torch.ones_like(storage, dtype=torch.bool)
    guard[4 : 4 + n * stride : stride] = False

    def check(expected):
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        assert bool((storage[guard] == -777).all())

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        kernel(src, dst)
    stream.synchronize()
    check(base * 2 + 1)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        kernel(src, dst)
    for replay in range(3):
        values = base + (replay + 1) * 17
        src.copy_(values)
        torch.npu.synchronize()
        storage.fill_(-777)
        graph.replay()
        torch.npu.synchronize()
        check(values * 2 + 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_mapped_preserves_validation(backend, mapped_storage):
    n = 33
    kernel = tilelang.compile(_affine(n), target="ascend", execution_backend=backend)
    src = torch.arange(n, dtype=torch.float32).npu()
    good = mapped_storage(n)
    invalid_outputs = [
        (torch.empty(n), "registered|mapped"),
        (torch.empty_like(src), "device|host|Host"),
        (mapped_storage(n, torch.int32), "dtype"),
        (mapped_storage(n - 1), "shape"),
        (mapped_storage((1, n)), "shape|ndim|dimensions"),
        (mapped_storage(n * 2)[::2], "stride|contiguous"),
    ]
    for dst, message in invalid_outputs:
        with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
            kernel(src, dst)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="device"):
        kernel(src.cpu(), good)

    normal = tilelang.compile(_affine(n, annotated=False), target="ascend", execution_backend=backend)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="device"):
        normal(src, good)
    device_dst = torch.empty_like(src)
    normal(src, device_dst)
    torch.testing.assert_close(device_dst.cpu(), src.cpu() * 2 + 1, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.usefixtures("simt_host_mapping")
def test_host_mapped_first_parameter_and_host_only(backend, mapped_storage):
    def program(host_input):
        @T.prim_func
        def kernel(dst: T.Tensor((129,), T.float32), src: T.Tensor((129,), T.float32)):
            T.annotate_host_mapped_buffers(dst)
            if host_input:
                T.annotate_host_mapped_buffers(src)
            with T.Kernel(1), T.SimtVF(threads=128):
                for i in T.Parallel(129):
                    dst[i] = src[i] + 2

        return kernel

    src = mapped_storage(129)
    src.copy_(torch.arange(129, dtype=torch.float32))
    dst = mapped_storage(129)
    first = tilelang.compile(program(False), target="ascend", execution_backend=backend)
    first(dst, src.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src + 2, rtol=0, atol=0)

    only = tilelang.compile(program(True), target="ascend", execution_backend=backend)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        only(dst, src)
    stream.synchronize()
    torch.testing.assert_close(dst, src + 2, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        only(dst, src)
    torch.npu.synchronize()
    src.add_(17)
    dst.fill_(-777)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src + 2, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_mapped_simd_and_dma(backend, mapped_storage):
    kernel = tilelang.compile(_double_dma(), target="ascend", execution_backend=backend)
    # Exercise nonzero storage offsets while preserving DMA alignment.
    src_storage = mapped_storage(80).fill_(-777)
    dst_storage = mapped_storage(80).fill_(-777)
    src = src_storage[8:72]
    dst = dst_storage[8:72]
    src.copy_(torch.arange(64, dtype=torch.float32))

    def check():
        torch.testing.assert_close(dst, src * 2, rtol=0, atol=0)
        for storage in (src_storage, dst_storage):
            assert bool((storage[:8] == -777).all())
            assert bool((storage[72:] == -777).all())

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        kernel(src, dst)
    stream.synchronize()
    check()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        kernel(src, dst)
    torch.npu.synchronize()
    for _ in range(3):
        src.add_(17)
        dst.fill_(-777)
        graph.replay()
        torch.npu.synchronize()
        check()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["simt", "dma"])
def test_host_input_with_automatic_device_output(backend, mode, mapped_storage, request):
    if mode == "simt":
        request.getfixturevalue("simt_host_mapping")

    @T.prim_func
    def kernel(src: T.Tensor((33,), T.float32), dst: T.Tensor((33,), T.float32)):
        T.annotate_host_mapped_buffers(src)
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 4

    func = kernel if mode == "simt" else _double_dma(host_output=False)
    n = 33 if mode == "simt" else 64
    compiled = tilelang.compile(func, out_idx=[1], target="ascend", execution_backend=backend)
    src = mapped_storage(n)
    src.copy_(torch.arange(n, dtype=torch.float32))
    dst = compiled(src)
    assert dst.device == torch.device("npu", torch.npu.current_device())
    expected = src + 4 if mode == "simt" else src * 2
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_mapping_checks_execution_device(backend, mapped_storage):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPUs")
    kernel = tilelang.compile(_affine(33), target="ascend", execution_backend=backend)
    current_device = torch.npu.current_device()
    src = torch.arange(33, dtype=torch.float32).npu()
    dst = mapped_storage(33)
    torch.npu.synchronize()
    with torch.npu.device((current_device + 1) % torch.npu.device_count()), pytest.raises((ValueError, RuntimeError), match="device"):
        kernel(src, dst)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.usefixtures("simt_host_mapping")
def test_host_mapped_eager_jit(backend, mapped_storage):
    @tilelang.jit(target="ascend", execution_backend=backend)
    def kernel(src, dst):
        src: T.Tensor((33,), T.float32)
        dst: T.Tensor((33,), T.float32)
        T.annotate_host_mapped_buffers(dst)
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(33):
                dst[i] = src[i] + 3

    src = torch.arange(33, dtype=torch.float32).npu()
    dst = mapped_storage(33)
    kernel(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src.cpu() + 3, rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["simt", "dma"])
def test_host_mapped_disk_cache(backend, mode, mapped_storage, tmp_path, monkeypatch, request):
    from tilelang import tvm
    from tilelang.backend import create_backend_context
    from tilelang.cache import _dispatch_map
    from tilelang.env import env
    from tilelang.jit.kernel import JITKernel

    if mode == "simt":
        request.getfixturevalue("simt_host_mapping")
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    func = _affine(33) if mode == "simt" else _double_dma(host_input=False)
    n = 33 if mode == "simt" else 64
    kernel = JITKernel(func, target="ascend", execution_backend=backend)
    cache = _dispatch_map[backend]
    key = "ascend-host-mapped-round-trip"
    cache._save_kernel_to_disk(key, kernel)
    restored = cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("ascend", execution_backend=backend),
        out_idx=[],
        pass_configs=None,
        compile_flags=None,
        func=tvm.ir.load_json(tvm.ir.save_json(func)),
    )
    assert restored is not None
    src = torch.arange(n, dtype=torch.float32).npu()
    dst = mapped_storage(n)
    restored(src, dst)
    torch.npu.synchronize()
    expected = src.cpu() * 2 + (1 if mode == "simt" else 0)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    with pytest.raises((ValueError, RuntimeError), match="registered|mapped"):
        restored(src, torch.empty(n))


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_mapped_output_must_be_explicit(backend):
    with pytest.raises(ValueError, match="preallocated"):
        tilelang.compile(_affine(33), target="ascend", execution_backend=backend, out_idx=[1])


def test_host_mapped_pto_launcher_is_rejected():
    from tilelang import tvm
    from tilelang.engine.host_mapped import validate_host_mapped_params

    with pytest.raises(ValueError, match="currently require"):
        validate_host_mapped_params(_affine(33), tvm.target.Target("ascend"), "pto")
