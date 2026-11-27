"""Metal partial-region fill and copy coverage across supported layouts."""

import pytest
import torch
from tilelang.metal.language import cooperative_tensor_store

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_FILL_MATRIX_OP = "tirx.make_filled_simdgroup_matrix"
_COOP_FILL_OP = "tl.cooperative_tensor_fill"


def _lower(func, target="metal"):
    with tvm.transform.PassContext(), tvm.target.Target(target):
        return tilelang.lower(func, target=target)


def _compile(func):
    return tilelang.compile(
        func,
        out_idx=[],
        target="metal",
        target_host="c",
        execution_backend="tvm_ffi",
    )


def _execute(func, shape, dtype, outputs=1):
    compiled = _compile(func)
    tensors = [torch.full(shape, 9.0, dtype=dtype.as_torch(), device="mps") for _ in range(outputs)]
    compiled(*tensors)
    torch.mps.synchronize()
    return [tensor.cpu() for tensor in tensors]


def _calls(artifact, op_name):
    op = tvm.ir.Op.get(op_name)
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op.same_as(op):
            calls.append(node)

    for func in artifact.device_mod.functions.values():
        if isinstance(func, tvm.tirx.PrimFunc):
            tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _int_arg(call, index):
    arg = call.args[index]
    assert isinstance(arg, tvm.tirx.IntImm), arg
    return arg.value


def _float_arg(call, index):
    arg = call.args[index]
    while isinstance(arg, tvm.tirx.Cast):
        arg = arg.value
    assert isinstance(arg, tvm.tirx.FloatImm), arg
    return arg.value


def _apply_regions(shape, regions, dtype):
    reference = torch.empty(shape, dtype=dtype.as_torch())
    for region in regions:
        if len(region) == 5:
            row_begin, row_end, col_begin, col_end, value = region
            reference[row_begin:row_end, col_begin:col_end] = value
        else:
            begin, end, value = region
            reference[begin:end] = value
    return reference


def _region_fill_2d(scope, shape, regions, dtype):
    rows, cols = shape

    @T.prim_func
    def kernel(output: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((rows, cols), dtype, scope=scope)
            for row_begin, row_end, col_begin, col_end, value in regions:
                T.fill(buf[row_begin:row_end, col_begin:col_end], T.cast(value, dtype))
            T.copy(buf, output)

    return kernel


def _region_fill_1d(scope, extent, regions, dtype):
    tile = 64 if scope == "metal.simdgroup" else 512

    @T.prim_func
    def kernel(output: T.Tensor((extent,), dtype)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((extent,), dtype, scope=scope)
            for begin, end, value in regions:
                T.fill(buf[begin:end], T.cast(value, dtype))
            for i in T.unroll(extent // tile):
                if scope == "metal.simdgroup":
                    T.evaluate(tvm.tirx.simdgroup_store(buf.data, i, T.address_of(output[i * tile]), 8))
                else:
                    cooperative_tensor_store(
                        buf.data,
                        i,
                        T.address_of(output[i * tile]),
                        32,
                        16,
                        32,
                        False,
                        16,
                        32,
                        16,
                        2,
                    )

    return kernel


_SIMDGROUP_FILL_2D = [
    (0, 32, 0, 64, -7),
    (0, 16, 0, 32, 3),
    (16, 32, 32, 64, 5),
    (8, 24, 8, 40, 11),
    (24, 32, 48, 56, 13),
]
_COOPERATIVE_FILL_2D = [
    (0, 32, 0, 64, -7),
    (0, 16, 0, 32, 3),
    (16, 32, 32, 64, 5),
    (7, 25, 13, 49, 11),
    (31, 32, 63, 64, 13),
]
_SIMDGROUP_FILL_1D = [
    (0, 512, -7),
    (64, 192, 3),
    (320, 448, 5),
    (448, 512, 11),
]
_COOPERATIVE_FILL_1D = [
    (0, 1024, -7),
    (0, 512, 3),
    (512, 1024, 5),
    (7, 519, 11),
]


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "scope,dtype,regions",
    [
        pytest.param("metal.simdgroup", T.float32, _SIMDGROUP_FILL_2D, id="simdgroup-fp32"),
        pytest.param("metal.cooperative_tensor", T.float16, _COOPERATIVE_FILL_2D, id="cooperative-fp16"),
        pytest.param("metal.cooperative_tensor", T.float32, _COOPERATIVE_FILL_2D, id="cooperative-fp32"),
    ],
)
def test_region_fill_2d(scope, dtype, regions):
    shape = (32, 64)
    output = _execute(_region_fill_2d(scope, shape, regions, dtype), shape, dtype)[0]
    reference = _apply_regions(shape, regions, dtype)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "scope,dtype,regions,extent",
    [
        pytest.param("metal.simdgroup", T.float32, _SIMDGROUP_FILL_1D, 512, id="simdgroup-fp32"),
        pytest.param("metal.cooperative_tensor", T.float16, _COOPERATIVE_FILL_1D, 1024, id="cooperative-fp16"),
        pytest.param("metal.cooperative_tensor", T.float32, _COOPERATIVE_FILL_1D, 1024, id="cooperative-fp32"),
    ],
)
def test_region_fill_1d(scope, dtype, regions, extent):
    output = _execute(_region_fill_1d(scope, extent, regions, dtype), (extent,), dtype)[0]
    reference = _apply_regions((extent,), regions, dtype)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@tilelang.testing.requires_metal
def test_partial_source_copy_keeps_region_and_offsets():
    @T.prim_func
    def kernel(
        out_shared: T.Tensor((32, 64), T.float32),
        out_whole: T.Tensor((32, 64), T.float32),
        out_direct: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((32, 64), T.float32, scope="metal.simdgroup")
            staging = T.alloc_shared((32, 64), T.float32)
            T.fill(buf, T.float32(-7))
            T.fill(buf[8:24, 16:40], T.float32(3))
            T.fill(buf[24:32, 48:56], T.float32(5))
            T.fill(staging, T.float32(9))
            T.copy(buf[8:24, 16:40], staging[4:20, 8:32])
            T.copy(staging, out_shared)
            T.copy(buf, out_whole)
            T.copy(buf[8:24, 16:40], out_direct[4:20, 8:32])

    shared, whole, direct = _execute(kernel, (32, 64), T.float32, outputs=3)

    source = torch.full((32, 64), -7.0)
    source[8:24, 16:40] = 3
    source[24:32, 48:56] = 5

    expected_shared = torch.full((32, 64), 9.0)
    expected_shared[4:20, 8:32] = source[8:24, 16:40]
    expected_direct = torch.full((32, 64), 9.0)
    expected_direct[4:20, 8:32] = source[8:24, 16:40]

    torch.testing.assert_close(shared, expected_shared, rtol=0, atol=0)
    torch.testing.assert_close(whole, source, rtol=0, atol=0)
    torch.testing.assert_close(direct, expected_direct, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "region,expected_tiles",
    [
        pytest.param((8, 16, 0, 8), [2], id="row-min"),
        pytest.param((0, 16, 0, 8), [0, 2], id="row-stride"),
        pytest.param((0, 8, 8, 16), [1], id="col-min"),
    ],
)
def test_aligned_partial_fill_tile_indices(region, expected_tiles):
    row_begin, row_end, col_begin, col_end = region

    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(-7))
            T.fill(buf[row_begin:row_end, col_begin:col_end], T.float32(3))
            T.copy(buf, output)

    calls = _calls(_lower(kernel), _FILL_MATRIX_OP)
    initializers = [call for call in calls if _float_arg(call, 2) == -7.0]
    updates = [call for call in calls if _float_arg(call, 2) == 3.0]
    assert initializers
    buffer_var = initializers[0].args[0]
    assert all(call.args[0].same_as(buffer_var) for call in calls)
    assert [_int_arg(call, 1) for call in updates] == expected_tiles


def _cooperative_prefix_kernel(shape, elements):
    inner = 1
    for extent in shape[1:]:
        inner *= extent
    assert elements % inner == 0
    outer = elements // inner

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope="metal.cooperative_tensor")
            if len(shape) == 1:
                T.fill(buf[0:elements], T.float32(3))
            elif len(shape) == 2:
                T.fill(buf[0:outer, 0 : shape[1]], T.float32(3))
            elif len(shape) == 3:
                T.fill(buf[0:outer, 0 : shape[1], 0 : shape[2]], T.float32(3))
            else:
                T.fill(buf[0:outer, 0 : shape[1], 0 : shape[2], 0 : shape[3]], T.float32(3))

    return kernel


def _cooperative_whole_kernel(shape):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope="metal.cooperative_tensor")
            T.fill(buf, T.float32(3))

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "shape,elements",
    [
        pytest.param((1024,), 512, id="1d"),
        pytest.param((768,), 512, id="1d-legacy"),
        pytest.param((1280,), 1024, id="1d-legacy-multi"),
        pytest.param((4, 256), 512, id="2d-legacy"),
        pytest.param((4, 512), 1024, id="2d-legacy-multi"),
        pytest.param((32, 32), 512, id="2d-canonical"),
        pytest.param((2, 2, 256), 512, id="3d"),
        pytest.param((4, 2, 256), 1024, id="3d-multi"),
        pytest.param((2, 1, 2, 256), 512, id="4d"),
    ],
)
def test_cooperative_prefix_tile_indices(shape, elements):
    calls = _calls(_lower(_cooperative_prefix_kernel(shape, elements)), _COOP_FILL_OP)
    assert [_int_arg(call, 1) for call in calls] == list(range(elements // 512))
    assert all(_float_arg(call, 2) == 3.0 for call in calls)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "shape,expected_tiles",
    [
        pytest.param((1024,), [0, 1], id="1d"),
        pytest.param((32, 64), [0, 1, 2, 3], id="2d"),
        pytest.param((2, 2, 256), [0, 1], id="3d"),
    ],
)
def test_cooperative_whole_buffer_tile_indices(shape, expected_tiles):
    calls = _calls(_lower(_cooperative_whole_kernel(shape)), _COOP_FILL_OP)
    assert [_int_arg(call, 1) for call in calls] == expected_tiles


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "extent,elements",
    [
        pytest.param(768, 512, id="1d-legacy"),
        pytest.param(1280, 1024, id="1d-legacy-multi"),
    ],
)
def test_cooperative_legacy_prefix_fill_1d(extent, elements):
    @T.prim_func
    def kernel(output: T.Tensor((elements,), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((extent,), T.float32, scope="metal.cooperative_tensor")
            T.fill(buf[0:elements], T.float32(3))
            for i in T.unroll(elements // 512):
                cooperative_tensor_store(
                    buf.data,
                    i,
                    T.address_of(output[i * 512]),
                    32,
                    16,
                    32,
                    False,
                    16,
                    32,
                    16,
                    2,
                )

    output = _execute(kernel, (elements,), T.float32)[0]
    torch.testing.assert_close(output, torch.full((elements,), 3.0), rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "shape,elements",
    [
        pytest.param((4, 512), 1024, id="2d-cols-512"),
        pytest.param((8, 256), 1024, id="2d-cols-256"),
    ],
)
def test_cooperative_legacy_prefix_fill_2d(shape, elements):
    @T.prim_func
    def kernel(output: T.Tensor((elements,), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope="metal.cooperative_tensor")
            T.fill(buf[0 : elements // shape[1], :], T.float32(3))
            for i in T.unroll(elements // 512):
                cooperative_tensor_store(
                    buf.data,
                    i,
                    T.address_of(output[i * 512]),
                    32,
                    16,
                    32,
                    False,
                    16,
                    32,
                    16,
                    2,
                )

    output = _execute(kernel, (elements,), T.float32)[0]
    torch.testing.assert_close(output, torch.full((elements,), 3.0), rtol=0, atol=0)


def _fill_2d(scope, shape, region, threads=32, active_threads=None, value=3.0):
    @T.prim_func
    def kernel(output: T.Tensor(shape, T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local(shape, T.float32, scope=scope)
            if active_threads is None:
                T.fill(buf[region[0] : region[1], region[2] : region[3]], T.float32(value))
            else:
                tx = T.get_thread_binding()
                if tx < active_threads:
                    T.fill(buf[region[0] : region[1], region[2] : region[3]], T.float32(value))
            T.copy(buf, output)

    return kernel


def _fill_1d(scope, extent, region, threads=32, active_threads=None):
    @T.prim_func
    def kernel(output: T.Tensor((extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local((extent,), T.float32, scope=scope)
            if active_threads is None:
                T.fill(buf[region[0] : region[1]], T.float32(3))
            else:
                tx = T.get_thread_binding()
                if tx < active_threads:
                    T.fill(buf[region[0] : region[1]], T.float32(3))

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "kernel,reason",
    [
        pytest.param(_fill_2d("metal.simdgroup", (32, 64), (1, 9, 0, 8)), "8-aligned", id="2d-unaligned"),
        pytest.param(_fill_2d("metal.simdgroup", (32, 64), (8, 16, 0, 8), threads=128), "exactly one simdgroup", id="2d-multi-group"),
        pytest.param(_fill_1d("metal.simdgroup", 512, (1, 65)), "64-aligned", id="1d-unaligned"),
        pytest.param(_fill_1d("metal.simdgroup", 512, (64, 128), threads=64), "exactly one simdgroup", id="1d-multi-group"),
        pytest.param(_fill_2d("metal.cooperative_tensor", (16, 48), (0, 1, 0, 1)), "16x32-aligned buffer shape", id="cooperative-shape"),
        pytest.param(_fill_2d("metal.cooperative_tensor", (32, 64), (31, 33, 0, 1)), "exceeds", id="cooperative-bounds"),
        pytest.param(_fill_1d("metal.cooperative_tensor", 1024, (1024, 1025)), "exceeds", id="cooperative-bounds-1d"),
        pytest.param(
            _fill_1d("metal.cooperative_tensor", 1024, (7, 519), threads=64), "exactly one simdgroup", id="cooperative-multi-group"
        ),
    ],
)
def test_partial_fill_diagnostics(kernel, reason):
    with pytest.raises(ValueError, match=reason):
        _lower(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "threads,active_threads,expected",
    [
        pytest.param(64, 32, "exactly one simdgroup", id="predicated-launch"),
        pytest.param((32, 2), 32, "exactly one simdgroup", id="two-dimensional-launch"),
        pytest.param(32, 16, "complete simdgroup execution range", id="incomplete-range"),
    ],
)
def test_partial_fill_checks_launch_before_predicate(threads, active_threads, expected):
    kernel = _fill_2d("metal.simdgroup", (32, 64), (8, 16, 0, 8), threads=threads, active_threads=active_threads)
    with pytest.raises(ValueError, match=expected):
        _lower(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("scope", ["metal.simdgroup", "metal.cooperative_tensor"])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("region", [(0, 1), (1, 2)], ids=["prefix", "offset"])
def test_high_rank_partial_fill_diagnostics(scope, ndim, region):
    shape = (2,) * (ndim - 1) + (256,)

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope=scope)
            if ndim == 3:
                T.fill(buf[region[0] : region[1], :, :], T.float32(3))
            else:
                T.fill(buf[region[0] : region[1], :, :, :], T.float32(3))

    if scope == "metal.cooperative_tensor" and region == (0, 1):
        calls = _calls(_lower(kernel), _COOP_FILL_OP)
        tiles = 1 if ndim == 3 else 2
        assert [_int_arg(call, 1) for call in calls] == list(range(tiles))
    else:
        with pytest.raises(ValueError, match="3 or more dimensions"):
            _lower(kernel)


@tilelang.testing.requires_metal
def test_cooperative_noncontiguous_prefix_rejected():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((2, 4, 256), T.float32, scope="metal.cooperative_tensor")
            T.fill(buf[0:1, 0:4, 0:128], T.float32(3))

    with pytest.raises(ValueError, match="3 or more dimensions"):
        _lower(kernel)


@tilelang.testing.requires_metal
def test_partial_fill_rejects_divergent_lanes():
    @T.prim_func
    def kernel(output: T.Tensor((32, 64), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((32, 64), T.float32, scope="metal.simdgroup")
            T.clear(buf)
            tx = T.get_thread_binding()
            if tx != 16:
                T.fill(buf[8:16, 0:8], T.float32(3))
            T.copy(buf, output)

    with pytest.raises(ValueError, match="thread-uniform control flow"):
        _lower(kernel)


@tilelang.testing.requires_metal
def test_cooperative_partial_fill_rejects_divergent_lanes():
    @T.prim_func
    def kernel(flags: T.Tensor((32,), T.int32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((1024,), T.float32, scope="metal.cooperative_tensor")
            flag = flags[T.get_thread_binding()]
            if flag != 0:
                T.fill(buf[7:519], T.float32(3))

    with pytest.raises(ValueError, match="thread-uniform control flow"):
        _lower(kernel)


@tilelang.testing.requires_metal
def test_partial_copy_rejects_divergent_lanes():
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(3))
            tx = T.get_thread_binding()
            if tx != 16:
                T.copy(buf[8:16, 0:8], output[8:16, 0:8])

    with pytest.raises(ValueError, match="thread-uniform control flow"):
        _lower(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind", ["direct", "alias", "uniform"])
def test_partial_fill_branch_provenance(kind):
    @T.prim_func
    def kernel(flags: T.Tensor((32,), T.int32), output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(buf)
            index = 0 if kind == "uniform" else T.get_thread_binding()
            flag = flags[index]
            alias = flag * 2 if kind == "alias" else flag
            if alias != 0:
                T.fill(buf[8:16, 0:8], T.float32(3))
            T.copy(buf, output)

    if kind == "uniform":
        calls = _calls(_lower(kernel), _FILL_MATRIX_OP)
        assert [_int_arg(call, 1) for call in calls if _float_arg(call, 2) == 3.0] == [2]
    else:
        with pytest.raises(ValueError, match="thread-uniform control flow"):
            _lower(kernel)


@tilelang.testing.requires_metal
def test_launch_size_does_not_leak_between_kernels():
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=64):
            first = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(first)
            T.copy(first, output)
        with T.Kernel(1, threads=32):
            second = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(second, T.float32(-7))
            T.fill(second[8:16, 0:8], T.float32(3))
            T.copy(second, output)

    calls = _calls(_lower(kernel), _FILL_MATRIX_OP)
    assert [_int_arg(call, 1) for call in calls if _float_arg(call, 2) == 3.0] == [2]


@tilelang.testing.requires_metal
def test_partial_fill_after_gemm_targets_native_tensor():
    @T.prim_func
    def kernel(
        a: T.Tensor((32, 16), T.float16),
        b: T.Tensor((16, 64), T.float16),
        output: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((32, 64), T.float32, scope="metal.cooperative_tensor")
            T.gemm(a, b, buf, clear_accum=True)
            T.fill(buf[31:32, 63:64], T.float32(3))
            T.copy(buf, output)

    target = tvm.target.Target({"kind": "metal", "keys": ["metal", "gpu", "metal4"]})
    with target:
        artifact = tilelang.lower(kernel, target=target)
    assert "__pct_c3[15] = " in artifact.kernel_source
