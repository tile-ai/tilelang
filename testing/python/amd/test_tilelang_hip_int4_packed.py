import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T, tvm
from tvm import tirx


@pytest.fixture(autouse=True)
def _disable_tilelang_cache():
    tilelang.disable_cache()
    try:
        yield
    finally:
        tilelang.enable_cache()


def _build_source(name, func):
    func = func.with_attr("global_symbol", name)
    func = func.with_attr(
        "calling_conv",
        tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH,
    )
    mod = tvm.IRModule({name: func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


def _generated_call_lines(source, function):
    return [line for line in source.splitlines() if f"{function}(" in line and not line.lstrip().startswith("TL_DEVICE")]


def _generated_call_count(source, function):
    return sum(line.count(f"{function}(") for line in _generated_call_lines(source, function))


def _packed_x2_constructor_source(dtype, constructor):
    name = f"packed_{dtype}_x2_{constructor}"
    if constructor == "broadcast":
        value = -3 if dtype == "int4" else 13
        body = tirx.Broadcast(tirx.const(value, dtype), 2)
        params = []
    elif constructor == "scalar_load":
        buffer = tirx.decl_buffer((4,), dtype=dtype, name="input")
        index = tirx.Ramp(tirx.const(0, "int32"), tirx.const(2, "int32"), 2)
        body = tirx.BufferLoad(buffer, [index])
        params = [buffer.data]
    elif constructor == "shuffle":
        low = tirx.Var("low", dtype)
        high = tirx.Var("high", dtype)
        body = tirx.Shuffle([low, high], [0, 1])
        params = [low, high]
    elif constructor == "ramp":
        body = tirx.Ramp(tirx.const(0, dtype), tirx.const(1, dtype), 2)
        params = []
    else:
        raise ValueError(f"Unknown constructor: {constructor}")

    return _build_source(name, tirx.PrimFunc(params, tirx.Evaluate(body)))


def _packed_vector_load_source(dtype, base, stride, lanes=2):
    name = f"packed_{dtype}_x{lanes}_load"
    buffer = tirx.decl_buffer((64,), dtype=dtype, name="input")
    index = tirx.Ramp(
        tirx.const(base, "int32"),
        tirx.const(stride, "int32"),
        lanes,
    )
    func = tirx.PrimFunc(
        [buffer.data],
        tirx.Evaluate(tirx.BufferLoad(buffer, [index])),
        buffer_map={buffer.data: buffer},
    )
    return _build_source(name, func)


def _packed_vector_store_source(dtype, base, stride, lanes=2):
    name = f"packed_{dtype}_x{lanes}_store"
    buffer = tirx.decl_buffer((64,), dtype=dtype, name="output")
    value = tirx.Var("value", f"{dtype}x{lanes}")
    index = tirx.Ramp(
        tirx.const(base, "int32"),
        tirx.const(stride, "int32"),
        lanes,
    )
    func = tirx.PrimFunc(
        [buffer.data, value],
        tirx.BufferStore(buffer, value, [index]),
        buffer_map={buffer.data: buffer},
    )
    return _build_source(name, func)


def _packed_x2_broadcast_kernel(dtype, value):
    @T.prim_func
    def kernel(output: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(2):
                output[i] = T.cast(value, dtype)

    return kernel


def _packed_x2_ramp_kernel(dtype, offset):
    @T.prim_func
    def kernel(output: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(2):
                output[i] = T.cast(i + offset, dtype)

    return kernel


def _packed_vector_gather_kernel(dtype, lanes, base, stride):
    @T.prim_func
    def kernel(
        source: T.Tensor((64,), dtype),
        output: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1, threads=1):
            output[T.Ramp(0, 1, lanes)] = source[T.Ramp(base, stride, lanes)]

    return kernel


def _packed_x2_store_kernel(dtype):
    @T.prim_func
    def kernel(
        source: T.Tensor((2,), dtype),
        output: T.Tensor((8,), dtype),
    ):
        with T.Kernel(1, threads=1):
            output[T.Ramp(2, 1, 2)] = source[T.Ramp(0, 1, 2)]

    return kernel


def _packed_scalar_load_kernel():
    @T.prim_func
    def kernel(
        source: T.Tensor((4,), "int4"),
        output: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                output[i] = T.cast(source[i], "int32")

    return kernel


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4_scalar_load_codegen(dtype):
    buffer = tirx.decl_buffer((8,), dtype=dtype, name="input")
    func = tirx.PrimFunc(
        [buffer.data],
        tirx.Evaluate(tirx.BufferLoad(buffer, [tirx.const(3, "int32")])),
        buffer_map={buffer.data: buffer},
    )
    source = _build_source(f"packed_{dtype}_scalar_load", func)

    assert _generated_call_count(source, f"tl_{dtype}_packed_load") == 1
    assert " + 3 / 8" not in source


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("constructor", ["broadcast", "scalar_load", "shuffle", "ramp"])
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_constructor_codegen(dtype, constructor):
    source = _packed_x2_constructor_source(dtype, constructor)
    helper = f"tl_pack_{dtype}x2"

    assert len(_generated_call_lines(source, helper)) == 1
    assert "make_int8_t(" not in source
    assert "make_uint8_t(" not in source


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("dtype", "storage_type"),
    [("int4", "int8_t"), ("uint4", "uint8_t")],
)
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_packed_int4_aligned_load_keeps_direct_path(dtype, storage_type, lanes):
    source = _packed_vector_load_source(dtype, base=lanes, stride=1, lanes=lanes)
    carrier_type = {
        2: storage_type,
        4: "uint16_t" if dtype == "uint4" else "int16_t",
        8: "uint" if dtype == "uint4" else "int",
        16: "uint2" if dtype == "uint4" else "int2",
        32: "uint4" if dtype == "uint4" else "int4",
    }[lanes]

    assert f"(({carrier_type}*)input)" in source
    assert not _generated_call_lines(source, f"tl_{dtype}_packed_load")


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_packed_int4_gather_uses_logical_indices(dtype, lanes):
    source = _packed_vector_load_source(dtype, base=1, stride=2, lanes=lanes)

    assert _generated_call_count(source, f"tl_{dtype}_packed_load") == lanes
    if lanes == 2:
        assert len(_generated_call_lines(source, f"tl_pack_{dtype}x2")) == 1


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_packed_int4_unsafe_store_rejected(dtype, lanes):
    with pytest.raises(
        tvm.TVMError,
        match="provably divisible by the vector lane count",
    ):
        _packed_vector_store_source(dtype, base=1, stride=1, lanes=lanes)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_packed_int4_aligned_store_keeps_direct_path(dtype, lanes):
    source = _packed_vector_store_source(dtype, base=lanes, stride=1, lanes=lanes)

    assert f"tl_{dtype}_packed_store(" not in source
    assert "= value;" in source


@tilelang.testing.requires_rocm
def test_packed_int4_scalar_load_sign_extension():
    compiled = tilelang.compile(
        _packed_scalar_load_kernel(),
        out_idx=[1],
        target="hip",
    )
    source = torch.tensor([0xF8, 0x70], dtype=torch.uint8, device="cuda").view(torch.int8)

    result = compiled(source)
    expected = torch.tensor([-8, -1, 0, 7], dtype=torch.int32, device="cuda")

    assert torch.equal(result, expected)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("dtype", "value"), [("int4", -3), ("uint4", 13)])
def test_packed_int4x2_broadcast_runtime(dtype, value):
    compiled = tilelang.compile(
        _packed_x2_broadcast_kernel(dtype, value),
        target="hip",
    )
    storage_dtype = torch.uint8 if dtype == "uint4" else torch.int8
    output = torch.empty((1,), dtype=storage_dtype, device="cuda")

    compiled(output)

    assert output.view(torch.uint8).item() == 0xDD
    assert f"tl_pack_{dtype}x2(" in compiled.get_kernel_source()


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("dtype", "offset", "expected"),
    [("int4", -3, 0xED), ("uint4", 2, 0x32)],
)
def test_packed_int4x2_distinct_lane_runtime(dtype, offset, expected):
    compiled = tilelang.compile(
        _packed_x2_ramp_kernel(dtype, offset),
        target="hip",
    )
    storage_dtype = torch.uint8 if dtype == "uint4" else torch.int8
    output = torch.empty((1,), dtype=storage_dtype, device="cuda")

    compiled(output)

    assert output.view(torch.uint8).item() == expected


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_packed_int4_strided_gather_runtime(dtype, lanes):
    base = 1
    stride = 2
    compiled = tilelang.compile(
        _packed_vector_gather_kernel(dtype, lanes, base, stride),
        out_idx=[1],
        target="hip",
    )
    logical = [i % 16 for i in range(64)]
    packed = [logical[i] | (logical[i + 1] << 4) for i in range(0, 64, 2)]
    source = torch.tensor(packed, dtype=torch.uint8, device="cuda")
    if dtype == "int4":
        source = source.view(torch.int8)

    result = compiled(source)
    gathered = [logical[base + i * stride] for i in range(lanes)]
    expected = torch.tensor(
        [gathered[i] | (gathered[i + 1] << 4) for i in range(0, lanes, 2)],
        dtype=torch.uint8,
        device="cuda",
    )

    assert torch.equal(result.view(torch.uint8), expected)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_aligned_store_preserves_neighboring_bytes(dtype):
    compiled = tilelang.compile(_packed_x2_store_kernel(dtype), target="hip")
    source = torch.tensor([0x21], dtype=torch.uint8, device="cuda")
    output = torch.tensor(
        [0xBA, 0xDC, 0xFE, 0x98],
        dtype=torch.uint8,
        device="cuda",
    )
    if dtype == "int4":
        source = source.view(torch.int8)
        output = output.view(torch.int8)

    compiled(source, output)

    expected = torch.tensor(
        [0xBA, 0x21, 0xFE, 0x98],
        dtype=torch.uint8,
        device="cuda",
    )
    assert torch.equal(output.view(torch.uint8), expected)


if __name__ == "__main__":
    tilelang.testing.main()
