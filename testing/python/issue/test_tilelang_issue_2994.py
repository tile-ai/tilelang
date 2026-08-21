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


def _packed_input(dtype, n):
    values = [(((byte // 8) % 8) << 4) | (byte % 8) for byte in range(n // 2)]
    packed = torch.tensor(values, dtype=torch.uint8, device="cuda")
    return packed if dtype == "uint4" else packed.view(torch.int8)


def _copy_kernel(dtype, n, threads):
    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=threads):
            T.copy(A, B)

    return kernel


def _packed_x2_select_source(dtype):
    on_true = tirx.Var("on_true", f"{dtype}x2")
    on_false = tirx.Var("on_false", f"{dtype}x2")
    ramp = tirx.Ramp(0, 1, 2)
    limit = tirx.Broadcast(tirx.const(1, "int32"), 2)
    selected = tirx.Select(ramp < limit, on_true, on_false)
    func = tirx.PrimFunc([on_true, on_false], tirx.Evaluate(selected))
    func = func.with_attr("global_symbol", "packed_x2_select")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"packed_x2_select": func})
    build = tvm.get_global_func("target.build.tilelang_cuda")
    return build(mod, tvm.target.Target("cuda")).inspect_source()


def _packed_x2_constructor_source(dtype, constructor):
    name = f"packed_{dtype}_x2_{constructor}"
    if constructor == "broadcast":
        value = -3 if dtype == "int4" else 13
        func = tirx.PrimFunc([], tirx.Evaluate(tirx.Broadcast(tirx.const(value, dtype), 2)))
    elif constructor == "scalar_load":
        buffer = tirx.decl_buffer((4,), dtype=dtype, name="input")
        index = tirx.Ramp(tirx.const(0, "int32"), tirx.const(2, "int32"), 2)
        func = tirx.PrimFunc(
            [buffer.data],
            tirx.Evaluate(tirx.BufferLoad(buffer, [index])),
            buffer_map={buffer.data: buffer},
        )
    elif constructor == "shuffle":
        low = tirx.Var("low", dtype)
        high = tirx.Var("high", dtype)
        func = tirx.PrimFunc([low, high], tirx.Evaluate(tirx.Shuffle([low, high], [0, 1])))
    elif constructor == "ramp":
        ramp = tirx.Ramp(tirx.const(0, dtype), tirx.const(1, dtype), 2)
        func = tirx.PrimFunc([], tirx.Evaluate(ramp))
    else:
        raise ValueError(f"Unknown constructor: {constructor}")

    func = func.with_attr("global_symbol", name)
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({name: func})
    build = tvm.get_global_func("target.build.tilelang_cuda")
    return build(mod, tvm.target.Target("cuda")).inspect_source()


def _packed_vector_load_source(dtype, base, stride, dynamic_base=False, lanes=2):
    name = f"packed_{dtype}_x{lanes}_load"
    buffer = tirx.decl_buffer((64,), dtype=dtype, name="input")
    params = [buffer.data]

    if dynamic_base:
        base_expr = tirx.Var("base", "int32")
        params.append(base_expr)
    else:
        base_expr = tirx.const(base, "int32")

    index = tirx.Ramp(
        base_expr,
        tirx.const(stride, "int32"),
        lanes,
    )
    func = tirx.PrimFunc(
        params,
        tirx.Evaluate(tirx.BufferLoad(buffer, [index])),
        buffer_map={buffer.data: buffer},
    )
    func = func.with_attr("global_symbol", name)
    func = func.with_attr(
        "calling_conv",
        tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH,
    )
    mod = tvm.IRModule({name: func})
    build = tvm.get_global_func("target.build.tilelang_cuda")
    return build(mod, tvm.target.Target("cuda")).inspect_source()


def _packed_vector_store_source(
    dtype,
    base,
    stride,
    *,
    dynamic_base=False,
    dynamic_scale=1,
    lanes=2,
):
    name = f"packed_{dtype}_x{lanes}_store"
    buffer = tirx.decl_buffer((64,), dtype=dtype, name="output")
    value = tirx.Var("value", f"{dtype}x{lanes}")
    params = [buffer.data, value]

    if dynamic_base:
        dynamic = tirx.Var("base", "int32")
        params.append(dynamic)
        base_expr = dynamic * tirx.const(dynamic_scale, "int32") + tirx.const(
            base,
            "int32",
        )
    else:
        base_expr = tirx.const(base, "int32")

    index = tirx.Ramp(
        base_expr,
        tirx.const(stride, "int32"),
        lanes,
    )
    func = tirx.PrimFunc(
        params,
        tirx.BufferStore(buffer, value, [index]),
        buffer_map={buffer.data: buffer},
    )
    func = func.with_attr("global_symbol", name)
    func = func.with_attr(
        "calling_conv",
        tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH,
    )
    mod = tvm.IRModule({name: func})
    build = tvm.get_global_func("target.build.tilelang_cuda")
    return build(mod, tvm.target.Target("cuda")).inspect_source()


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


def _packed_x2_gather_kernel(dtype):
    @T.prim_func
    def kernel(
        source: T.Tensor((8,), dtype),
        output: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, threads=1):
            output[T.Ramp(0, 1, 2)] = source[T.Ramp(0, 2, 2)]

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


def _packed_int4_scalar_load_kernel():
    @T.prim_func
    def kernel(
        source: T.Tensor((4,), "int4"),
        output: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                output[i] = T.cast(source[i], "int32")

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("dtype", "storage_type"), [("int4", "int8_t"), ("uint4", "uint8_t")])
def test_packed_int4x2_copy_round_trip(dtype, storage_type):
    n = 256
    kernel = _copy_kernel(dtype, n, threads=128)
    compiled = tilelang.compile(kernel, out_idx=[1])

    assert storage_type in compiled.get_kernel_source()
    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_lane_select_codegen(dtype):
    source = _packed_x2_select_source(dtype)
    assignments = [line for line in source.splitlines() if "on_true" in line and "on_false" in line and "=" in line]

    assert len(assignments) == 2
    assert all("((const unsigned char*)" in line for line in assignments)
    assert all(line.startswith("  ((unsigned char*)") for line in assignments)
    assert all(f"on_true.{lane}" not in source for lane in "xy")
    assert all(f"on_false.{lane}" not in source for lane in "xy")
    assert ">> 0" in assignments[0] and "<< 0" in assignments[0]
    assert ">> 4" in assignments[1] and "<< 4" in assignments[1]
    assert all(("^ 8) - 8" in line) == (dtype == "int4") for line in assignments)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constructor", ["broadcast", "scalar_load", "shuffle", "ramp"])
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_constructor_codegen(dtype, constructor):
    source = _packed_x2_constructor_source(dtype, constructor)
    helper = "tl_pack_uint4x2" if dtype == "uint4" else "tl_pack_int4x2"
    calls = [line for line in source.splitlines() if f"{helper}(" in line and not line.lstrip().startswith("TL_DEVICE")]

    assert len(calls) == 1
    assert "make_int8_t(" not in source
    assert "make_uint8_t(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize(
    ("base", "stride", "dynamic_base"),
    [
        (0, 2, False),
        (1, 1, False),
        (0, 1, True),
    ],
)
def test_packed_int4x2_gather_uses_logical_indices(
    dtype,
    base,
    stride,
    dynamic_base,
):
    source = _packed_vector_load_source(
        dtype,
        base,
        stride,
        dynamic_base=dynamic_base,
    )

    assert source.count(f"tl_{dtype}_packed_load(") == 2
    assert source.count(f"tl_pack_{dtype}x2(") == 1
    pack_line = next(line for line in source.splitlines() if f"tl_pack_{dtype}x2(" in line)
    assert pack_line.count("v_.x") == 1
    assert pack_line.count("v_.y") == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("dtype", "storage_type"),
    [
        ("int4", "int8_t"),
        ("uint4", "uint8_t"),
    ],
)
def test_packed_int4x2_aligned_load_keeps_direct_path(dtype, storage_type):
    source = _packed_vector_load_source(dtype, base=2, stride=1)

    assert f"*((({storage_type}*)input) + 1);" in source
    assert f"tl_{dtype}_packed_load(" not in source
    assert f"tl_pack_{dtype}x2(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("dtype", "storage_type"),
    [
        ("int4", "int8_t"),
        ("uint4", "uint8_t"),
    ],
)
@pytest.mark.parametrize(
    ("base", "dynamic_base", "dynamic_scale"),
    [
        (2, False, 1),
        (0, True, 2),
        (2, True, 4),
    ],
)
def test_packed_int4x2_aligned_store_keeps_direct_path(
    dtype,
    storage_type,
    base,
    dynamic_base,
    dynamic_scale,
):
    source = _packed_vector_store_source(
        dtype,
        base,
        stride=1,
        dynamic_base=dynamic_base,
        dynamic_scale=dynamic_scale,
    )

    assert f"(({storage_type}*)output)" in source
    assert f"tl_{dtype}_packed_store(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize(
    ("base", "stride", "dynamic_base", "dynamic_scale"),
    [
        (1, 1, False, 1),
        (0, 2, False, 1),
        (0, 1, True, 1),
        (1, 1, True, 2),
    ],
)
def test_packed_int4x2_unsafe_store_rejected(
    dtype,
    base,
    stride,
    dynamic_base,
    dynamic_scale,
):
    with pytest.raises(tvm.TVMError, match="provably divisible by the vector lane count"):
        _packed_vector_store_source(
            dtype,
            base,
            stride,
            dynamic_base=dynamic_base,
            dynamic_scale=dynamic_scale,
        )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8])
@pytest.mark.parametrize("access", ["load", "store"])
def test_packed_int4_wide_aligned_access_keeps_direct_path(
    dtype,
    lanes,
    access,
):
    if access == "load":
        constant_source = _packed_vector_load_source(
            dtype,
            base=lanes,
            stride=1,
            lanes=lanes,
        )
        symbolic_source = None
    else:
        constant_source = _packed_vector_store_source(
            dtype,
            base=lanes,
            stride=1,
            lanes=lanes,
        )
        symbolic_source = _packed_vector_store_source(
            dtype,
            base=0,
            stride=1,
            dynamic_base=True,
            dynamic_scale=lanes,
            lanes=lanes,
        )

    carrier_type = {
        ("int4", 4): "int16_t",
        ("uint4", 4): "uint16_t",
        ("int4", 8): "int",
        ("uint4", 8): "uint",
    }[(dtype, lanes)]
    buffer_name = "input" if access == "load" else "output"
    expected_access = f"*((({carrier_type}*){buffer_name}) + 1)"

    if access == "load":
        assert f"{expected_access};" in constant_source
    else:
        assert f"{expected_access} = value;" in constant_source
    assert f"tl_{dtype}_packed_load(" not in constant_source
    assert f"tl_{dtype}_packed_store(" not in constant_source

    if symbolic_source is not None:
        assert f"tl_{dtype}_packed_store(" not in symbolic_source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8])
@pytest.mark.parametrize("access", ["load", "store"])
def test_packed_int4_wide_unsafe_access_rejected(dtype, lanes, access):
    unsafe_cases = [
        (1, 1, False, 1),
        (lanes // 2, 1, False, 1),
        (0, 2, False, 1),
        (0, 1, True, 1),
    ]
    for base, stride, dynamic_base, dynamic_scale in unsafe_cases:
        with pytest.raises(
            tvm.TVMError,
            match="provably divisible by the vector lane count",
        ):
            if access == "load":
                _packed_vector_load_source(
                    dtype,
                    base,
                    stride,
                    dynamic_base=dynamic_base,
                    lanes=lanes,
                )
            else:
                _packed_vector_store_source(
                    dtype,
                    base,
                    stride,
                    dynamic_base=dynamic_base,
                    dynamic_scale=dynamic_scale,
                    lanes=lanes,
                )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("int4", [0xF8, 0x72, 0x05, 0x00], 0x28),
        ("uint4", [0x21, 0x43, 0x65, 0x07], 0x31),
    ],
)
def test_packed_int4x2_strided_gather_runtime(dtype, values, expected):
    compiled = tilelang.compile(_packed_x2_gather_kernel(dtype), out_idx=[1])
    source = torch.tensor(values, dtype=torch.uint8, device="cuda")
    if dtype == "int4":
        source = source.view(torch.int8)

    result = compiled(source)
    kernel_source = compiled.get_kernel_source()

    assert kernel_source.count(f"tl_{dtype}_packed_load(") == 2
    assert kernel_source.count(f"tl_pack_{dtype}x2(") == 1
    assert result.view(torch.uint8).item() == expected


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_aligned_store_preserves_neighboring_bytes(dtype):
    compiled = tilelang.compile(_packed_x2_store_kernel(dtype))
    storage_dtype = torch.uint8 if dtype == "uint4" else torch.int8
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
    assert f"tl_{dtype}_packed_store(" not in compiled.get_kernel_source()
    assert output.dtype == storage_dtype


@tilelang.testing.requires_cuda
def test_packed_int4_scalar_load_sign_extension():
    compiled = tilelang.compile(_packed_int4_scalar_load_kernel(), out_idx=[1])
    source = torch.tensor(
        [0xF8, 0x70],
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.int8)

    result = compiled(source)
    expected = torch.tensor(
        [-8, -1, 0, 7],
        dtype=torch.int32,
        device="cuda",
    )

    assert torch.equal(result, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("dtype", "value"), [("int4", -3), ("uint4", 13)])
def test_packed_int4x2_broadcast_runtime(dtype, value):
    compiled = tilelang.compile(_packed_x2_broadcast_kernel(dtype, value))
    storage_dtype = torch.uint8 if dtype == "uint4" else torch.int8
    output = torch.empty((1,), dtype=storage_dtype, device="cuda")

    compiled(output)

    assert output.view(torch.uint8).item() == 0xDD
    assert f"tl_pack_{dtype}x2(" in compiled.get_kernel_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("dtype", "offset", "expected"), [("int4", -3, 0xED), ("uint4", 2, 0x32)])
def test_packed_int4x2_distinct_lane_runtime(dtype, offset, expected):
    compiled = tilelang.compile(_packed_x2_ramp_kernel(dtype, offset))
    storage_dtype = torch.uint8 if dtype == "uint4" else torch.int8
    output = torch.empty((1,), dtype=storage_dtype, device="cuda")

    compiled(output)

    assert output.view(torch.uint8).item() == expected
    assert f"tl_pack_{dtype}x2(" in compiled.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
