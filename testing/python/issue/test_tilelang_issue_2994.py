import pytest
import torch
import tvm_ffi

import tilelang
import tilelang.testing
import tvm_tirx
from tilelang import language as T
from tvm_tirx import ir, prim, tirx


@pytest.fixture(autouse=True)
def _disable_tilelang_cache():
    tilelang.disable_cache()
    try:
        yield
    finally:
        tilelang.enable_cache()


def _generated_call_lines(source, function):
    return [line for line in source.splitlines() if f"{function}(" in line and not line.lstrip().startswith("TL_DEVICE")]


def _generated_call_count(source, function):
    return sum(line.count(f"{function}(") for line in _generated_call_lines(source, function))


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


def _device_function(params, body):
    # The terminal native device-codegen ABI uses typed pointer parameters.
    # Explicit tensor views replace the old PrimFunc.buffer_map side table.
    device_params = []
    declarations = []
    for param in params:
        if isinstance(param.ty, tirx.TensorType):
            pointer = ir.Var(param.name, ir.PtrType(param.ty.dtype, param.ty.scope))
            device_params.append(pointer)
            declarations.append(ir.Bind(param, ir.Call("tirx.decl_tensor", [pointer], ty_args=[param.ty])))
        else:
            device_params.append(param)
    function = tirx.Function(device_params, ir.SeqStmt([*declarations, body]))
    assert tirx.analysis.check_well_formed(function)
    return function


def _packed_x2_select_source(dtype):
    on_true = ir.Var("on_true", f"{dtype}x2")
    on_false = ir.Var("on_false", f"{dtype}x2")
    ramp = prim.Ramp(0, 1, 2)
    limit = prim.Broadcast(ir.const(1, "int32"), 2)
    selected = prim.Select(ramp < limit, on_true, on_false)
    func = _device_function([on_true, on_false], ir.Evaluate(selected))
    func = func.with_attr("global_symbol", "packed_x2_select")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm_tirx.IRModule({"packed_x2_select": func})
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    return build(mod, tvm_tirx.target.Target("cuda")).inspect_source()


def _packed_x2_constructor_source(dtype, constructor):
    name = f"packed_{dtype}_x2_{constructor}"
    if constructor == "broadcast":
        value = -3 if dtype == "int4" else 13
        func = _device_function([], ir.Evaluate(prim.Broadcast(ir.const(value, dtype), 2)))
    elif constructor == "scalar_load":
        buffer = ir.Var("input", tirx.TensorType((4,), dtype, layout="default"))
        index = prim.Ramp(ir.const(0, "int32"), ir.const(2, "int32"), 2)
        func = _device_function(
            [buffer],
            ir.Evaluate(ir.TensorLoad(buffer, [index])),
        )
    elif constructor == "shuffle":
        low = ir.Var("low", dtype)
        high = ir.Var("high", dtype)
        func = _device_function([low, high], ir.Evaluate(prim.Shuffle([low, high], [0, 1])))
    elif constructor == "ramp":
        # Native Ramp carries index-width arithmetic; narrow each lane only
        # after constructing it, as the vectorized public kernel does.
        ramp = prim.Cast(f"{dtype}x2", prim.Ramp(ir.const(0, "int32"), ir.const(1, "int32"), 2))
        func = _device_function([], ir.Evaluate(ramp))
    else:
        raise ValueError(f"Unknown constructor: {constructor}")

    func = func.with_attr("global_symbol", name)
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm_tirx.IRModule({name: func})
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    return build(mod, tvm_tirx.target.Target("cuda")).inspect_source()


def _packed_vector_load_source(
    dtype,
    base,
    stride,
    *,
    dynamic_base=False,
    dynamic_scale=1,
    lanes=2,
):
    name = f"packed_{dtype}_x{lanes}_load"
    buffer = ir.Var("input", tirx.TensorType((64,), dtype, layout="default"))
    params = [buffer]

    if dynamic_base:
        dynamic = ir.Var("base", "int32")
        params.append(dynamic)
        base_expr = dynamic * ir.const(dynamic_scale, "int32") + ir.const(
            base,
            "int32",
        )
    else:
        base_expr = ir.const(base, "int32")

    index = prim.Ramp(
        base_expr,
        ir.const(stride, "int32"),
        lanes,
    )
    func = _device_function(
        params,
        ir.Evaluate(ir.TensorLoad(buffer, [index])),
    )
    func = func.with_attr("global_symbol", name)
    func = func.with_attr(
        "calling_conv",
        tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH,
    )
    mod = tvm_tirx.IRModule({name: func})
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    return build(mod, tvm_tirx.target.Target("cuda")).inspect_source()


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
    buffer = ir.Var("output", tirx.TensorType((64,), dtype, layout="default"))
    value = ir.Var("value", f"{dtype}x{lanes}")
    params = [buffer, value]

    if dynamic_base:
        dynamic = ir.Var("base", "int32")
        params.append(dynamic)
        base_expr = dynamic * ir.const(dynamic_scale, "int32") + ir.const(
            base,
            "int32",
        )
    else:
        base_expr = ir.const(base, "int32")

    index = prim.Ramp(
        base_expr,
        ir.const(stride, "int32"),
        lanes,
    )
    func = _device_function(
        params,
        ir.TensorStore(buffer, [index], value),
    )
    func = func.with_attr("global_symbol", name)
    func = func.with_attr(
        "calling_conv",
        tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH,
    )
    mod = tvm_tirx.IRModule({name: func})
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    return build(mod, tvm_tirx.target.Target("cuda")).inspect_source()


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

    if constructor == "ramp":
        # Native index Ramp followed by a narrow cast packs each lane into the
        # same byte. Verify both disjoint nibble writes; runtime tests below
        # also check the exact packed result for distinct signed/unsigned lanes.
        stores = [line for line in source.splitlines() if line.lstrip().startswith("((unsigned char*)") and " = " in line]
        assert len(stores) == 2
        assert "v_.x" in stores[0] and "& 0x0fu" in stores[0] and "<< 0" in stores[0]
        assert "v_.y" in stores[1] and "& 0x0fu" in stores[1] and "<< 4" in stores[1]
    else:
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

    pack_lines = _generated_call_lines(source, f"tl_pack_{dtype}x2")
    assert _generated_call_count(source, f"tl_{dtype}_packed_load") == 2
    assert len(pack_lines) == 1
    pack_line = pack_lines[0]
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
def test_packed_int4x2_symbolic_aligned_load_keeps_direct_path(dtype, storage_type):
    source = _packed_vector_load_source(
        dtype,
        base=0,
        stride=1,
        dynamic_base=True,
        dynamic_scale=2,
    )

    assert f"(({storage_type}*)input)" in source
    assert not _generated_call_lines(source, f"tl_{dtype}_packed_load")
    assert not _generated_call_lines(source, f"tl_pack_{dtype}x2")


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
    with pytest.raises(RuntimeError, match="provably divisible by the vector lane count"):
        _packed_vector_store_source(
            dtype,
            base,
            stride,
            dynamic_base=dynamic_base,
            dynamic_scale=dynamic_scale,
        )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8, 16, 32])
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
        symbolic_source = _packed_vector_load_source(
            dtype,
            base=0,
            stride=1,
            dynamic_base=True,
            dynamic_scale=lanes,
            lanes=lanes,
        )
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
        ("int4", 16): "int2",
        ("uint4", 16): "uint2",
        ("int4", 32): "int4",
        ("uint4", 32): "uint4",
    }[(dtype, lanes)]
    buffer_name = "input" if access == "load" else "output"
    expected_access = f"*((({carrier_type}*){buffer_name}) + 1)"

    if access == "load":
        assert f"{expected_access};" in constant_source
    else:
        assert f"{expected_access} = value;" in constant_source
    assert f"tl_{dtype}_packed_load(" not in constant_source
    assert f"tl_{dtype}_packed_store(" not in constant_source

    helper = f"tl_{dtype}_packed_load" if access == "load" else f"tl_{dtype}_packed_store"
    assert f"(({carrier_type}*){buffer_name})" in symbolic_source
    assert not _generated_call_lines(symbolic_source, helper)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8, 16, 32])
def test_packed_int4_wide_unsafe_store_rejected(dtype, lanes):
    unsafe_cases = [
        (1, 1, False, 1),
        (lanes // 2, 1, False, 1),
        (0, 2, False, 1),
        (0, 1, True, 1),
    ]
    for base, stride, dynamic_base, dynamic_scale in unsafe_cases:
        with pytest.raises(
            RuntimeError,
            match="provably divisible by the vector lane count",
        ):
            _packed_vector_store_source(
                dtype,
                base,
                stride,
                dynamic_base=dynamic_base,
                dynamic_scale=dynamic_scale,
                lanes=lanes,
            )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8, 16, 32])
@pytest.mark.parametrize(
    ("base", "stride", "dynamic_base"),
    [
        (1, 1, False),
        (0, 2, False),
        (0, 1, True),
    ],
)
def test_packed_int4_wide_unaligned_load_uses_logical_indices(
    dtype,
    lanes,
    base,
    stride,
    dynamic_base,
):
    source = _packed_vector_load_source(
        dtype,
        base,
        stride,
        dynamic_base=dynamic_base,
        lanes=lanes,
    )

    assert _generated_call_count(source, f"tl_{dtype}_packed_load") == lanes


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [4, 8])
def test_packed_int4_wide_strided_gather_runtime(dtype, lanes):
    base = 1
    stride = 2
    compiled = tilelang.compile(
        _packed_vector_gather_kernel(dtype, lanes, base, stride),
        out_idx=[1],
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
    assert _generated_call_count(compiled.get_kernel_source(), f"tl_{dtype}_packed_load") == lanes


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

    assert _generated_call_count(kernel_source, f"tl_{dtype}_packed_load") == 2
    assert len(_generated_call_lines(kernel_source, f"tl_pack_{dtype}x2")) == 1
    assert result.view(torch.uint8).item() == expected


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4x2_aligned_store_preserves_neighboring_bytes(dtype):
    compiled = tilelang.compile(_packed_x2_store_kernel(dtype))
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
    # This exact byte distinguishes the two lanes even when native lowering
    # constructs the packed carrier with masked lane stores instead of a helper.


if __name__ == "__main__":
    tilelang.testing.main()
