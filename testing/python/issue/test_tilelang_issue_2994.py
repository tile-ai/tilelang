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
