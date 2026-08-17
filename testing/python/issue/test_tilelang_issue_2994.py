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


if __name__ == "__main__":
    tilelang.testing.main()
