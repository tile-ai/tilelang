import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


THREADS = 128
COUNTS = (16, 8, 4, 2, 1)

F32 = "float32"
F16 = "float16"
FP8_E4M3 = "float8_e4m3fn"
FP8_E5M2 = "float8_e5m2"
FP8_TORCH_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@tilelang.jit(target="ascend", out_idx=[1])
def cast_kernel(n: int, in_dtype: T.dtype, out_dtype: T.dtype):
    @T.prim_func
    def main(
        X: T.Tensor((n,), in_dtype),
        Y: T.Tensor((n,), out_dtype),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), in_dtype)
            y_ub = T.alloc_shared((n,), out_dtype)

            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                x_local = T.alloc_fragment((n,), in_dtype)
                y_local = T.alloc_fragment((n,), out_dtype)

                T.copy(x_ub, x_local)
                T.copy(x_local, y_local)
                T.copy(y_local, y_ub)
            T.copy(y_ub, Y)

    return main


CASES = (
    ("fp32->e4m3", F32, FP8_E4M3, torch.float32, torch.float8_e4m3fn, "e4m3_range"),
    ("e4m3->fp32", FP8_E4M3, F32, torch.float8_e4m3fn, torch.float32, "e4m3_range"),
    ("half->e4m3", F16, FP8_E4M3, torch.float16, torch.float8_e4m3fn, "e4m3_range"),
    ("e4m3->half", FP8_E4M3, F16, torch.float8_e4m3fn, torch.float16, "e4m3_range"),
    ("fp32->e5m2", F32, FP8_E5M2, torch.float32, torch.float8_e5m2, "e5m2_range"),
    ("e5m2->fp32", FP8_E5M2, F32, torch.float8_e5m2, torch.float32, "e5m2_range"),
    ("half->e5m2", F16, FP8_E5M2, torch.float16, torch.float8_e5m2, "e5m2_range"),
    ("e5m2->half", FP8_E5M2, F16, torch.float8_e5m2, torch.float16, "e5m2_range"),
    ("fp32->half", F32, F16, torch.float32, torch.float16, "small_range"),
    ("half->fp32", F16, F32, torch.float16, torch.float32, "small_range"),
)


def make_input(n: int, torch_dtype: torch.dtype, value_kind: str) -> torch.Tensor:
    if value_kind == "e4m3_range":
        x = torch.linspace(-448.0, 448.0, n, device="npu", dtype=torch.float32)
    elif value_kind == "e5m2_range":
        x = torch.linspace(-57344.0, 57344.0, n, device="npu", dtype=torch.float32)
    else:
        x = torch.arange(n, device="npu", dtype=torch.float32) - (n // 2)
    return x.to(torch_dtype)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("elems_per_thread", COUNTS)
def test_simtvf_cast(case, elems_per_thread):
    name, in_tl_dtype, out_tl_dtype, in_torch_dtype, out_torch_dtype, value_kind = case
    n = THREADS * elems_per_thread

    kernel = cast_kernel(n, in_tl_dtype, out_tl_dtype)
    x = make_input(n, in_torch_dtype, value_kind)
    y = kernel(x)
    torch.npu.synchronize()

    ref = x.to(out_torch_dtype)
    if y.dtype in FP8_TORCH_DTYPES:
        y, ref = y.view(torch.uint8), ref.view(torch.uint8)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def _broadcast_cast_arithmetic():
    @T.prim_func
    def main(
        x: T.Tensor((64,), "float32"),
        y: T.Tensor((128,), "bfloat16"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((64,), "float32")
            y_ub = T.alloc_shared((128,), "bfloat16")
            T.copy(x, x_ub)
            with T.SimtVF(threads=64):
                tx = T.get_thread_binding()
                for i in T.vectorized(2):
                    y_ub[tx * 2 + i] = T.cast(x_ub[tx] * x_ub[tx] + 1.0, "bfloat16")
            T.copy(y_ub, y)

    return main


def test_broadcast_cast_arbitrary_expression_uses_vector_cast():
    kernel = tilelang.compile(_broadcast_cast_arithmetic(), target="ascend")
    source = kernel.get_kernel_source()
    assert "make_float2" in source
    assert "__float22bfloat162_rn" in source

    x = torch.linspace(-2, 2, 64, dtype=torch.float32, device="npu")
    y = torch.empty(128, dtype=torch.bfloat16, device="npu")
    kernel(x, y)
    torch.npu.synchronize()

    expected = (x.cpu() * x.cpu() + 1).to(torch.bfloat16).repeat_interleave(2)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
