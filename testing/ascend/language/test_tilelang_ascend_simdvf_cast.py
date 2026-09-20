import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


N = 128

F32 = "float32"
F16 = "float16"
BF16 = "bfloat16"
FP8_E4M3 = "float8_e4m3fn"
FP8_E5M2 = "float8_e5m2"
FP4_E2M1 = "float4_e2m1fn"
INT32 = "int32"
INT16 = "int16"
INT8 = "int8"
UINT8 = "uint8"
FP8_TYPES = (FP8_E4M3, FP8_E5M2)
F16_TYPES = (F16, BF16)
INT_TYPES = (INT32, INT16, INT8, UINT8)
FP8_TORCH_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP4_E2M1_CODES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@tilelang.jit(target="ascend", out_idx=[1])
def cast_kernel(n: int, in_dtype: T.dtype, out_dtype: T.dtype):
    is_fp8_in = in_dtype in FP8_TYPES
    is_fp8_out = out_dtype in FP8_TYPES

    @T.prim_func
    def main(
        X: T.Tensor((n,), in_dtype),
        Y: T.Tensor((n,), out_dtype),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), in_dtype)
            y_ub = T.alloc_shared((n,), out_dtype)

            T.copy(X, x_ub)

            with T.SimdVF():
                if in_dtype == F32:
                    if out_dtype == F16:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64])
                            y = T.simd.vcvt(x, "float16")
                            T.simd.vsts(y_ub[i * 64], y, dist="PK_B32")
                    elif is_fp8_out:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64])
                            y = T.simd.vcvt(x, out_dtype)
                            T.simd.vsts(y_ub[i * 64], y, dist="PK4_B32")
                    elif out_dtype == INT32:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64])
                            y = T.simd.vcvt(x, "int32")
                            T.simd.vsts(y_ub[i * 64], y)
                elif in_dtype in F16_TYPES:
                    if is_fp8_out:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK_B16")
                            x_f32 = T.simd.vcvt(x, "float32", part=0)
                            y = T.simd.vcvt(x_f32, out_dtype)
                            T.simd.vsts(y_ub[i * 64], y, dist="PK4_B32")
                    elif in_dtype == BF16 and out_dtype == FP4_E2M1:
                        for i in range(n // 128):
                            x = T.simd.vld(x_ub[i * 128])
                            y = T.simd.vcvt(x, out_dtype)
                            T.simd.vsts(y_ub[i * 128], y, dist="PK4_B32")
                    elif in_dtype == F16 and out_dtype == F32:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK_B16")
                            y = T.simd.vcvt(x, "float32", part=0)
                            T.simd.vsts(y_ub[i * 64], y)
                elif in_dtype == FP4_E2M1:
                    if out_dtype == BF16:
                        for i in range(n // 128):
                            x = T.simd.vld(x_ub[i * 128], dist="UNPK4_B8")
                            y = T.simd.vcvt(x, out_dtype)
                            T.simd.vsts(y_ub[i * 64], y)
                elif is_fp8_in:
                    if out_dtype == F32:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK4_B8")
                            y = T.simd.vcvt(x, "float32")
                            T.simd.vsts(y_ub[i * 64], y)
                    elif out_dtype in F16_TYPES:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK4_B8")
                            y_f32 = T.simd.vcvt(x, "float32")
                            y = T.simd.vcvt(y_f32, out_dtype)
                            T.simd.vsts(y_ub[i * 64], y, dist="PK_B32")
                elif in_dtype == INT32:
                    if out_dtype == F32:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64])
                            y = T.simd.vcvt(x, "float32")
                            T.simd.vsts(y_ub[i * 64], y)
                    elif out_dtype == INT16:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64])
                            y = T.simd.vcvt(x, "int16")
                            T.simd.vsts(y_ub[i * 64], y, dist="PK_B32")
                elif in_dtype == INT16:
                    if out_dtype == INT32:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK_B16")
                            y = T.simd.vcvt(x, "int32", part=0)
                            T.simd.vsts(y_ub[i * 64], y)
                    elif out_dtype == UINT8:
                        for i in range(n // 64):
                            x = T.simd.vld(x_ub[i * 64], dist="UNPK_B16")
                            y = T.simd.vcvt(x, out_dtype)
                            T.simd.vsts(y_ub[i * 64], y, dist="PK4_B32")
                elif in_dtype == INT8 and out_dtype == INT16:
                    for i in range(n // 64):
                        x = T.simd.vld(x_ub[i * 64], dist="UNPK4_B8")
                        y = T.simd.vcvt(x, "int16")
                        T.simd.vsts(y_ub[i * 64], y, dist="PK_B32")

            T.copy(y_ub, Y)

    return main


CASES = (
    ("fp32->e4m3", F32, FP8_E4M3, torch.float32, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->fp32", FP8_E4M3, F32, torch.float8_e4m3fn, torch.float32, "e4m3"),
    ("half->e4m3", F16, FP8_E4M3, torch.float16, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->half", FP8_E4M3, F16, torch.float8_e4m3fn, torch.float16, "e4m3"),
    ("bf16->e4m3", BF16, FP8_E4M3, torch.bfloat16, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->bf16", FP8_E4M3, BF16, torch.float8_e4m3fn, torch.bfloat16, "e4m3"),
    ("bf16->e2m1x2", BF16, FP4_E2M1, torch.bfloat16, torch.int8, "e2m1"),
    ("e2m1x2->bf16", FP4_E2M1, BF16, torch.int8, torch.bfloat16, "e2m1"),
    ("fp32->half", F32, F16, torch.float32, torch.float16, "int"),
    ("half->fp32", F16, F32, torch.float16, torch.float32, "int"),
    ("s32->f32", INT32, F32, torch.int32, torch.float32, "int"),
    ("f32->s32", F32, INT32, torch.float32, torch.int32, "int"),
    ("s32->s16", INT32, INT16, torch.int32, torch.int16, "int"),
    ("s16->s32", INT16, INT32, torch.int16, torch.int32, "int"),
    ("s8->s16", INT8, INT16, torch.int8, torch.int16, "int"),
    ("s16->u8", INT16, UINT8, torch.int16, torch.uint8, "uint"),
)


def fp4_e2m1_codes(n: int, device: str = "npu") -> torch.Tensor:
    repeats = (n + FP4_E2M1_CODES.numel() - 1) // FP4_E2M1_CODES.numel()
    return FP4_E2M1_CODES.repeat(repeats)[:n].to(device=device)


def make_input(n: int, torch_dtype: torch.dtype, value_kind: str) -> torch.Tensor:
    if value_kind == "e4m3":
        x = torch.linspace(-448.0, 448.0, n, device="npu", dtype=torch.float32)
    elif value_kind == "e5m2":
        x = torch.linspace(-57344.0, 57344.0, n, device="npu", dtype=torch.float32)
    elif value_kind == "e2m1":
        if torch_dtype == torch.int8:
            # packed fp4 input: 2 codes/byte -> int8[n//2].
            codes = fp4_e2m1_codes(n)
            packed = codes[0::2] | (codes[1::2] << 4)
            return packed.view(torch.int8)
        idx = torch.arange(n, dtype=torch.long, device=FP4_E2M1_VALUES.device) % FP4_E2M1_VALUES.numel()
        x = FP4_E2M1_VALUES[idx].to(device="npu")
    elif value_kind == "uint":
        x = torch.arange(n, device="npu", dtype=torch.int32)
    else:
        x = torch.arange(n, device="npu", dtype=torch.float32) - (n // 2)
    return x.to(torch_dtype)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_simdvf_cast(case):
    name, in_tl_dtype, out_tl_dtype, in_torch_dtype, out_torch_dtype, value_kind = case

    kernel = cast_kernel(N, in_tl_dtype, out_tl_dtype)
    x = make_input(N, in_torch_dtype, value_kind)

    y = kernel(x)
    torch.npu.synchronize()

    if out_tl_dtype == FP4_E2M1:
        y = y.view(torch.uint8)
        codes = fp4_e2m1_codes(y.numel() * 2, device=y.device)
        ref = codes[0::2] | (codes[1::2] << 4)
    elif in_tl_dtype == FP4_E2M1 and out_tl_dtype == BF16:
        idx = torch.arange(y.numel(), dtype=torch.long, device=FP4_E2M1_VALUES.device) % FP4_E2M1_VALUES.numel()
        ref = FP4_E2M1_VALUES[idx].to(device=y.device, dtype=y.dtype)
    else:
        ref = x.to(out_torch_dtype)
    if y.dtype in FP8_TORCH_DTYPES:
        y, ref = y.view(torch.uint8), ref.view(torch.uint8)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
