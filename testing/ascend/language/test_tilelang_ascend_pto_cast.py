"""Idiomatic PTO VMI port of test_tilelang_ascend_simdvf_cast.py.

One specialized PrimFunc body per dtype pair (avoids Python nesting limits).
Uses continuous ``vcvt`` / ``vstore``. Integer stores bitcast ``pto.si*``
(``vcvt`` result) to the UB element type. Integer widen stays inside
``T.SimdVF``: continuous ``vload``, ``vinterpret_cast`` to ``si*`` (PTODSL
rejects signless int-to-int widen), then ``vcvt``. Unpack ``vload`` is not
legalized on this VPTO path.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


N = 128

F32 = "float32"
F16 = "float16"
BF16 = "bfloat16"
FP8_E4M3 = "float8_e4m3fn"
INT32 = "int32"
INT16 = "int16"
INT8 = "int8"
UINT8 = "uint8"


def _kernel_f32_f16(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), F32), Y: T.Tensor((n,), F16)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), F32)
                y_ub = T.alloc_shared((n,), F16)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        T.vmi.vstore(T.vmi.vcvt(x, "float16"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_f16_f32(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), F16), Y: T.Tensor((n,), F32)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), F16)
                y_ub = T.alloc_shared((n,), F32)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        T.vmi.vstore(T.vmi.vcvt(x, "float32"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_s32_f32(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), INT32), Y: T.Tensor((n,), F32)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), INT32)
                y_ub = T.alloc_shared((n,), F32)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        T.vmi.vstore(T.vmi.vcvt(x, "float32"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_f32_s32(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), F32), Y: T.Tensor((n,), INT32)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), F32)
                y_ub = T.alloc_shared((n,), INT32)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        # vcvt emits pto.si32; UB is i32, so bitcast si32→i32.
                        as_si = T.vmi.vcvt(x, "int32")
                        T.vmi.vstore(
                            T.vmi.vinterpret_cast(as_si, "int32"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_f32_e4m3(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), F32), Y: T.Tensor((n,), FP8_E4M3)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), F32)
                y_ub = T.alloc_shared((n,), FP8_E4M3)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        T.vmi.vstore(
                            T.vmi.vcvt(x, "float8_e4m3fn", rounding="R", saturate="SAT"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_e4m3_f32(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), FP8_E4M3), Y: T.Tensor((n,), F32)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), FP8_E4M3)
                y_ub = T.alloc_shared((n,), F32)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        T.vmi.vstore(T.vmi.vcvt(x, "float32"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_f16_e4m3(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), F16), Y: T.Tensor((n,), FP8_E4M3)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), F16)
                y_ub = T.alloc_shared((n,), FP8_E4M3)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        mid = T.vmi.vcvt(x, "float32")
                        T.vmi.vstore(
                            T.vmi.vcvt(mid, "float8_e4m3fn", rounding="R", saturate="SAT"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_e4m3_f16(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), FP8_E4M3), Y: T.Tensor((n,), F16)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), FP8_E4M3)
                y_ub = T.alloc_shared((n,), F16)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        mid = T.vmi.vcvt(x, "float32")
                        T.vmi.vstore(T.vmi.vcvt(mid, "float16"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_bf16_e4m3(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), BF16), Y: T.Tensor((n,), FP8_E4M3)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), BF16)
                y_ub = T.alloc_shared((n,), FP8_E4M3)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        mid = T.vmi.vcvt(x, "float32")
                        T.vmi.vstore(
                            T.vmi.vcvt(mid, "float8_e4m3fn", rounding="R", saturate="SAT"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_e4m3_bf16(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), FP8_E4M3), Y: T.Tensor((n,), BF16)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), FP8_E4M3)
                y_ub = T.alloc_shared((n,), BF16)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        mid = T.vmi.vcvt(x, "float32")
                        T.vmi.vstore(T.vmi.vcvt(mid, "bfloat16"), y_ub[i * 64], mask64)
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_s32_s16(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), INT32), Y: T.Tensor((n,), INT16)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), INT32)
                y_ub = T.alloc_shared((n,), INT16)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        # vcvt emits pto.si16; UB is i16, so bitcast si16→i16.
                        as_si = T.vmi.vcvt(x, "int16")
                        T.vmi.vstore(
                            T.vmi.vinterpret_cast(as_si, "int16"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_s16_s32(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), INT16), Y: T.Tensor((n,), INT32)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), INT16)
                y_ub = T.alloc_shared((n,), INT32)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        # Signless i16→si32 vcvt is illegal; bitcast to si16
                        # first. Unpack vload is not legalized by VPTO.
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        as_si = T.vmi.vcvt(T.vmi.vinterpret_cast(x, "si16"), "int32")
                        T.vmi.vstore(
                            T.vmi.vinterpret_cast(as_si, "int32"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_s8_s16(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), INT8), Y: T.Tensor((n,), INT16)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), INT8)
                y_ub = T.alloc_shared((n,), INT16)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        # Signless i8→si16 vcvt is illegal; bitcast to si8
                        # first. Unpack vload is not legalized by VPTO.
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        as_si = T.vmi.vcvt(T.vmi.vinterpret_cast(x, "si8"), "int16")
                        T.vmi.vstore(
                            T.vmi.vinterpret_cast(as_si, "int16"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


def _kernel_s16_u8(n):
    @tilelang.jit(out_idx=[1], target="pto")
    def k():
        @T.prim_func
        def main(X: T.Tensor((n,), INT16), Y: T.Tensor((n,), UINT8)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n,), INT16)
                y_ub = T.alloc_shared((n,), UINT8)
                T.copy(X, x_ub)
                with T.SimdVF():
                    mask64 = T.vmi.create_mask(64, size=64)
                    for i in range(n // 64):
                        x = T.vmi.vload(x_ub[i * 64], size=64)
                        as_u = T.vmi.vcvt(x, "uint8")
                        T.vmi.vstore(
                            T.vmi.vinterpret_cast(as_u, "uint8"),
                            y_ub[i * 64],
                            mask64,
                        )
                T.copy(y_ub, Y)

        return main

    return k()


_FACTORIES = {
    (F32, F16): _kernel_f32_f16,
    (F16, F32): _kernel_f16_f32,
    (INT32, F32): _kernel_s32_f32,
    (F32, INT32): _kernel_f32_s32,
    (F32, FP8_E4M3): _kernel_f32_e4m3,
    (FP8_E4M3, F32): _kernel_e4m3_f32,
    (F16, FP8_E4M3): _kernel_f16_e4m3,
    (FP8_E4M3, F16): _kernel_e4m3_f16,
    (BF16, FP8_E4M3): _kernel_bf16_e4m3,
    (FP8_E4M3, BF16): _kernel_e4m3_bf16,
    (INT32, INT16): _kernel_s32_s16,
    (INT16, INT32): _kernel_s16_s32,
    (INT8, INT16): _kernel_s8_s16,
    (INT16, UINT8): _kernel_s16_u8,
}


def cast_kernel(n: int, in_dtype: str, out_dtype: str):
    return _FACTORIES[(in_dtype, out_dtype)](n)


CASES = (
    ("fp32->half", F32, F16, torch.float32, torch.float16, "int"),
    ("half->fp32", F16, F32, torch.float16, torch.float32, "int"),
    ("s32->f32", INT32, F32, torch.int32, torch.float32, "int"),
    ("f32->s32", F32, INT32, torch.float32, torch.int32, "int"),
    ("fp32->e4m3", F32, FP8_E4M3, torch.float32, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->fp32", FP8_E4M3, F32, torch.float8_e4m3fn, torch.float32, "e4m3"),
    ("half->e4m3", F16, FP8_E4M3, torch.float16, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->half", FP8_E4M3, F16, torch.float8_e4m3fn, torch.float16, "e4m3"),
    ("bf16->e4m3", BF16, FP8_E4M3, torch.bfloat16, torch.float8_e4m3fn, "e4m3"),
    ("e4m3->bf16", FP8_E4M3, BF16, torch.float8_e4m3fn, torch.bfloat16, "e4m3"),
    ("s32->s16", INT32, INT16, torch.int32, torch.int16, "int"),
    ("s16->s32", INT16, INT32, torch.int16, torch.int32, "int"),
    ("s8->s16", INT8, INT16, torch.int8, torch.int16, "int"),
    ("s16->u8", INT16, UINT8, torch.int16, torch.uint8, "uint"),
)


def make_input(n: int, torch_dtype: torch.dtype, value_kind: str) -> torch.Tensor:
    if value_kind == "e4m3":
        x = torch.linspace(-448.0, 448.0, n, device="cpu", dtype=torch.float32)
    elif value_kind == "uint":
        x = torch.arange(n, device="cpu", dtype=torch.int32).clamp(0, 255)
    else:
        x = torch.arange(n, device="cpu", dtype=torch.float32) - (n // 2)
    return x.to(torch_dtype).to("npu")


def compare_result(name: str, y: torch.Tensor, ref: torch.Tensor) -> bool:
    if y.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        y = y.view(torch.uint8)
        ref = ref.view(torch.uint8)
    ok = torch.equal(y.cpu(), ref.cpu())
    if not ok:
        y_cpu, ref_cpu = y.cpu(), ref.cpu()
        mismatch = torch.nonzero(y_cpu != ref_cpu).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        print(f"{name}: first mismatch {first}: tl={y_cpu[first : first + 8]} ref={ref_cpu[first : first + 8]}")
    return ok


@pytest.mark.pto
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_pto_cast(case):
    name, in_tl, out_tl, in_torch, out_torch, value_kind = case
    kernel = cast_kernel(N, in_tl, out_tl)
    x = make_input(N, in_torch, value_kind)
    y = kernel(x)
    torch.npu.synchronize()
    assert compare_result(name, y, x.to(out_torch)), f"PTO VMI {name} failed"


if __name__ == "__main__":
    tilelang.testing.main()
