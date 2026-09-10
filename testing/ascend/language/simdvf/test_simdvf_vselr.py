import warnings
from typing import Any

warnings.filterwarnings("ignore", message="Permission mismatch.*", module="torch_npu.utils._path_manager")
warnings.filterwarnings("ignore", message="Warning: The .* owner does not match the current owner\\.", module="torch_npu.utils.collect_env")

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def vselr_prefix_sum_kernel() -> Any:
    @T.prim_func
    def main(src: T.Tensor((64,), T.float32), dst: T.Tensor((64,), T.float32)) -> None:
        with T.Kernel(1):
            src_ub = T.alloc_shared((64,), T.float32)
            dst_ub = T.alloc_shared((64,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(0.0, "float32", full)
                lane = T.simd.vci(0, "int32", "INC_ORDER")
                x = T.simd.alloc_local((1,), "float32")
                x[0] = T.simd.vld(src_ub[0])
                shifted_1 = T.simd.vsel(
                    T.simd.vselr(x[0], T.simd.vadds(lane, -1, full)),
                    zero,
                    T.simd.vcmps(lane, 1, full, "ge"),
                )
                x[0] = T.simd.vadd(x[0], shifted_1, full)
                shifted_2 = T.simd.vsel(
                    T.simd.vselr(x[0], T.simd.vadds(lane, -2, full)),
                    zero,
                    T.simd.vcmps(lane, 2, full, "ge"),
                )
                x[0] = T.simd.vadd(x[0], shifted_2, full)
                T.simd.vsts(dst_ub[0], x[0], full)
            T.copy(dst_ub, dst)

    return main


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def bound_buffer_load_kernel() -> Any:
    @T.prim_func
    def main(src: T.Tensor((64,), T.float32), dst: T.Tensor((64,), T.float32)) -> None:
        with T.Kernel(1):
            src_ub = T.alloc_shared((64,), T.float32)
            dst_ub = T.alloc_shared((64,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                x = T.simd.alloc_local((1,), "float32")
                x[0] = T.simd.vld(src_ub[0])
                snapshot = x[0]
                x[0] = T.simd.vadds(x[0], 10.0, full)
                T.simd.vsts(dst_ub[0], snapshot, full)
            T.copy(dst_ub, dst)

    return main


def test_simdvf_vselr_reloads_mutated_local():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel = vselr_prefix_sum_kernel()

    source = kernel.get_kernel_source()
    assert source.count(" = x[0];") == 2

    kernel(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 3.0, 6.0, 10.0], dtype=torch.float32)
    assert torch.equal(dst[:4].cpu(), expected)


def pto_vselr_prefix_sum_kernel():
    """Hillis-Steele prefix steps via native T.vmi.vselr (64-lane f32)."""

    @T.prim_func
    def main(src: T.Tensor((64,), "float32"), dst: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            src_ub = T.alloc_shared((64,), "float32")
            dst_ub = T.alloc_shared((64,), "float32")
            T.copy(src, src_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(64, size=64)
                zero = T.vmi.vbrc(T.float32(0), size=64)
                lane = T.vmi.vci(T.int32(0), size=64, order="ASC")
                x0 = T.vmi.vload(src_ub[0], size=64)
                # Clamp gather indices before vselr so inactive low lanes never
                # feed negative OOB indices (results are still masked by vsel).
                idx_m1 = T.vmi.vmaxs(T.vmi.vadds(lane, T.int32(-1), full), T.int32(0), full)
                shifted_1 = T.vmi.vsel(
                    T.vmi.vcmps(lane, T.int32(1), full, "ge"),
                    T.vmi.vselr(x0, idx_m1),
                    zero,
                )
                x1 = T.vmi.vadd(x0, shifted_1, full)
                idx_m2 = T.vmi.vmaxs(T.vmi.vadds(lane, T.int32(-2), full), T.int32(0), full)
                shifted_2 = T.vmi.vsel(
                    T.vmi.vcmps(lane, T.int32(2), full, "ge"),
                    T.vmi.vselr(x1, idx_m2),
                    zero,
                )
                x2 = T.vmi.vadd(x1, shifted_2, full)
                T.vmi.vstore(x2, dst_ub[0], full)
            T.copy(dst_ub, dst)

    return main


@pytest.mark.pto
def test_pto_vselr_prefix_sum():
    kernel = tilelang.compile(pto_vselr_prefix_sum_kernel(), target="pto")
    source = kernel.get_kernel_source()
    assert "pto.vmi.vselr(" in source

    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 3.0, 6.0, 10.0], dtype=torch.float32)
    assert torch.equal(dst[:4].cpu(), expected)


def test_simdvf_bind_materializes_buffer_load_snapshot():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel = bound_buffer_load_kernel()

    source = kernel.get_kernel_source()
    assert source.count("snapshot = x[0];") == 1

    kernel(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), src.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
