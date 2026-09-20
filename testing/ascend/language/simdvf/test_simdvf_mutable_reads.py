"""Mutable SIMD reads and immutable snapshots observe the correct version."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@tilelang.jit(
    target="ascend",
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def vselr_prefix_sum_kernel():
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
    target="ascend",
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def bound_buffer_load_kernel():
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

    kernel(src, dst)
    torch.npu.synchronize()
    expected = src.cpu().clone()
    for offset in (1, 2, 3):
        expected[offset:] += src.cpu()[:-offset]
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_simdvf_bind_materializes_buffer_load_snapshot():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel = bound_buffer_load_kernel()

    kernel(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), src.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
