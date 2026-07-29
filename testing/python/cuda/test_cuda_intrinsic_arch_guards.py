"""CUDA intrinsic architecture guards exercised by device compilation.

This collects architecture-gate coverage from CUDA language and issue tests.
"""

from functools import partial

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.contrib import nvcc


_PASS_CONFIG = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED.value: True,
}


def _require_cuda_12_8():
    if nvcc.get_cuda_version() < (12, 8):
        pytest.skip("CUDA intrinsic architecture-guard tests require CUDA toolkit >= 12.8")


def _lower_for_arch(prim_func, arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with tvm.transform.PassContext(config=_PASS_CONFIG), target:
        return tilelang.lower(
            prim_func,
            target=target,
            enable_device_compile=True,
        )


def _make_tmem_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            tmem = T.alloc_tmem((128, 128), T.float32)
            T.deallocate_tmem(tmem)

    return main


def _make_tma_store_arrive_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            T.tma_store_arrive()

    return main


def _make_tma_store_wait_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            T.tma_store_wait(0)

    return main


def _make_tma_atomic_add_prim_func():
    @T.prim_func
    def main(out: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=128):
            out_shared = T.alloc_shared((16, 16), T.float32)
            T.fill(out_shared, 1)
            T.atomic_add(out, out_shared, use_tma=True)

    return main


def _make_tma_descriptor_prefetch_prim_func():
    @T.prim_func
    def main(descriptor: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1, threads=32):
            if T.shuffle_elect(0):
                T.call_intrin(
                    "handle",
                    tvm.tirx.op.Op.get("tl.prefetch_tma_descriptor"),
                    descriptor,
                )

    return main


def _make_tma_gather4_prim_func():
    @T.prim_func
    def main(
        src: T.Tensor((64, 64), T.float16),
        indices: T.Tensor((4,), T.int32),
        out: T.Tensor((4, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((4, 64), T.float16)
            mbar = T.alloc_barrier(1)

            if T.shuffle_elect(128):
                T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(64, "float16"))
                T.tma_gather4(
                    src,
                    smem,
                    0,
                    [indices[0], indices[1], indices[2], indices[3]],
                    barrier=mbar,
                )
                T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

            for i, j in T.Parallel(4, 64):
                out[i, j] = smem[i, j]

    return main


def _make_tma_scatter4_prim_func():
    @T.prim_func
    def main(
        indices: T.Tensor((4,), T.int32),
        out: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((4, 64), T.float16)
            T.fill(smem, 1)

            if T.shuffle_elect(128):
                T.tma_scatter4(
                    smem,
                    out,
                    0,
                    [indices[0], indices[1], indices[2], indices[3]],
                )
                T.tma_store_arrive()
            T.tma_store_wait(0, read=False)

    return main


def _make_cluster_prim_func():
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            T.cluster_arrive_relaxed()
            T.cluster_arrive()
            T.cluster_wait()
            T.cluster_sync()
            out[0] = T.block_rank_in_cluster()

    return main


def _make_clc_prim_func():
    @T.prim_func
    def main(out: T.Tensor((4,), T.uint32)):
        with T.Kernel(1, threads=1):
            result = T.alloc_shared((4,), T.uint32)
            mbarrier = T.alloc_shared((1,), T.uint64)
            T.clc_try_cancel(result, mbarrier)
            T.clc_try_cancel_multicast(result, mbarrier)
            out[0] = T.Cast("uint32", T.clc_is_canceled(result))
            out[1] = T.clc_get_first_ctaid_x(result)
            out[2] = T.clc_get_first_ctaid_y(result)
            out[3] = T.clc_get_first_ctaid_z(result)

    return main


def _make_fence_proxy_async_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=1):
            T.fence_proxy_async()

    return main


def _make_tcgen05_thread_fence_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=1):
            T.tcgen05_before_thread_sync()
            T.tcgen05_after_thread_sync()

    return main


def _make_warpgroup_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            T.warpgroup_arrive()
            T.warpgroup_commit_batch()
            T.warpgroup_wait(0)
            T.wait_wgmma(0)

    return main


def _make_shuffle_elect_prim_func():
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.shuffle_elect(0):
                out[0] = 1

    return main


def _make_register_reconfiguration_prim_func():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            T.inc_max_nreg(40)
            T.dec_max_nreg(40)

    return main


def _make_wgmma_prim_func():
    @T.prim_func
    def main(
        a: T.Tensor((64, 16), T.float16),
        b: T.Tensor((16, 64), T.float16),
        out: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 16), T.float16)
            b_shared = T.alloc_shared((16, 64), T.float16)
            accum = T.alloc_fragment((64, 64), T.float16)

            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.wgmma_gemm(a_shared, b_shared, accum, clear_accum=True)
            T.wait_wgmma(0)
            T.copy(accum, out)

    return main


def _make_tcgen05_mma_prim_func():
    @T.prim_func
    def main(
        a: T.Tensor((128, 128), T.bfloat16),
        b: T.Tensor((128, 128), T.bfloat16),
        out: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((128, 128), T.bfloat16)
            b_shared = T.alloc_shared((128, 128), T.bfloat16)
            accum_tmem = T.alloc_tmem((128, 128), T.float32)
            mbarrier = T.alloc_barrier(1)
            accum = T.alloc_fragment((128, 128), T.float32)
            out_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.tcgen05_gemm(
                a_shared,
                b_shared,
                accum_tmem,
                transpose_B=True,
                mbar=mbarrier,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbarrier, 0)
            T.copy(accum_tmem, accum)
            T.copy(accum, out_shared)
            T.copy(out_shared, out)

    return main


def _make_stochastic_rounding_prim_func(target_dtype):
    @T.prim_func
    def main(
        a: T.Tensor((128,), T.float32),
        out: T.Tensor((128,), target_dtype),
    ):
        with T.Kernel(1, threads=128):
            a_local = T.alloc_fragment((128,), T.float32)
            out_local = T.alloc_fragment((128,), target_dtype)
            rbits = T.alloc_fragment((1,), T.int32)
            T.copy(a, a_local)
            rbits[0] = T.int32(0x12345678)
            for i in T.Parallel(128):
                out_local[i] = T.cast(
                    a_local[i],
                    target_dtype,
                    round="rs",
                    rbits=rbits[0],
                )
            T.copy(out_local, out)

    return main


_UNSUPPORTED_CASES = [
    (
        _make_tmem_prim_func,
        "sm_90",
        ("tl::tmem_allocate requires sm_100a or a compatible architecture-specific target",),
    ),
    (
        _make_tmem_prim_func,
        "sm_90",
        ("tl::tmem_deallocate requires sm_100a or a compatible architecture-specific target",),
    ),
    (
        _make_tma_store_arrive_prim_func,
        "sm_80",
        ("tl::tma_store_arrive requires sm_90 or later",),
    ),
    (
        _make_tma_store_wait_prim_func,
        "sm_80",
        ("tl::tma_store_wait requires sm_90 or later",),
    ),
    (
        _make_tma_atomic_add_prim_func,
        "sm_80",
        ("tl::tma_store_add requires sm_90 or later",),
    ),
    (
        _make_tma_descriptor_prefetch_prim_func,
        "sm_80",
        ("tl::prefetch_tma_descriptor requires sm_90 or later",),
    ),
    (
        _make_tma_gather4_prim_func,
        "sm_90",
        ("tl::tma_load_gather4 requires sm_100 or later",),
    ),
    (
        _make_tma_scatter4_prim_func,
        "sm_100",
        ("tl::tma_store_scatter4 requires sm_100a",),
    ),
    (
        _make_cluster_prim_func,
        "sm_89",
        (
            "tl::cluster_arrive_relaxed requires sm_90 or later",
            "tl::cluster_arrive requires sm_90 or later",
            "tl::cluster_wait requires sm_90 or later",
            "tl::cluster_sync requires sm_90 or later",
            "tl::block_rank_in_cluster requires sm_90 or later",
        ),
    ),
    (
        _make_clc_prim_func,
        "sm_90",
        (
            "tl::clc_try_cancel requires sm_100a or a compatible architecture-specific target",
            "tl::clc_try_cancel_multicast requires sm_100a or a compatible architecture-specific target",
            "tl::clc_is_canceled requires sm_100a or a compatible architecture-specific target",
            "tl::clc_get_first_ctaid_x requires sm_100a or a compatible architecture-specific target",
            "tl::clc_get_first_ctaid_y requires sm_100a or a compatible architecture-specific target",
            "tl::clc_get_first_ctaid_z requires sm_100a or a compatible architecture-specific target",
        ),
    ),
    (
        _make_fence_proxy_async_prim_func,
        "sm_89",
        ("tl::fence_proxy_async requires sm_90 or later",),
    ),
    (
        _make_tcgen05_thread_fence_prim_func,
        "sm_100",
        (
            "tl::tcgen05_before_thread_sync requires sm_100a or a compatible architecture-specific target",
            "tl::tcgen05_after_thread_sync requires sm_100a or a compatible architecture-specific target",
        ),
    ),
    (
        _make_warpgroup_prim_func,
        "sm_89",
        (
            "tl::warpgroup_arrive requires sm_90",
            "tl::warpgroup_commit_batch requires sm_90",
            "tl::warpgroup_wait requires sm_90",
            "tl::wait_wgmma requires sm_90",
        ),
    ),
    (
        _make_shuffle_elect_prim_func,
        "sm_89",
        ("tl::tl_shuffle_elect requires sm_90 or later",),
    ),
    (
        _make_register_reconfiguration_prim_func,
        "sm_100",
        (
            "tl::warpgroup_reg_alloc requires a target with CTA register reconfiguration, such as sm_90a",
            "tl::warpgroup_reg_dealloc requires a target with CTA register reconfiguration, such as sm_90a",
        ),
    ),
    (
        _make_tcgen05_mma_prim_func,
        "sm_100",
        (
            "tl::tcgen05mma_ss requires sm_100a or a compatible architecture-specific target",
            "tl::tcgen05_ld_32dp32bNx requires sm_100a or a compatible architecture-specific target",
        ),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float16"),
        "sm_100",
        ("Stochastic rounding f32-to-FP16 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "bfloat16"),
        "sm_100",
        ("Stochastic rounding f32-to-BF16 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e4m3fn"),
        "sm_89",
        ("Stochastic rounding f32-to-FP8 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e4m3fn"),
        "sm_100",
        ("Stochastic rounding f32-to-FP8 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e5m2"),
        "sm_89",
        ("Stochastic rounding f32-to-FP8 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e5m2"),
        "sm_100",
        ("Stochastic rounding f32-to-FP8 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float4_e2m1fn"),
        "sm_89",
        ("Stochastic rounding f32-to-FP4 requires sm_100a",),
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float4_e2m1fn"),
        "sm_100",
        ("Stochastic rounding f32-to-FP4 requires sm_100a",),
    ),
]


_SUPPORTED_CASES = [
    (_make_tmem_prim_func, "sm_100a", "tl::tmem_deallocate"),
    (_make_tma_store_arrive_prim_func, "sm_90", "tl::tma_store_arrive"),
    (_make_tma_store_wait_prim_func, "sm_90", "tl::tma_store_wait<0"),
    (_make_tma_atomic_add_prim_func, "sm_90", "tl::tma_store_add"),
    (
        _make_tma_descriptor_prefetch_prim_func,
        "sm_90",
        "tl::prefetch_tma_descriptor",
    ),
    (_make_tma_gather4_prim_func, "sm_100", "tl::tma_load_gather4"),
    (_make_tma_scatter4_prim_func, "sm_100a", "tl::tma_store_scatter4"),
    (_make_cluster_prim_func, "sm_90", "tl::block_rank_in_cluster()"),
    (_make_clc_prim_func, "sm_100a", "tl::clc_get_first_ctaid_z("),
    (
        _make_fence_proxy_async_prim_func,
        "sm_90",
        "tl::fence_proxy_async()",
    ),
    (
        _make_tcgen05_thread_fence_prim_func,
        "sm_100a",
        "tl::tcgen05_after_thread_sync()",
    ),
    (_make_warpgroup_prim_func, "sm_90", "tl::wait_wgmma<0>()"),
    (_make_shuffle_elect_prim_func, "sm_90", "tl::tl_shuffle_elect<0>()"),
    (
        _make_register_reconfiguration_prim_func,
        "sm_100a",
        "tl::warpgroup_reg_dealloc<40>()",
    ),
    (_make_wgmma_prim_func, "sm_90", "tl::wgmma_ss"),
    (
        _make_tcgen05_mma_prim_func,
        "sm_100a",
        "tl::tcgen05mma_ss",
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float16"),
        "sm_100a",
        "__tl_cvt_f32x1_to_f16x1_rs_sat",
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "bfloat16"),
        "sm_100a",
        "__tl_cvt_f32x1_to_bf16x1_rs_sat",
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e4m3fn"),
        "sm_100a",
        "__tl_cvt_f32x1_to_e4m3x1_rs_sat",
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float8_e5m2"),
        "sm_100a",
        "__tl_cvt_f32x1_to_e5m2x1_rs_sat",
    ),
    (
        partial(_make_stochastic_rounding_prim_func, "float4_e2m1fn"),
        "sm_100a",
        "__tl_cvt_f32x1_to_e2m1x1_rs_sat",
    ),
]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "factory,arch,messages",
    _UNSUPPORTED_CASES,
    ids=[
        "tmem-allocate-sm90",
        "tmem-deallocate-sm90",
        "tma-store-arrive-sm80",
        "tma-store-wait-sm80",
        "tma-atomic-add-sm80",
        "tma-descriptor-prefetch-sm80",
        "tma-gather4-sm90",
        "tma-scatter4-sm100",
        "cluster-sm89",
        "clc-sm90",
        "fence-proxy-sm89",
        "tcgen05-fence-sm100",
        "warpgroup-sm89",
        "shuffle-elect-sm89",
        "register-reconfiguration-sm100",
        "tcgen05-mma-sm100",
        "stochastic-rounding-fp16-sm100",
        "stochastic-rounding-bf16-sm100",
        "stochastic-rounding-e4m3-sm89",
        "stochastic-rounding-e4m3-sm100",
        "stochastic-rounding-e5m2-sm89",
        "stochastic-rounding-e5m2-sm100",
        "stochastic-rounding-e2m1-sm89",
        "stochastic-rounding-e2m1-sm100",
    ],
)
def test_cuda_intrinsics_reject_unsupported_arch(factory, arch, messages):
    _require_cuda_12_8()

    with pytest.raises(RuntimeError) as exc_info:
        _lower_for_arch(factory(), arch)

    error = str(exc_info.value)
    for message in messages:
        assert message in error


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "factory,arch,expected_helper",
    _SUPPORTED_CASES,
    ids=[
        "tmem-sm100a",
        "tma-store-arrive-sm90",
        "tma-store-wait-sm90",
        "tma-atomic-add-sm90",
        "tma-descriptor-prefetch-sm90",
        "tma-gather4-sm100",
        "tma-scatter4-sm100a",
        "cluster-sm90",
        "clc-sm100a",
        "fence-proxy-sm90",
        "tcgen05-fence-sm100a",
        "warpgroup-sm90",
        "shuffle-elect-sm90",
        "register-reconfiguration-sm100a",
        "wgmma-sm90",
        "tcgen05-mma-sm100a",
        "stochastic-rounding-fp16-sm100a",
        "stochastic-rounding-bf16-sm100a",
        "stochastic-rounding-e4m3-sm100a",
        "stochastic-rounding-e5m2-sm100a",
        "stochastic-rounding-e2m1-sm100a",
    ],
)
def test_cuda_intrinsics_compile_for_supported_arch(factory, arch, expected_helper):
    _require_cuda_12_8()

    artifact = _lower_for_arch(factory(), arch)
    assert expected_helper in artifact.kernel_source
    assert "require_tcgen05" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
