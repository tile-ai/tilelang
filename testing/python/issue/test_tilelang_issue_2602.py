"""Architecture diagnostics for CUDA device helpers."""

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
        pytest.skip("CUDA architecture-gate tests require CUDA toolkit >= 12.8")


def _lower_for_arch(prim_func, arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with tvm.transform.PassContext(config=_PASS_CONFIG), target:
        return tilelang.lower(
            prim_func,
            target=target,
            enable_device_compile=True,
        )


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


_UNSUPPORTED_CASES = [
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
]


_SUPPORTED_CASES = [
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
]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "factory,arch,messages",
    _UNSUPPORTED_CASES,
    ids=[
        "cluster-sm89",
        "clc-sm90",
        "fence-proxy-sm89",
        "tcgen05-fence-sm100",
        "warpgroup-sm89",
        "shuffle-elect-sm89",
        "register-reconfiguration-sm100",
        "tcgen05-mma-sm100",
    ],
)
def test_device_helpers_reject_unsupported_arch(factory, arch, messages):
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
        "cluster-sm90",
        "clc-sm100a",
        "fence-proxy-sm90",
        "tcgen05-fence-sm100a",
        "warpgroup-sm90",
        "shuffle-elect-sm90",
        "register-reconfiguration-sm100a",
        "wgmma-sm90",
        "tcgen05-mma-sm100a",
    ],
)
def test_device_helpers_compile_for_supported_arch(factory, arch, expected_helper):
    _require_cuda_12_8()

    artifact = _lower_for_arch(factory(), arch)
    assert expected_helper in artifact.kernel_source
    assert "require_tcgen05" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
