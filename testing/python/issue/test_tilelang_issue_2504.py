"""Architecture diagnostics for TMEM and TMA DSL architecture gates."""

import re

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
        pytest.skip("TMEM/TMA architecture-gate tests require CUDA toolkit >= 12.8")


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


_UNSUPPORTED_CASES = [
    (
        _make_tmem_prim_func,
        "sm_90",
        "T.alloc_tmem requires sm_100a",
    ),
    (
        _make_tmem_prim_func,
        "sm_90",
        "T.deallocate_tmem requires sm_100a",
    ),
    (
        _make_tma_store_arrive_prim_func,
        "sm_80",
        "T.tma_store_arrive requires sm_90 or later",
    ),
    (
        _make_tma_store_wait_prim_func,
        "sm_80",
        "T.tma_store_wait requires sm_90 or later",
    ),
    (
        _make_tma_atomic_add_prim_func,
        "sm_80",
        "T.atomic_add(..., use_tma=True) requires sm_90 or later",
    ),
    (
        _make_tma_descriptor_prefetch_prim_func,
        "sm_80",
        "T.prefetch_tma_descriptor requires sm_90 or later",
    ),
    (
        _make_tma_gather4_prim_func,
        "sm_90",
        "T.tma_gather4 requires sm_100 or later",
    ),
    (
        _make_tma_scatter4_prim_func,
        "sm_100",
        "T.tma_scatter4 requires sm_100a",
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
]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "factory,arch,message",
    _UNSUPPORTED_CASES,
    ids=[
        "tmem-allocate",
        "tmem-deallocate",
        "store-arrive",
        "store-wait",
        "atomic-add",
        "descriptor-prefetch",
        "gather4",
        "scatter4",
    ],
)
def test_tmem_tma_builtins_reject_unsupported_arch(factory, arch, message):
    _require_cuda_12_8()

    with pytest.raises(RuntimeError, match=re.escape(message)):
        _lower_for_arch(factory(), arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "factory,arch,expected_helper",
    _SUPPORTED_CASES,
    ids=[
        "tmem",
        "store-arrive",
        "store-wait",
        "atomic-add",
        "descriptor-prefetch",
        "gather4",
        "scatter4",
    ],
)
def test_tmem_tma_builtins_compile_for_supported_arch(factory, arch, expected_helper):
    _require_cuda_12_8()

    artifact = _lower_for_arch(factory(), arch)

    assert expected_helper in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
