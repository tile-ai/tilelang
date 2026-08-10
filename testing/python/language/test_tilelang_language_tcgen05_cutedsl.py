# TMEM copies (tcgen05.ld/st) through the CuTeDSL backend, exercising the
# 32-datapath wrappers and the half-subpartition (16dp, PTX Layout F)
# wrappers of tilelang/contrib/cutedsl/gemm_tcgen05.py.
#
# The roundtrips prove st and ld agree bit-exactly on every layout; the TS
# GEMMs are the hardware ground truth: the MMA reads the A operand straight
# from TMEM in the layout PTX prescribes, so a store that placed values on
# the wrong datapaths shows up as a wrong product, not just a wrong readback.

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _is_cutedsl_available():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
        return True
    except (ImportError, AssertionError):
        return False


pytestmark = pytest.mark.skipif(not _is_cutedsl_available(), reason="CuTeDSL not installed")

_PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

# (name, shape, tmem forward fn, threads, wrapper the copies must use)
LAYOUT_CASES = [
    # (128,128):(1@0,1@1) -- the sub-partition-filling baseline.
    ("std_2d", (128, 128), lambda i, j: [i, j], 128, "32dp32bNx"),
    # ((16,4),128):((1@0,32@0),1@1) -- PTX Layout F, the 1SM M=64 fragment:
    # only the LOW 16 datapaths of each 32-datapath sub-partition are
    # occupied, so the atom is issued once per warp instead of duplicated
    # onto the high 16.
    ("layout_f_m64", (64, 128), lambda i, j: [i % 16 + 32 * (i // 16), j], 128, "16dp64bNx"),
    ("layout_f_m64_2wg", (64, 256), lambda i, j: [i % 16 + 32 * (i // 16), j], 256, "16dp64bNx"),
]


def _make_roundtrip_kernel(shape, forward, threads):
    @T.prim_func
    def main(
        A: T.Tensor(shape, T.float32),
        D: T.Tensor(shape, T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared(shape, T.float32)
            A_frag = T.alloc_fragment(shape, T.float32)
            tmem = T.alloc_tmem(shape, T.float32)
            B_frag = T.alloc_fragment(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            T.copy(A_frag, tmem)
            T.copy(tmem, B_frag)
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("name", "shape", "forward", "threads", "wrapper"),
    LAYOUT_CASES,
    ids=[case[0] for case in LAYOUT_CASES],
)
def test_tmem_copy_roundtrip_cutedsl(name, shape, forward, threads, wrapper):
    import torch

    kernel = tilelang.compile(
        _make_roundtrip_kernel(shape, forward, threads),
        target="cutedsl",
        pass_configs=_PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    assert f"tl.tcgen05_st_{wrapper}" in source, f"{name} must store through {wrapper}"
    assert f"tl.tcgen05_ld_{wrapper}" in source, f"{name} must load through {wrapper}"

    a = torch.randn(*shape, device="cuda", dtype=torch.float32)
    d = torch.zeros(*shape, device="cuda", dtype=torch.float32)
    kernel(a, d)
    torch.testing.assert_close(d, a, rtol=0, atol=0)


def _make_ts_gemm_kernel(M, N, K):
    """A TS GEMM whose A operand reaches TMEM through registers.

    At M=64 the A fragment is PTX Layout F, so the register->TMEM store and
    the MMA's read of it have to agree on a half-subpartition layout.
    """
    from tilelang.cuda.intrinsics import make_mma_swizzle_layout

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((M, K), T.bfloat16)
            b_shared = T.alloc_shared((N, K), T.bfloat16)
            T.annotate_layout({a_shared: make_mma_swizzle_layout(a_shared)})
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            with T.sblock():
                T.reads()
                T.writes()
                a_tmem = T.alloc_tmem((M, K), T.bfloat16)
                c_tmem = T.alloc_tmem((M, N), T.float32)
                a_frag = T.alloc_fragment((M, K), T.bfloat16)
                c_frag = T.alloc_fragment((M, N), T.float32)
                done = T.alloc_barrier(1)
                T.copy(a_shared, a_frag)
                T.copy(a_frag, a_tmem)
                if T.get_warp_idx_sync() == 0:
                    T.tcgen05_gemm(a_tmem, b_shared, c_tmem, transpose_B=True, clear_accum=True, mbar=done)
                T.mbarrier_wait_parity(done, 0)
                T.copy(c_tmem, c_frag)
                T.copy(c_frag, C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("M", [64, 128], ids=["m64", "m128"])
def test_tcgen05_ts_gemm_tmem_a_correctness_cutedsl(M):
    import torch

    N, K = 128, 128
    kernel = tilelang.compile(
        _make_ts_gemm_kernel(M, N, K),
        target="cutedsl",
        pass_configs=_PASS_CONFIGS,
    )
    # M=64 must go through the half-subpartition wrappers, M=128 must not.
    source = kernel.get_kernel_source()
    want = "tl.tcgen05_st_16dp" if M == 64 else "tl.tcgen05_st_32dp"
    assert want in source, f"M={M} expected {want} in the emitted source"

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
