"""Coverage for atom-aligned operand regions in dense SM100 TCGEN5MMA.

The cases here intentionally keep the backing buffers larger than an
individual MMA.  That distinguishes region lowering from the already-covered
whole-buffer path and mirrors the K-sliced SS and four-way TS schedules that a
tiling pass needs to generate.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _make_sliced_ss_kernel(k_extent, k_tile):
    """Accumulate a whole BF16 GEMM from K regions of one SMEM allocation."""

    @T.prim_func
    def main(
        A: T.Tensor((128, k_extent), T.bfloat16),
        B: T.Tensor((128, k_extent), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, k_extent), T.bfloat16)
            B_shared = T.alloc_shared((128, k_extent), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for ko in T.serial(0, k_extent, k_tile):
                T.tcgen05_gemm(
                    A_shared[:, ko : ko + k_tile],
                    B_shared[:, ko : ko + k_tile],
                    C_tmem,
                    transpose_B=True,
                    mbar=done,
                    clear_accum=ko == 0,
                )
                T.mbarrier_wait_parity(done, (ko // k_tile) % 2)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_offset_swizzle_atom_ss_kernel():
    """A K32 view starting two swizzle atoms into a wider K-major buffer."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.tcgen05_gemm(
                A_shared[:, 64:96],
                B_shared[:, 64:96],
                C_tmem,
                transpose_B=True,
                mbar=done,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_issue_only_sliced_ts_kernel():
    """Compute A @ B using sliced TMEM A and a staged, K-sliced SMEM B."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 64), T.bfloat16),
        D: T.Tensor((128, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            A_local = T.alloc_fragment((128, 128), T.bfloat16)
            A_tmem = T.alloc_tmem((128, 128), T.bfloat16)
            # Slot zero is deliberately unused: B pins a parent-buffer mode to
            # extent one and K-slices one of the two remaining Region modes.
            B_ring = T.alloc_shared((2, 128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            T.copy(A_local, A_tmem)
            T.copy(B, B_ring[1, :, :])

            # All four issues stay ordered in the TC issue stream; a manual
            # commit after the loop publishes one completion event for the
            # whole sequence.  The commit must come from the issue warp (the
            # lowering guards the MMAs to warp zero), or idle warps would
            # arrive the barrier before the MMAs complete.
            for k in T.serial(4):
                T.tcgen05_gemm(
                    A_tmem[:, k * 32 : (k + 1) * 32],
                    B_ring[1, k * 32 : (k + 1) * 32, :],
                    C_tmem,
                    mbar=None,
                    clear_accum=k == 0,
                )
            if T.get_thread_binding() < 32:
                T.tcgen05_mma_arrive(done)
            T.mbarrier_wait_parity(done, 0)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_2cta_sliced_ts_kernel():
    """Run four per-CTA K32 views of one cooperative TS accumulation chain.

    Each CTA holds its rank's M shard of A in TMEM and its rank's N shard of
    B in shared memory; the four K slices accumulate into one cooperative C.
    """

    @T.prim_func
    def main(
        A: T.Tensor((256, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((256, 128), T.bfloat16),
    ):
        with T.ClusterKernel(2, threads=128, cluster_dims=2):
            rank = T.block_rank_in_cluster()
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            A_local = T.alloc_fragment((128, 128), T.bfloat16)
            A_tmem = T.alloc_tmem((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            done = T.alloc_cluster_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A[rank * 128 : (rank + 1) * 128, :], A_shared)
            T.copy(A_shared, A_local)
            T.copy(A_local, A_tmem)
            T.copy(B[:, rank * 64 : (rank + 1) * 64], B_shared)
            # Explicit tcgen05_gemm does not auto-emit the cooperative input
            # handoff, so publish both ranks' TMEM/SMEM shards before the
            # leader issues.
            T.tcgen05_before_thread_sync()
            T.cluster_sync()
            T.tcgen05_after_thread_sync()
            for k in T.serial(4):
                T.tcgen05_gemm(
                    A_tmem[:, k * 32 : (k + 1) * 32],
                    B_shared[k * 32 : (k + 1) * 32, :],
                    C_tmem,
                    mbar=None,
                    clear_accum=k == 0,
                    use_2cta=True,
                )
            # Only the leader CTA's issue warp commits; the multicast arrives
            # both ranks' barriers.
            if rank == 0 and T.get_thread_binding() < 32:
                T.tcgen05_mma_arrive(done, arrive_2cta=True)
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[rank * 128 : (rank + 1) * 128, :])

    return main


def _make_transposed_sliced_ss_kernel():
    """Slice K on the penultimate physical axis of both SS operands."""

    @T.prim_func
    def main(
        A: T.Tensor((32, 128), T.bfloat16),
        B: T.Tensor((32, 64), T.bfloat16),
        D: T.Tensor((128, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 128), T.bfloat16)
            B_shared = T.alloc_shared((32, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.tcgen05_gemm(
                A_shared[16:32, :],
                B_shared[16:32, :],
                C_tmem,
                transpose_A=True,
                mbar=done,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_nonzero_mn_origin_ts_kernel():
    """Use the second M atom and second N atom of larger TMEM/SMEM owners."""

    @T.prim_func
    def main(
        A: T.Tensor((256, 16), T.bfloat16),
        B: T.Tensor((16, 128), T.bfloat16),
        D: T.Tensor((128, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((256, 16), T.bfloat16)
            A_local = T.alloc_fragment((256, 16), T.bfloat16)
            A_tmem = T.alloc_tmem((256, 16), T.bfloat16)
            B_shared = T.alloc_shared((16, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            T.copy(A_local, A_tmem)
            T.copy(B, B_shared)
            T.tcgen05_gemm(
                A_tmem[128:256, :],
                B_shared[:, 64:128],
                C_tmem,
                mbar=done,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_nonzero_c_origin_ts_kernel():
    """Compile a C view that starts at the second outer M and N atoms."""

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem((128, 16), T.bfloat16)
            B_shared = T.alloc_shared((16, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((256, 128), T.float32)
            T.tcgen05_gemm(
                A_tmem,
                B_shared,
                C_tmem[128:256, 64:128],
                mbar=None,
                clear_accum=True,
            )

    return main


def _make_pinned_leading_modes_ss_kernel():
    """Pin both shared operands' leading batch modes to one element each."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 16), T.bfloat16),
        B: T.Tensor((16, 64), T.bfloat16),
        D: T.Tensor((128, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((2, 2, 128, 16), T.bfloat16)
            B_shared = T.alloc_shared((2, 2, 16, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 64), T.float32)
            C_shared = T.alloc_shared((128, 64), T.bfloat16)

            T.copy(A, A_shared[1, 0, :, :])
            T.copy(B, B_shared[0, 1, :, :])
            T.tcgen05_gemm(
                A_shared[1, 0, :, :],
                B_shared[0, 1, :, :],
                C_tmem,
                mbar=done,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_misaligned_ss_slice_kernel(operand):
    """Make only the selected operand begin halfway through a BF16 K16 atom."""

    a_begin = 8 if operand == "A" else 0
    b_begin = 8 if operand == "B" else 0

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 32), T.bfloat16)
            B_shared = T.alloc_shared((128, 32), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            T.tcgen05_gemm(
                A_shared[:, a_begin : a_begin + 16],
                B_shared[:, b_begin : b_begin + 16],
                C_tmem,
                transpose_B=True,
                mbar=None,
                clear_accum=True,
            )

    return main


def _make_region_validation_kernel(case):
    """Build one otherwise-valid SS atom with one misaligned region axis."""

    transpose_a = case == "a_k_penultimate"
    transpose_b = case == "b_k_last"
    a_m = 64 if case == "a_m" else 0
    a_k = 8 if case in {"a_k_last", "a_k_penultimate"} else 0
    b_n = 32 if case == "b_n" else 0
    b_k = 8 if case in {"b_k_last", "b_k_penultimate"} else 0
    c_m = 64 if case == "c_m" else 0
    c_n = 32 if case == "c_n" else 0

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 256) if transpose_a else (256, 32), T.bfloat16)
            B_shared = T.alloc_shared((128, 32) if transpose_b else (32, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((256, 128), T.float32)
            T.tcgen05_gemm(
                (A_shared[a_k : a_k + 16, a_m : a_m + 128] if transpose_a else A_shared[a_m : a_m + 128, a_k : a_k + 16]),
                (B_shared[b_n : b_n + 64, b_k : b_k + 16] if transpose_b else B_shared[b_k : b_k + 16, b_n : b_n + 64]),
                C_tmem[c_m : c_m + 128, c_n : c_n + 64],
                transpose_A=transpose_a,
                transpose_B=transpose_b,
                mbar=None,
                clear_accum=True,
            )

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("k_extent,k_tile", [(64, 16), (128, 32)])
def test_tcgen05_ss_atom_aligned_k_slices(k_extent, k_tile):
    kernel = tilelang.compile(
        _make_sliced_ss_kernel(k_extent, k_tile),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    assert "tcgen05mma_ss" in source
    assert "increase_descriptor_offset" in source

    a = torch.randn((128, k_extent), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, k_extent), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a.cpu().float() @ b.cpu().float().T).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ss_slice_at_offset_swizzle_atom():
    kernel = tilelang.compile(
        _make_offset_swizzle_atom_ss_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a[:, 64:96].cpu().float() @ b[:, 64:96].cpu().float().T).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ts_issue_only_sliced_a_and_b():
    kernel = tilelang.compile(
        _make_issue_only_sliced_ts_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    # The four K32 issues stay in the serial loop, so one instruction carries
    # a loop-dependent TMEM A origin and descriptor offset.
    assert source.count("tcgen05mma_ts") == 1
    assert source.count("tcgen05_mma_arrive") == 1
    assert "(k * 16)" in source
    assert "increase_descriptor_offset" in source
    assert "(k * 4096)" in source

    a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a.cpu().float() @ b.cpu().float()).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_2cta_ts_slices_have_distinct_a_and_b_origins():
    kernel = tilelang.compile(
        _make_2cta_sliced_ts_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    # The four K32 issues stay in the serial loop, so one cooperative
    # instruction carries the loop-dependent TMEM A origin (16 raw columns
    # per slice) and B descriptor byte offset (4096 per slice).
    assert source.count("tcgen05mma_ts<tl::DataType::kBFloat16, true>") == 1
    assert "(k * 16)" in source
    assert "(k * 4096)" in source
    assert source.count("tcgen05_mma_arrive<true>") == 1

    a = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ss_k_slice_on_penultimate_axes():
    kernel = tilelang.compile(
        _make_transposed_sliced_ss_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    a = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a[16:32].cpu().float().T @ b[16:32].cpu().float()).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ts_nonzero_a_m_and_b_n_origins():
    kernel = tilelang.compile(
        _make_nonzero_mn_origin_ts_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    a = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a[128:256].cpu().float() @ b[:, 64:128].cpu().float()).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ts_nonzero_c_m_n_origins_compile():
    source = tilelang.compile(
        _make_nonzero_c_origin_ts_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    ).get_kernel_source()
    ts_line = next(line for line in source.splitlines() if "tcgen05mma_ts" in line)
    assert "C_tmem" in ts_line and "+ 192" in ts_line


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ss_pinned_leading_modes_end_to_end():
    kernel = tilelang.compile(
        _make_pinned_leading_modes_ss_kernel(),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    assert source.count("tcgen05mma_ss") == 1

    a = torch.randn((128, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    expected = (a.cpu().float() @ b.cpu().float()).to(torch.bfloat16)
    tilelang.testing.torch_assert_close(d.cpu(), expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("operand", ["A", "B"])
def test_tcgen05_ss_rejects_k_origin_inside_atom(operand):
    with pytest.raises(AssertionError, match=rf"TCGEN5MMA {operand} K origin 8 must be aligned.*atom 16"):
        tilelang.compile(
            _make_misaligned_ss_slice_kernel(operand),
            target="cuda",
            pass_configs=PASS_CONFIGS,
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("a_k_last", r"TCGEN5MMA A K origin 8 must be aligned.*atom 16"),
        ("a_k_penultimate", r"TCGEN5MMA A K origin 8 must be aligned.*atom 16"),
        ("b_k_last", r"TCGEN5MMA B K origin 8 must be aligned.*atom 16"),
        ("b_k_penultimate", r"TCGEN5MMA B K origin 8 must be aligned.*atom 16"),
        ("a_m", r"TCGEN5MMA A M origin 64 must be aligned.*atom 128"),
        ("b_n", r"TCGEN5MMA B N origin 32 must be aligned.*atom 64"),
        ("c_m", r"TCGEN5MMA C M origin 64 must be aligned.*atom 128"),
        ("c_n", r"TCGEN5MMA C N origin 32 must be aligned.*atom 64"),
    ],
)
def test_tcgen05_rejects_regions_that_split_instruction_atoms(case, message):
    with pytest.raises(AssertionError, match=message):
        tilelang.compile(
            _make_region_validation_kernel(case),
            target="cuda",
            pass_configs=PASS_CONFIGS,
        )


if __name__ == "__main__":
    tilelang.testing.main()
