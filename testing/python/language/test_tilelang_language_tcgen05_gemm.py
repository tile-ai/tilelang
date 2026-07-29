import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _marker_line(marker: str) -> int:
    with open(__file__) as f:
        for i, line in enumerate(f, 1):
            if marker in line:
                return i
    raise ValueError(f"marker not found: {marker}")


def _assert_error_at(excinfo, marker: str):
    message = str(excinfo.value)
    expected = f"--> {__file__}:{_marker_line(marker)}:"
    assert expected in message, f"expected {expected!r} in error message:\n{message}"
    assert marker in message, f"source snippet was not rendered:\n{message}"


def _make_sync_tcgen05_kernel(gemm_op):
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
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            gemm_op(A_shared, B_shared, C_tmem, mbar)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    return main


def _make_async_tcgen05_kernel(gemm_op):
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
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            gemm_op(A_shared, B_shared, C_tmem, mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    return main


def _tcgen05_issue_block(source):
    """Extract the complete guarded block that initializes and issues TCGEN05."""

    lines = source.splitlines()
    anchor = next(i for i, line in enumerate(lines) if "initialize_tcgen05_descriptor" in line)
    start = next(i for i in range(anchor, -1, -1) if "if (" in lines[i] and "{" in lines[i])

    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(line.strip() for line in lines[start : end + 1])
    raise AssertionError("unterminated TCGEN05 issue block")


def _make_issue_only_tcgen05_kernel():
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
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            # The MMAs remain ordered in the TC issue stream without
            # publishing independent completion events; one manual commit
            # from the issue warp posts the event consumed before reading C.
            for k in T.serial(2):
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    mbar=None,
                    clear_accum=k == 0,
                )
            if T.get_thread_binding() < 32:
                T.tcgen05_mma_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


def _make_sync_sliced_ts_tmem_kernel():
    """The high-level GEMM keeps nonzero K origins until SM100 selection."""

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            P_tmem = T.alloc_tmem((128, 128), T.bfloat16)
            V_shared = T.alloc_shared((128, 64), T.bfloat16)
            O_tmem = T.alloc_tmem((128, 64), T.float32)
            done_0 = T.alloc_barrier(1)
            done_1 = T.alloc_barrier(1)
            done_2 = T.alloc_barrier(1)
            done_3 = T.alloc_barrier(1)

            T.gemm(P_tmem[:, 0:32], V_shared[0:32, :], O_tmem, mbar=done_0, clear_accum=True)
            T.gemm(P_tmem[:, 32:64], V_shared[32:64, :], O_tmem, mbar=done_1)
            T.gemm(P_tmem[:, 64:96], V_shared[64:96, :], O_tmem, mbar=done_2)
            T.gemm(P_tmem[:, 96:128], V_shared[96:128, :], O_tmem, mbar=done_3)

    return main


def _make_explicit_2cta_gemm_kernel():
    """Explicit two-CTA GEMM: user-scheduled operand publish and completion.

    The cluster-wide handoffs are manual: ``cluster_sync`` publishes both
    ranks' shared shards before the leader issues, and the completion commit
    multicasts to both ranks' barriers, so each rank waits locally.
    """

    @T.prim_func
    def main(
        A: T.Tensor((256, 64), T.bfloat16),
        B: T.Tensor((256, 64), T.bfloat16),
        D: T.Tensor((256, 256), T.bfloat16),
    ):
        with T.ClusterKernel(2, threads=128, cluster_dims=2):
            rank = T.block_rank_in_cluster()
            A_shared = T.alloc_shared((128, 64), T.bfloat16)
            B_shared = T.alloc_shared((128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 256), T.float32)
            done = T.alloc_cluster_barrier(1)
            C_local = T.alloc_fragment((128, 256), T.float32)
            C_shared = T.alloc_shared((128, 256), T.bfloat16)

            T.copy(A[rank * 128 : (rank + 1) * 128, :], A_shared)
            T.copy(B[rank * 128 : (rank + 1) * 128, :], B_shared)
            T.cluster_sync()
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=done,
                use_2cta=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[rank * 128 : (rank + 1) * 128, :])

    return main


def _make_explicit_2cta_ts_handoff_kernel():
    """Thread-produced TMEM A consumed by a cooperative TS GEMM.

    A TMEM input needs the canonical TCGEN visibility handoff around the
    cluster rendezvous before the leader issues.
    """

    @T.prim_func
    def main(
        P: T.Tensor((256, 32), T.bfloat16),
        V: T.Tensor((32, 64), T.bfloat16),
        D: T.Tensor((256, 64), T.bfloat16),
    ):
        with T.ClusterKernel(2, threads=128, cluster_dims=2):
            rank = T.block_rank_in_cluster()
            P_shared = T.alloc_shared((128, 32), T.bfloat16)
            P_local = T.alloc_fragment((128, 32), T.bfloat16)
            P_tmem = T.alloc_tmem((128, 32), T.bfloat16)
            V_shared = T.alloc_shared((32, 32), T.bfloat16)
            O_tmem = T.alloc_tmem((128, 64), T.float32)
            done = T.alloc_cluster_barrier(1)
            O_local = T.alloc_fragment((128, 64), T.float32)
            O_shared = T.alloc_shared((128, 64), T.bfloat16)

            T.copy(P[rank * 128 : (rank + 1) * 128, :], P_shared)
            T.copy(P_shared, P_local)
            T.copy(P_local, P_tmem)
            T.copy(V[:, rank * 32 : (rank + 1) * 32], V_shared)
            T.tcgen05_before_thread_sync()
            T.cluster_sync()
            T.tcgen05_after_thread_sync()
            T.tcgen05_gemm(
                P_tmem,
                V_shared,
                O_tmem,
                clear_accum=True,
                mbar=done,
                use_2cta=True,
            )
            T.mbarrier_wait_parity(done, 0)
            T.copy(O_tmem, O_local)
            T.copy(O_local, O_shared)
            T.copy(O_shared, D[rank * 128 : (rank + 1) * 128, :])

    return main


def _make_2cta_without_cluster_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.bfloat16)
            B_shared = T.alloc_shared((128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 256), T.float32)
            done = T.alloc_barrier(1)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=done,
                use_2cta=True,
            )

    return main


def _make_2cta_invalid_cluster_kernel():
    @T.prim_func
    def main():
        with T.ClusterKernel(4, threads=128, cluster_dims=4):
            A_shared = T.alloc_shared((128, 64), T.bfloat16)
            B_shared = T.alloc_shared((128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 256), T.float32)
            done = T.alloc_cluster_barrier(1)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=done,
                use_2cta=True,
            )

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_matches_sync_gemm_issue_codegen():
    def sync_api(A, B, C, mbar):
        return T.gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True)

    def async_api(A, B, C, mbar):
        return T.tcgen05_gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True)

    sync_kernel = tilelang.compile(_make_sync_tcgen05_kernel(sync_api), target="cuda")
    async_kernel = tilelang.compile(_make_async_tcgen05_kernel(async_api), target="cuda")
    sync_source = sync_kernel.get_kernel_source()
    async_source = async_kernel.get_kernel_source()

    assert _tcgen05_issue_block(sync_source) == _tcgen05_issue_block(async_source)
    for source in (sync_source, async_source):
        assert source.count(".wait(0)") == 1
        assert source.index("tcgen05_mma_arrive") < source.index(".wait(0)")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_can_omit_intermediate_completion_event():
    kernel = tilelang.compile(_make_issue_only_tcgen05_kernel(), target="cuda")
    source = kernel.get_kernel_source()
    assert source.count("tcgen05_mma_arrive") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_sync_gemm_preserves_sliced_ts_operands_for_sm100_selection():
    source = tilelang.compile(_make_sync_sliced_ts_tmem_kernel(), target="cuda").get_kernel_source()
    assert source.count("tl::tcgen05mma_ts") == 4
    assert "increase_descriptor_offset" in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_explicit_gemm_2cta_correctness():
    import torch

    kernel = tilelang.compile(_make_explicit_2cta_gemm_kernel(), target="cuda")
    source = kernel.get_kernel_source()
    assert "tcgen05mma_ss<tl::DataType::kBFloat16, true>" in source
    mma_pos = source.index("tcgen05mma_ss<tl::DataType::kBFloat16, true>")
    acquire_sync_pos = source.rfind("tl::cluster_sync()", 0, mma_pos)
    assert 0 <= acquire_sync_pos < mma_pos

    wait_pos = source.index(".wait(", mma_pos)
    load_pos = source.index("tl::tcgen05_ld_", wait_pos)
    assert 0 <= mma_pos < wait_pos < load_pos

    a = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_explicit_gemm_2cta_ts_thread_handoff_correctness():
    import torch

    kernel = tilelang.compile(_make_explicit_2cta_ts_handoff_kernel(), target="cuda")
    source = kernel.get_kernel_source()
    mma_pos = source.index("tcgen05mma_ts<tl::DataType::kBFloat16, true>")
    store_pos = source.rfind("tl::tcgen05_st_", 0, mma_pos)
    sync_pos = source.rfind("tl::cluster_sync()", 0, mma_pos)
    before_pos = source.rfind("tl::tcgen05_before_thread_sync()", 0, sync_pos)
    after_pos = source.index("tl::tcgen05_after_thread_sync()", sync_pos)
    assert 0 <= store_pos < before_pos < sync_pos < after_pos < mma_pos

    p = torch.randn(256, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(256, 64, device="cuda", dtype=torch.bfloat16)
    kernel(p, v, d)
    ref = (p.float() @ v.float()).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    "kernel_factory",
    [_make_2cta_without_cluster_kernel, _make_2cta_invalid_cluster_kernel],
)
def test_tcgen05_gemm_2cta_requires_two_cta_cluster(kernel_factory):
    with pytest.raises(tvm.error.InternalError, match=r"requires cluster_dims"):
        tilelang.compile(kernel_factory(), target="cuda")


@tilelang.testing.requires_cuda
def test_tcgen05_gemm_2cta_rejects_non_blackwell_target():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with pytest.raises(tvm.error.InternalError, match=r"requires Blackwell TCGEN5MMA"):
        tilelang.compile(_make_explicit_2cta_gemm_kernel(), target=target)


def test_sync_gemm_has_no_2cta_parameter():
    # The synchronous convenience wrapper deliberately does not model the
    # cluster-wide handoffs; 2CTA lives only on T.tcgen05_gemm.
    with pytest.raises(TypeError, match=r"use_2cta"):
        T.gemm(None, None, None, use_2cta=True)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_rejects_non_tcgen05_lowering():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)  # tcgen05_dense_error_span
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_local = T.alloc_fragment((128, 128), T.float32)
            mbar = T.alloc_barrier(1)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            T.tcgen05_gemm(A_shared, B_shared, C_local, transpose_B=True, mbar=mbar, clear_accum=True)
            T.copy(C_local, D[0:128, 0:128])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.tcgen05_gemm\(\) requires Blackwell TCGEN5MMA lowering",
    ) as excinfo:
        tilelang.compile(main, target="cuda")
    message = str(excinfo.value)
    assert "target=" in message
    assert "A(scope=shared.dyn, dtype=bfloat16)" in message
    assert "B(scope=shared.dyn, dtype=bfloat16)" in message
    assert "C(scope=local.fragment, dtype=float32)" in message
    assert "M=128, N=128, K=128" in message
    _assert_error_at(excinfo, "tcgen05_dense_error_span")


def _make_sliced_tcgen05_kernel(M, N, K, num_k_tiles):
    k_tile = K // num_k_tiles

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        D: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.bfloat16)
            B_shared = T.alloc_shared((N, K), T.bfloat16)
            C_tmem = T.alloc_tmem((M, N), T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((M, N), T.float32)
            C_shared = T.alloc_shared((M, N), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            for j in T.serial(num_k_tiles):
                T.tcgen05_gemm(
                    A_shared[:, j * k_tile : (j + 1) * k_tile],
                    B_shared[:, j * k_tile : (j + 1) * k_tile],
                    C_tmem,
                    transpose_B=True,
                    mbar=mbar,
                    clear_accum=(j == 0),
                )
                T.mbarrier_wait_parity(mbar, j % 2)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_sliced_operand_emits_offset():
    # A sliced (non-zero-origin) UMMA operand must build the descriptor from the
    # buffer base and advance it with increase_descriptor_offset (warp-uniform),
    # mirroring the WGMMA path.
    kernel = tilelang.compile(_make_sliced_tcgen05_kernel(128, 128, 128, 2), target="cuda")
    src = kernel.get_kernel_source()
    assert "increase_descriptor_offset" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    "M,N,K,num_k_tiles",
    [
        (128, 128, 128, 2),
        (128, 128, 256, 4),
        (128, 256, 128, 2),
    ],
)
def test_tcgen05_gemm_sliced_operand_correctness(M, N, K, num_k_tiles):
    import torch

    kernel = tilelang.compile(_make_sliced_tcgen05_kernel(M, N, K, num_k_tiles), target="cuda")
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().t()).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


def _make_batched_tcgen05_kernel(batch_size):
    """One independent GEMM per batch entry, all operands 3-D.

    Every buffer carries the batch mode: the leading mode repeats the TMEM
    accumulator fragment along columns, so per-entry results occupy disjoint
    column windows of one allocation, while the SS operands slice their own
    batch entry out of 3-D shared buffers.  The GEMM loop and the tcgen05.ld
    epilogue both carry a dynamic batch origin: the MMA and copy tiles are
    static relative to the Region origin, which only moves the descriptor and
    TMEM base addresses.
    """

    @T.prim_func
    def main(
        A: T.Tensor((batch_size, 128, 64), T.bfloat16),
        B: T.Tensor((batch_size, 128, 64), T.bfloat16),
        D: T.Tensor((batch_size, 128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((batch_size, 128, 64), T.bfloat16)
            B_shared = T.alloc_shared((batch_size, 128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((batch_size, 128, 128), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for i in T.serial(batch_size):
                T.tcgen05_gemm(
                    A_shared[i, :, :],
                    B_shared[i, :, :],
                    C_tmem[i, :, :],
                    transpose_B=True,
                    mbar=done,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(done, i % 2)
            for i in T.serial(batch_size):
                T.copy(C_tmem[i, :, :], C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, D[i, :, :])

    return main


def _make_versioned_blockscaled_tcgen05_kernel():
    """Block-scaled GEMM with accumulator versions and colocated scales."""

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            B_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            C_tmem = T.alloc_tmem((4, 128, 128), T.float32)
            done = T.alloc_barrier(1)

            T.tcgen05_gemm_blockscaled(
                A_shared,
                B_shared,
                C_tmem[2, :, :],
                C_tmem[3, :, 4:8],
                C_tmem[3, :, 12:16],
                transpose_B=True,
                mbar=done,
                clear_accum=True,
                k_start=0,
                sf_a_granularity_k=32,
                sf_b_granularity_k=32,
            )

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_blockscaled_versioned_tmem_regions():
    kernel = tilelang.compile(
        _make_versioned_blockscaled_tcgen05_kernel(),
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    mma_line = next(line for line in kernel.get_kernel_source().splitlines() if "tcgen05mma_blockscaled_ss" in line)

    # Version 2 starts at column 256. Scale regions live in version 3 and
    # start at columns 384 + 4 and 384 + 12 of the same TMEM allocation.
    assert "+ 256" in mma_line
    assert "+ 388" in mma_line
    assert "+ 396" in mma_line


def _make_batched_whole_buffer_ld_kernel(batch_size):
    """One tcgen05.ld epilogue over a whole 3-D TMEM buffer.

    The leading batch mode repeats the accumulator along TMEM columns, so the
    full-buffer copy is one (128, batch_size*128)-column tile; the register
    fragment repeats along the value vector (each thread holds
    batch_size*128 accumulators, batch slowest).
    """

    @T.prim_func
    def main(
        A: T.Tensor((batch_size, 128, 64), T.bfloat16),
        B: T.Tensor((batch_size, 128, 64), T.bfloat16),
        D: T.Tensor((batch_size, 128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((batch_size, 128, 64), T.bfloat16)
            B_shared = T.alloc_shared((batch_size, 128, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((batch_size, 128, 128), T.float32)
            done = T.alloc_barrier(1)
            C_local = T.alloc_fragment((batch_size, 128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for i in T.serial(batch_size):
                T.tcgen05_gemm(
                    A_shared[i, :, :],
                    B_shared[i, :, :],
                    C_tmem[i, :, :],
                    transpose_B=True,
                    mbar=done,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(done, i % 2)
            T.copy(C_tmem, C_local)
            for i in T.serial(batch_size):
                T.copy(C_local[i, :, :], C_shared)
                T.copy(C_shared, D[i, :, :])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_ld_whole_batched_buffer_correctness():
    import torch

    batch_size = 3
    kernel = tilelang.compile(
        _make_batched_whole_buffer_ld_kernel(batch_size),
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    source = kernel.get_kernel_source()
    # One instruction covers all batch_size * 128 columns of the buffer.
    assert f"tl::tcgen05_ld_32dp32bNx<{batch_size * 128}, false>" in source
    a = torch.randn(batch_size, 128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch_size, 128, 64, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(batch_size, 128, 128, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().transpose(1, 2)).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


def _make_fp16_accum_tcgen05_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float8_e4m3fn),
        B: T.Tensor((128, 128), T.float8_e4m3fn),
        D: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            B_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            C_tmem = T.alloc_tmem((128, 128), T.float16)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float16)
            C_shared = T.alloc_shared((128, 128), T.float16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, clear_accum=True)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_fp16_accumulator_pack16b_epilogue():
    import torch

    # A 16-bit matrix D element occupies the lower 16 bits of its own 32-bit
    # tensor-memory word (PTX ISA "Packing format for matrix D in Tensor
    # Memory"), so an FP8 GEMM with an fp16 accumulator writes a TMEM
    # fragment the epilogue copy must plan in b32 columns and gather with
    # the pack::16b modifier -- one register per two b32 columns, hence x64
    # for 128 b32 columns.
    kernel = tilelang.compile(_make_fp16_accum_tcgen05_kernel(), target="cuda")
    source = kernel.get_kernel_source()
    ld_line = next(line for line in source.splitlines() if "tcgen05_ld_" in line)
    assert "tcgen05_ld_32dp32bNx<64, true>" in ld_line

    a = (torch.randn(128, 128, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(128, 128, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    d = torch.empty(128, 128, device="cuda", dtype=torch.float16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().T).to(torch.float16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_batched_correctness():
    import torch

    # Three batch entries: dynamic indices reach past the first two windows
    # (C_tmem[2, ...]), so the test observes real column repetition rather
    # than a base-window access that a batch size of one would also hit.
    batch_size = 3
    kernel = tilelang.compile(
        _make_batched_tcgen05_kernel(batch_size),
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(batch_size, 128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch_size, 128, 64, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(batch_size, 128, 128, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().transpose(1, 2)).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)
