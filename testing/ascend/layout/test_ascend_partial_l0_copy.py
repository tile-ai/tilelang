"""Explicit copy l0_pitch assembles L0 tiles across compact tails and ring reuse."""

import pytest
import torch

import tilelang
from tilelang.ascend import language as T


def _partial_copy_kernel(mode, dtype, versions):
    steps, n, k_alloc = 7, 64, 64
    m, m_alloc = (40, 64) if mode == "m-tail" else (64, 64)
    k = 56 if mode == "k-tail" else k_alloc
    v_layout = mode in ("v", "k-tail", "v-tail")

    @T.prim_func
    def main(
        A: T.Tensor((steps, m_alloc, k_alloc), dtype),
        B: T.Tensor((steps, n, k_alloc), dtype),
        C: T.Tensor((steps, m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m_alloc, k_alloc), dtype)
            b_l1 = T.alloc_l1((n, k_alloc), dtype)
            a = T.alloc_l0a((m_alloc, k_alloc), dtype)
            b = T.alloc_l0b((n, k_alloc), dtype)
            acc = T.alloc_l0c((m_alloc, n), "float32")
            T.annotate_buffer_versions({a: versions, b: versions})
            for e in T.Pipelined(steps, num_stages=versions):
                effective_m = 24 + 16 * (e % 3) if mode == "dynamic-tail" else m
                effective_n = 24 + 16 * (e % 3) if mode == "v-tail" else n
                padded_m = T.ceildiv(effective_m, 16) * 16
                padded_n = T.ceildiv(effective_n, 16) * 16
                T.copy(A[e, :, :], a_l1)
                T.copy(B[e, :, :], b_l1)
                if mode == "overwrite":
                    T.copy(a_l1, a)
                    T.copy(a_l1[:32, :], a[32:64, :], l0_pitch=T.ceildiv(effective_m, 16))
                elif mode in ("m-tail", "dynamic-tail"):
                    T.copy(a_l1[:16, :], a[:16, :], l0_pitch=T.ceildiv(effective_m, 16))
                    T.copy(a_l1[16:padded_m, :], a[16:padded_m, :], l0_pitch=T.ceildiv(effective_m, 16))
                else:
                    for part in T.serial(2):
                        p = 1 - part if mode == "reverse" else part
                        if mode == "k-tail":
                            T.copy(a_l1[:, p * 32 : (p + 1) * 32], a[:, p * 32 : (p + 1) * 32], l0_pitch=T.ceildiv(effective_m, 16))
                        else:
                            T.copy(a_l1[p * 32 : (p + 1) * 32, :], a[p * 32 : (p + 1) * 32, :], l0_pitch=T.ceildiv(effective_m, 16))
                for part in T.serial(2):
                    # B is [K,N] for V, [N,K] otherwise. GEMM anchors the
                    # MajorMN/MajorK layout; the copy has no explicit transpose.
                    if mode == "v-tail":
                        T.copy(
                            b_l1[part * 32 : (part + 1) * 32, :padded_n],
                            b[part * 32 : (part + 1) * 32, :padded_n],
                            l0_pitch=T.ceildiv(effective_n, 16),
                        )
                    else:
                        T.copy(b_l1[part * 32 : (part + 1) * 32, :], b[part * 32 : (part + 1) * 32, :], l0_pitch=T.ceildiv(effective_n, 16))
                if v_layout:
                    T.gemm(a[:effective_m, :k], b[:k, :effective_n], acc[:effective_m, :effective_n], clear_accum=True)
                else:
                    T.gemm(a[:effective_m, :k], b[:, :k], acc[:effective_m, :], transpose_B=True, clear_accum=True)
                drain_m = 16 if mode == "partial-drain" else effective_m
                T.copy(
                    acc[:drain_m, :effective_n],
                    C[e, :drain_m, :effective_n],
                    l0_pitch=T.ceildiv(effective_m, 16) if mode == "partial-drain" else None,
                )

    return main


@pytest.mark.parametrize(
    "mode,dtype,versions",
    [
        ("ordinary", "bfloat16", 2),
        ("reverse", "bfloat16", 3),
        ("v", "bfloat16", 2),
        ("k-tail", "bfloat16", 3),
        ("m-tail", "bfloat16", 2),
        ("v-tail", "bfloat16", 3),
        ("dynamic-tail", "bfloat16", 3),
        ("overwrite", "float16", 2),
        ("partial-drain", "float16", 2),
    ],
)
def test_partial_copies_feed_full_gemm_across_versions(mode, dtype, versions):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    m, m_alloc = (40, 64) if mode == "m-tail" else (64, 64)
    k = 56 if mode == "k-tail" else 64
    generator = torch.Generator().manual_seed(23)
    a = torch.randint(-4, 5, (7, m_alloc, 64), generator=generator).to(getattr(torch, dtype)) / 8
    b = torch.randint(-4, 5, (7, 64, 64), generator=generator).to(getattr(torch, dtype)) / 8
    # Different block/epoch values expose wrong pitch, missing loads and old slots.
    a[:, 32:, :] += 1
    a += torch.arange(7, dtype=getattr(torch, dtype))[:, None, None] / 8
    if mode == "k-tail":
        a[:, :, k:] = 2
        b[:, k:, :] = 3
    effective_a = torch.cat((a[:, :32], a[:, :32]), dim=1) if mode == "overwrite" else a
    effective_b = b[:, :k, :] if mode in ("v", "k-tail", "v-tail") else b[:, :, :k].transpose(-1, -2)
    expected = effective_a[:, :m, :k].float() @ effective_b.float()
    compiled = tilelang.compile(_partial_copy_kernel(mode, dtype, versions), out_idx=-1, target="ascend")
    actual = compiled(a.npu(), b.npu())
    torch.npu.synchronize()
    actual = actual.cpu()
    for e in range(7):
        valid_m = 24 + 16 * (e % 3) if mode == "dynamic-tail" else m
        if mode == "partial-drain":
            valid_m = 16
        valid_n = 24 + 16 * (e % 3) if mode == "v-tail" else 64
        torch.testing.assert_close(actual[e, :valid_m, :valid_n], expected[e, :valid_m, :valid_n], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("v_layout", [False, True], ids=["ordinary", "v-transpose"])
def test_manual_partial_copies_with_branch_specific_gemm_pitch(v_layout):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")

    @T.prim_func
    def main(A: T.Tensor((3, 64, 64), "bfloat16"), B: T.Tensor((3, 64, 64), "bfloat16"), C: T.Tensor((3, 64, 64), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            b_l1 = T.alloc_l1((64, 64), "bfloat16")
            a = T.alloc_l0a((64, 64), "bfloat16")
            b = T.alloc_l0b((64, 64), "bfloat16")
            acc = T.alloc_l0c((64, 64), "float32")
            with T.Cube():
                for e in T.serial(3):
                    m = 24 + 32 * (e % 2)
                    n = 56 - 32 * (e % 2)
                    padded_m = T.ceildiv(m, 16) * 16
                    padded_n = T.ceildiv(n, 16) * 16
                    T.copy(A[e, :, :], a_l1)
                    T.copy(B[e, :, :], b_l1)
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    for part in T.serial(2):
                        T.copy(
                            a_l1[:padded_m, part * 32 : (part + 1) * 32],
                            a[:padded_m, part * 32 : (part + 1) * 32],
                            l0_pitch=T.ceildiv(m, 16),
                        )
                        if v_layout:
                            T.copy(
                                b_l1[part * 32 : (part + 1) * 32, :padded_n],
                                b[part * 32 : (part + 1) * 32, :padded_n],
                                l0_pitch=T.ceildiv(n, 16),
                            )
                        else:
                            T.copy(
                                b_l1[:padded_n, part * 32 : (part + 1) * 32],
                                b[:padded_n, part * 32 : (part + 1) * 32],
                                l0_pitch=T.ceildiv(n, 16),
                            )
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    if e % 2 == 0:
                        if v_layout:
                            T.gemm(a[:24, :], b[:, :56], acc[:24, :56], clear_accum=True)
                        else:
                            T.gemm(a[:24, :], b[:56, :], acc[:24, :56], transpose_B=True, clear_accum=True)
                    else:
                        if v_layout:
                            T.gemm(a[:56, :], b[:, :24], acc[:56, :24], clear_accum=True)
                        else:
                            T.gemm(a[:56, :], b[:24, :], acc[:56, :24], transpose_B=True, clear_accum=True)
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    # The complete result uses the original compact-copy default.
                    T.copy(acc[:m, :n], C[e, :m, :n])
                    T.ascend_pipe_barrier("PIPE_ALL")

    generator = torch.Generator().manual_seed(31)
    a = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    b = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    a[:, :, 32:] += 1
    expected = a.float() @ (b.float() if v_layout else b.float().transpose(-1, -2))
    compiled = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    actual = compiled(a.npu(), b.npu())
    torch.npu.synchronize()
    actual = actual.cpu()
    for e in range(3):
        m, n = 24 + 32 * (e % 2), 56 - 32 * (e % 2)
        torch.testing.assert_close(actual[e, :m, :n], expected[e, :m, :n], atol=1e-5, rtol=1e-5)
