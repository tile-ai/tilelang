from dataclasses import dataclass

import pytest
import torch
import tilelang
import tilelang.ascend.language as T


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Case:
    m: int
    n: int
    k: int
    k_on_row: bool

    @property
    def m_alloc(self) -> int:
        return ceildiv(self.m, 16) * 16

    @property
    def n_alloc(self) -> int:
        return ceildiv(self.n, 16) * 16

    @property
    def k_alloc(self) -> int:
        return ceildiv(self.k, 64) * 64


def case_id(case: Case) -> str:
    orientation = "k-row" if case.k_on_row else "k-col"
    return f"{case.m}x{case.n}x{case.k}-{orientation}"


BF16_CASES = [
    Case(15, 13, 47, False),
    Case(17, 29, 65, False),
    Case(33, 18, 127, False),
    Case(15, 13, 47, True),
    Case(17, 29, 65, True),
    Case(31, 7, 95, True),
]


FP8_CASES = [
    Case(15, 13, 65, False),
    Case(17, 9, 95, False),
    Case(31, 7, 127, False),
    Case(18, 21, 150, False),
    Case(17, 9, 65, True),
    Case(31, 7, 95, True),
    Case(17, 9, 127, True),
    Case(17, 9, 128, True),
]


def make_bf16_kernel(case: Case):
    m, n, k = case.m, case.n, case.k
    m_alloc, n_alloc, k_alloc = case.m_alloc, case.n_alloc, case.k_alloc
    a_shape = (k, m) if case.k_on_row else (m, k)
    b_shape = (k, n) if case.k_on_row else (n, k)
    a_l1_shape = (k_alloc, m_alloc) if case.k_on_row else (m_alloc, k_alloc)
    b_l1_shape = (k_alloc, n_alloc) if case.k_on_row else (n_alloc, k_alloc)

    @T.prim_func
    def kernel(
        A: T.Buffer(a_shape, "bfloat16"),
        B: T.Buffer(b_shape, "bfloat16"),
        C: T.Buffer((m_alloc, n_alloc), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_l1_shape, "bfloat16")
            b_l1 = T.alloc_l1(b_l1_shape, "bfloat16")
            a_l0 = T.alloc_l0a((m_alloc, k_alloc), "bfloat16")
            b_l0 = T.alloc_l0b((n_alloc, k_alloc), "bfloat16")
            acc = T.alloc_l0c((m_alloc, n_alloc), "float32")

            T.copy(A[0 : a_l1_shape[0], 0 : a_l1_shape[1]], a_l1)
            T.copy(B[0 : b_l1_shape[0], 0 : b_l1_shape[1]], b_l1)
            T.copy(a_l1, a_l0, transpose=case.k_on_row)
            T.copy(b_l1, b_l0, transpose=case.k_on_row)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return kernel


def fp8_mn_alloc(case: Case) -> tuple[int, int]:
    # A transposed FP8 16x32 source fractal produces 32 MN rows in L0.  K-row
    # therefore requires M/N allocation to align to FP8 C0=32, while K-col
    # keeps the ordinary row-fractal alignment of 16.
    mn_align = 32 if case.k_on_row else 16
    return ceildiv(case.m, mn_align) * mn_align, ceildiv(case.n, mn_align) * mn_align


def make_fp8_blockscaled_kernel(case: Case):
    m, n, k = case.m, case.n, case.k
    m_alloc, n_alloc = fp8_mn_alloc(case)
    k_alloc = case.k_alloc
    sf_e8m0 = k_alloc // 32
    a_shape = (k, m) if case.k_on_row else (m, k)
    b_shape = (k, n) if case.k_on_row else (n, k)
    a_l1_shape = (k_alloc, m_alloc) if case.k_on_row else (m_alloc, k_alloc)
    b_l1_shape = (k_alloc, n_alloc) if case.k_on_row else (n_alloc, k_alloc)

    @T.prim_func
    def kernel(
        A: T.Buffer(a_shape, "float8_e4m3fn"),
        B: T.Buffer(b_shape, "float8_e4m3fn"),
        SFA: T.Buffer((m_alloc, sf_e8m0), "uint8"),
        SFB: T.Buffer((n_alloc, sf_e8m0), "uint8"),
        C: T.Buffer((m_alloc, n_alloc), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_l1_shape, "float8_e4m3fn")
            b_l1 = T.alloc_l1(b_l1_shape, "float8_e4m3fn")
            sfa_l1 = T.alloc_l1((m_alloc, sf_e8m0), "uint8")
            sfb_l1 = T.alloc_l1((n_alloc, sf_e8m0), "uint8")
            a_l0 = T.alloc_l0a((m_alloc, k_alloc), "float8_e4m3fn")
            b_l0 = T.alloc_l0b((n_alloc, k_alloc), "float8_e4m3fn")
            a_l0_sf = T.alloc_l0a_sf(a_l0, sf_dtype="uint8")
            b_l0_sf = T.alloc_l0b_sf(b_l0, sf_dtype="uint8")
            acc = T.alloc_l0c((m_alloc, n_alloc), "float32")

            T.copy(A[0 : a_l1_shape[0], 0 : a_l1_shape[1]], a_l1)
            T.copy(B[0 : b_l1_shape[0], 0 : b_l1_shape[1]], b_l1)
            T.copy(SFA, sfa_l1)
            T.copy(SFB, sfb_l1)
            T.copy(a_l1, a_l0, transpose=case.k_on_row)
            T.copy(sfa_l1, a_l0_sf)
            T.copy(b_l1, b_l0, transpose=case.k_on_row)
            T.copy(sfb_l1, b_l0_sf)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_l0_sf, b_l0_sf, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return kernel


def canonical_matrix(x: torch.Tensor, k_on_row: bool) -> torch.Tensor:
    return x.T if k_on_row else x


def padded_result(valid: torch.Tensor, m_alloc: int, n_alloc: int) -> torch.Tensor:
    result = torch.zeros((m_alloc, n_alloc), dtype=torch.float32, device="npu")
    result[: valid.shape[0], : valid.shape[1]] = valid
    return result


def _test_bf16_irregular_padding_gemm(case: Case, target: str):
    torch.manual_seed(42)
    a_shape = (case.k, case.m) if case.k_on_row else (case.m, case.k)
    b_shape = (case.k, case.n) if case.k_on_row else (case.n, case.k)
    a = torch.randn(a_shape, dtype=torch.bfloat16, device="npu")
    b = torch.randn(b_shape, dtype=torch.bfloat16, device="npu")

    kernel = tilelang.compile(make_bf16_kernel(case), target=target, out_idx=-1)
    actual = kernel(a, b)
    torch.npu.synchronize()

    a_matrix = canonical_matrix(a, case.k_on_row).float()
    b_matrix = canonical_matrix(b, case.k_on_row).float()
    expected = padded_result(a_matrix @ b_matrix.T, case.m_alloc, case.n_alloc)
    rel = ((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6)).item()
    assert rel < 1e-2, f"case={case_id(case)} rel_mean_diff={rel:.4e}"


def _test_fp8_blockscaled_irregular_padding_gemm(case: Case, target: str):
    torch.manual_seed(42)
    a_shape = (case.k, case.m) if case.k_on_row else (case.m, case.k)
    b_shape = (case.k, case.n) if case.k_on_row else (case.n, case.k)
    a = (torch.randn(a_shape, device="npu") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(b_shape, device="npu") * 0.1).to(torch.float8_e4m3fn)

    sf_e8m0 = case.k_alloc // 32
    sf_real = ceildiv(case.k, 32)
    m_alloc, n_alloc = fp8_mn_alloc(case)
    sfa = torch.full((m_alloc, sf_e8m0), 127, dtype=torch.uint8, device="npu")
    sfb = torch.full((n_alloc, sf_e8m0), 127, dtype=torch.uint8, device="npu")
    sfa[: case.m, :sf_real] = torch.randint(124, 131, (case.m, sf_real), dtype=torch.uint8, device="npu")
    sfb[: case.n, :sf_real] = torch.randint(124, 131, (case.n, sf_real), dtype=torch.uint8, device="npu")

    kernel = tilelang.compile(make_fp8_blockscaled_kernel(case), target=target, out_idx=-1)
    actual = kernel(a, b, sfa, sfb)
    torch.npu.synchronize()

    a_matrix = canonical_matrix(a, case.k_on_row).float()
    b_matrix = canonical_matrix(b, case.k_on_row).float()
    scale_a = torch.pow(2.0, sfa[: case.m].float() - 127.0).repeat_interleave(32, dim=1)[:, : case.k]
    scale_b = torch.pow(2.0, sfb[: case.n].float() - 127.0).repeat_interleave(32, dim=1)[:, : case.k]
    expected = padded_result((a_matrix * scale_a) @ (b_matrix * scale_b).T, m_alloc, n_alloc)
    rel = ((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6)).item()
    actual_valid = actual[: case.m, : case.n]
    assert rel < 2e-1, (
        f"case={case_id(case)} rel_mean_diff={rel:.4e} "
        f"actual_abs_mean={actual.abs().mean().item():.4e} "
        f"expected_abs_mean={expected.abs().mean().item():.4e} "
        f"nan_total={torch.isnan(actual).sum().item()} "
        f"nan_valid={torch.isnan(actual_valid).sum().item()}"
    )


@pytest.mark.parametrize("case", BF16_CASES, ids=case_id)
def test_bf16_irregular_padding_gemm(case: Case):
    _test_bf16_irregular_padding_gemm(case, target="ascend")


@pytest.mark.pto
@pytest.mark.parametrize("case", BF16_CASES, ids=case_id)
def test_bf16_irregular_padding_gemm_pto(case: Case):
    _test_bf16_irregular_padding_gemm(case, target="pto")


@pytest.mark.parametrize("case", FP8_CASES, ids=case_id)
def test_fp8_blockscaled_irregular_padding_gemm(case: Case):
    _test_fp8_blockscaled_irregular_padding_gemm(case, target="ascend")


@pytest.mark.pto
@pytest.mark.parametrize("case", FP8_CASES, ids=case_id)
def test_fp8_blockscaled_irregular_padding_gemm_pto(case: Case):
    _test_fp8_blockscaled_irregular_padding_gemm(case, target="pto")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
