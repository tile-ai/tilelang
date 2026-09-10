from dataclasses import dataclass

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


@dataclass(frozen=True)
class Case:
    dtype: str
    M: int
    K: int
    N: int
    atol: float
    rtol: float


def make_l0_gemm_kernel(M: int, K: int, N: int, dtype: str):
    @T.prim_func
    def gemm_kernel(
        X: T.Buffer((M, K), dtype),
        W: T.Buffer((N, K), dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            x_l0a = T.alloc_l0a((M, K), dtype)
            w_l0b = T.alloc_l0b((N, K), dtype)
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.copy(x_l1, x_l0a)
            T.copy(w_l1, w_l0b)
            T.gemm(x_l0a, w_l0b, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return gemm_kernel


def make_inputs(case: Case):
    torch_dtype = getattr(torch, case.dtype)
    x = (torch.randn(case.M, case.K, device="npu") * 0.25).to(torch_dtype)
    w = (torch.randn(case.N, case.K, device="npu") * 0.25).to(torch_dtype)
    return x, w


def run_case(case: Case):
    kernel = tilelang.compile(make_l0_gemm_kernel(case.M, case.K, case.N, case.dtype), out_idx=-1)

    x, w = make_inputs(case)
    expected = x.float() @ w.float().T
    actual = kernel(x, w)
    torch.npu.synchronize()

    diff = (actual - expected).abs()
    max_diff = diff.max().item()
    denom = expected.abs().max().clamp_min(1e-6).item()
    rel = max_diff / denom
    ok = max_diff <= case.atol + case.rtol * denom
    if ok:
        return

    message = f"dtype={case.dtype} shape=({case.M}, {case.K}, {case.N}) max_diff={max_diff:.4e} rel={rel:.4e}"
    mismatch = torch.nonzero(diff > case.atol + case.rtol * expected.abs().clamp_min(1e-6))
    if mismatch.numel() > 0:
        row = int(mismatch[0, 0].item())
        col = int(mismatch[0, 1].item())
        message += f" first mismatch=({row}, {col}) actual={actual[row, col].item():.6g} expected={expected[row, col].item():.6g}"
    pytest.fail(message)


def default_cases():
    shapes = [(1, 64, 64), (15, 15, 15), (32, 32, 32), (100, 100, 100)]
    cases = []
    for dtype, atol, rtol in [
        ("bfloat16", 1e-2, 1e-2),
        ("float8_e4m3fn", 1e-1, 5e-2),
    ]:
        for M, K, N in shapes:
            cases.append(Case(dtype, M, K, N, atol, rtol))
    return cases


def case_id(case: Case) -> str:
    return f"{case.dtype}-{case.M}x{case.K}x{case.N}"


@pytest.mark.parametrize("case", default_cases(), ids=case_id)
def test_l0_gemm_matrix(case: Case):
    torch.manual_seed(42)
    run_case(case)


if __name__ == "__main__":
    tilelang.testing.main()
