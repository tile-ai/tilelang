from dataclasses import dataclass
import re

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@dataclass(frozen=True)
class Case:
    M: int
    K: int
    N: int
    scale_dtype: str
    scale_packed: bool
    rel_tol: float
    trans_a: bool = False
    trans_b: bool = True


def make_l0_blockscaled_gemm_kernel(
    M: int,
    K: int,
    N: int,
    scale_dtype: str,
    scale_packed: bool,
    trans_a: bool,
    trans_b: bool,
    tile_m=None,
    tile_n=None,
):
    dtype = "float8_e4m3fn"
    sf_div = 64 if scale_packed else 32
    sf_k = K // sf_div
    sf_k_e8m0 = K // 32
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    mad_m = M if tile_m is None else tile_m
    mad_n = N if tile_n is None else tile_n

    @T.prim_func
    def gemm_kernel(
        X: T.Buffer(a_shape, dtype),
        W: T.Buffer(b_shape, dtype),
        SFX: T.Buffer((sf_k, M) if scale_packed else (M, sf_k_e8m0), scale_dtype),
        SFW: T.Buffer((sf_k, N) if scale_packed else (N, sf_k_e8m0), scale_dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1(a_shape, dtype)
            w_l1 = T.alloc_l1(b_shape, dtype)
            xsf_l1 = T.alloc_l1((M, sf_k), scale_dtype)
            wsf_l1 = T.alloc_l1((N, sf_k), scale_dtype)
            x_l0a = T.alloc_l0a(a_shape, dtype)
            w_l0b = T.alloc_l0b(b_shape, dtype)

            T.copy(X, x_l1)
            T.copy(W, w_l1)
            if scale_packed:
                T.copy(SFX, xsf_l1, transpose=True)
                T.copy(SFW, wsf_l1, transpose=True)
            else:
                T.copy(SFX, xsf_l1)
                T.copy(SFW, wsf_l1)
            if trans_a:
                T.copy(
                    x_l1[0:K, 0:mad_m],
                    x_l0a[0:K, 0:mad_m],
                    scale=xsf_l1[0:mad_m, 0:sf_k],
                )
            else:
                T.copy(
                    x_l1[0:mad_m, 0:K],
                    x_l0a[0:mad_m, 0:K],
                    scale=xsf_l1[0:mad_m, 0:sf_k],
                )
            if trans_b:
                T.copy(
                    w_l1[0:mad_n, 0:K],
                    w_l0b[0:mad_n, 0:K],
                    scale=wsf_l1[0:mad_n, 0:sf_k],
                )
            else:
                T.copy(
                    w_l1[0:K, 0:mad_n],
                    w_l0b[0:K, 0:mad_n],
                    scale=wsf_l1[0:mad_n, 0:sf_k],
                )
            if trans_a:
                if trans_b:
                    T.blockscaled_gemm(
                        x_l0a[0:K, 0:mad_m],
                        w_l0b[0:mad_n, 0:K],
                        res[0:mad_m, 0:mad_n],
                        transpose_A=True,
                        transpose_B=True,
                        clear_accum=True,
                    )
                else:
                    T.blockscaled_gemm(
                        x_l0a[0:K, 0:mad_m],
                        w_l0b[0:K, 0:mad_n],
                        res[0:mad_m, 0:mad_n],
                        transpose_A=True,
                        transpose_B=False,
                        clear_accum=True,
                    )
            else:
                if trans_b:
                    T.blockscaled_gemm(
                        x_l0a[0:mad_m, 0:K],
                        w_l0b[0:mad_n, 0:K],
                        res[0:mad_m, 0:mad_n],
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )
                else:
                    T.blockscaled_gemm(
                        x_l0a[0:mad_m, 0:K],
                        w_l0b[0:K, 0:mad_n],
                        res[0:mad_m, 0:mad_n],
                        transpose_A=False,
                        transpose_B=False,
                        clear_accum=True,
                    )
            T.copy(res[0:mad_m, 0:mad_n], C[0:mad_m, 0:mad_n])

    return gemm_kernel


def make_inputs(case: Case):
    dtype = torch.float8_e4m3fn
    x_logical = (torch.randn(case.M, case.K, device="npu") * 0.1).to(dtype)
    w_logical = (torch.randn(case.N, case.K, device="npu") * 0.1).to(dtype)
    x = x_logical.T.contiguous() if case.trans_a else x_logical
    w = w_logical if case.trans_b else w_logical.T.contiguous()

    sf_k_e8m0 = case.K // 32
    sfx_e8m0 = torch.randint(124, 131, (case.M, sf_k_e8m0), dtype=torch.uint8, device="npu")
    sfw_e8m0 = torch.randint(124, 131, (case.N, sf_k_e8m0), dtype=torch.uint8, device="npu")

    if case.scale_packed:
        sfx = sfx_e8m0.view(torch.uint16).T.contiguous()
        sfw = sfw_e8m0.view(torch.uint16).T.contiguous()
    else:
        scale_torch_dtype = getattr(torch, case.scale_dtype)
        sfx = sfx_e8m0.view(scale_torch_dtype)
        sfw = sfw_e8m0.view(scale_torch_dtype)
    return x, w, sfx, sfw, x_logical, w_logical, sfx_e8m0, sfw_e8m0


def ref_program(x, w, sfx_e8m0, sfw_e8m0):
    k = x.shape[1]
    sx = torch.pow(2.0, sfx_e8m0.float() - 127.0).repeat_interleave(32, dim=1)[:, :k]
    sw = torch.pow(2.0, sfw_e8m0.float() - 127.0).repeat_interleave(32, dim=1)[:, :k]
    return (x.float() * sx) @ (w.float() * sw).T


def run_case(case: Case):
    program = make_l0_blockscaled_gemm_kernel(
        case.M,
        case.K,
        case.N,
        case.scale_dtype,
        case.scale_packed,
        case.trans_a,
        case.trans_b,
    )
    kernel = tilelang.compile(program, out_idx=-1)

    x, w, sfx, sfw, x_logical, w_logical, sfx_e8m0, sfw_e8m0 = make_inputs(case)
    actual = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()

    expected = ref_program(x_logical, w_logical, sfx_e8m0, sfw_e8m0)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel = ((actual - expected).abs().mean() / denom).item()
    if rel < case.rel_tol:
        return

    diff = (actual - expected).abs()
    row, col = (int(v.item()) for v in torch.nonzero(diff == diff.max())[0])
    pytest.fail(
        f"shape=({case.M}, {case.K}, {case.N}) scale={case.scale_dtype} "
        f"packed={case.scale_packed} rel={rel:.4e} "
        f"actual[{row},{col}]={actual[row, col].item():.6g} "
        f"expected[{row},{col}]={expected[row, col].item():.6g}"
    )


def default_cases():
    return [
        Case(1, 128, 15, "uint16", True, 2e-1),
        Case(15, 128, 15, "uint16", True, 2e-1),
        Case(32, 128, 32, "uint16", True, 2e-1),
        Case(100, 128, 100, "uint16", True, 2e-1),
        Case(64, 128, 64, "uint16", True, 2e-1, False, False),
        Case(64, 128, 64, "uint16", True, 2e-1, True, False),
        Case(64, 128, 64, "uint16", True, 2e-1, True, True),
    ]


def case_id(case: Case) -> str:
    packed = "packed" if case.scale_packed else "unpacked"
    suffix = ("t" if case.trans_a else "n") + ("t" if case.trans_b else "n")
    return f"{case.M}x{case.K}x{case.N}-{suffix}-{case.scale_dtype}-{packed}"


@pytest.mark.parametrize("case", default_cases(), ids=case_id)
def test_l0_blockscaled_gemm_matrix(case: Case):
    torch.manual_seed(42)
    run_case(case)


def test_l0_blockscaled_gemm_regions_drive_mn_geometry():
    source = tilelang.lower(
        make_l0_blockscaled_gemm_kernel(
            32,
            128,
            32,
            "uint16",
            True,
            False,
            True,
            tile_m=17,
            tile_n=19,
        ),
        target="ascend",
    ).kernel_source
    assert re.search(r"mad_mx\([^;]*,\s*17,\s*128,\s*19,", source), source


if __name__ == "__main__":
    tilelang.testing.main()
