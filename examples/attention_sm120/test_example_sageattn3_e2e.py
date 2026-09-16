"""Tests for the TileLang SageAttention3 preprocessing kernels and the end-to-end API."""

import pytest

import tilelang
import tilelang.testing

torch = pytest.importorskip("torch")

from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402
from examples.attention_sm120 import sageattn3_prep as prep  # noqa: E402
from examples.attention_sm120 import sageattn3_e2e as e2e  # noqa: E402
from examples.attention_sm120 import sm120_sageattn3_fwd_ws as ws  # noqa: E402

D = 128


def _qkv(b, h, n, seed=0, outliers=True):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(b, h, n, D, device="cuda", dtype=torch.bfloat16) * 3 for _ in range(3))
    if outliers:
        q[..., 5, :] = 0  # an all-zero row: scale 0, codes 0
        k[..., 7, 17] = 5000.0  # a saturating outlier
        v[..., 3, 40] = -3000.0
    return q, k, v


def _host_means(q, k):
    b, h, n, _ = q.shape
    qm = (q.float().view(b, h, n // 128, 128, D).sum(dim=3) / 128).to(torch.bfloat16)
    km = (k.float().sum(dim=2, keepdim=True) / n).to(torch.bfloat16)
    return qm, km


def _host_inputs(q, k, v, qm, km):
    """The attention kernel inputs computed on the host reference (sageattn3_quant.py)."""
    q_sm = (q.float() - qm.float().repeat_interleave(128, dim=2)).to(torch.bfloat16)
    k_sm = (k.float() - km.float()).to(torch.bfloat16)
    t = sq.export_for_tilelang(sq.quantize_canonical(q_sm, k_sm, v))
    return t, k_sm


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("b,h,n", [(1, 1, 128), (2, 3, 1024), (1, 2, 16384)])
def test_sa3_means_exact(b, h, n):
    q, k, _ = _qkv(b, h, n, outliers=False)
    qm, km = prep.sa3_means(b, h, n)(q, k)
    qm_ref, km_ref = _host_means(q, k)
    assert torch.equal(qm, qm_ref)
    assert torch.equal(km, km_ref)
    assert torch.equal(km, k.mean(dim=-2, keepdim=True))  # what the original uses for K


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("b,h,n", [(1, 1, 128), (2, 3, 512), (4, 32, 1024)])
@pytest.mark.parametrize("emit_ksm", [False, True])
def test_sa3_quant_byte_exact(b, h, n, emit_ksm):
    q, k, v = _qkv(b, h, n)
    qm, km = _host_means(q, k)
    qp, kp, vtp, sfq, sfk, sfv, ksm = prep.sa3_quant(b, h, n, emit_ksm=emit_ksm)(q, k, v, qm, km)
    ref, k_sm = _host_inputs(q, k, v, qm, km)
    assert torch.equal(qp.view(torch.uint8).reshape(b, h, n, D // 2), ref["q"].view(torch.uint8))
    assert torch.equal(kp.view(torch.uint8).reshape(b, h, n, D // 2), ref["k"].view(torch.uint8))
    assert torch.equal(vtp.view(torch.uint8).reshape(b, h, D, n // 2), ref["vt"].view(torch.uint8))
    for got, want in ((sfq, ref["sfq"]), (sfk, ref["sfk"]), (sfv, ref["sfv"])):
        assert torch.equal(got.view(torch.uint32).view(torch.int32), want.view(torch.int32))
    if emit_ksm:
        assert torch.equal(ksm, k_sm)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("b,h,n", [(1, 2, 256), (4, 32, 1024), (1, 4, 4096)])
def test_sageattn3_tl_matches_host_pipeline(b, h, n):
    """End to end == the fp32-delta_s attention kernel on inputs prepared by the host reference.

    Covers both smoothed-K paths (torch subtraction below 2K, kernel output from 2K) and the bf16
    delta_s kernel, which must be bit-identical to feeding the same values in fp32.
    """
    q, k, v = _qkv(b, h, n, outliers=False)
    out = e2e.sageattn3_tl(q, k, v)
    qm, km = _host_means(q, k)
    t, k_sm = _host_inputs(q, k, v, qm, km)
    ds = torch.matmul(qm, k_sm.transpose(-2, -1)).float()
    ref = ws.sm120_sageattn3_fwd(b, h, n, D)(
        t["q"].view(torch.int8), t["k"].view(torch.int8), t["vt"].view(torch.int8), t["sfq"], t["sfk"], t["sfv"], ds
    )
    assert torch.equal(out, ref)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_tl_agrees_with_original():
    """Against ``sageattn3_blackwell``: same quality, and the only preprocessing difference is the Q
    block mean.

    With the original's own Triton block mean substituted, this pipeline reproduces exactly what
    our attention kernel computes on the original's preprocessing and quantization kernels. With
    the exact mean it uses by default, output quality against an fp64 reference is unchanged, while
    the alignment ratio to the original grows from ~0.16-0.18 to ~0.24.
    """
    pytest.importorskip("fp4attn_cuda")
    sage_api = pytest.importorskip("sageattn3.api")
    from sageattn3 import sageattn3_blackwell

    b, h, n = 4, 32, 1024
    q, k, v = _qkv(b, h, n, outliers=False)
    o_tl = e2e.sageattn3_tl(q, k, v)
    o_orig = sageattn3_blackwell(q, k.clone(), v)
    o_ref = sq.reference_attention(q, k, v, D**-0.5)
    assert sq.alignment_ratio(o_tl.float(), o_orig.float(), o_ref) < 0.3  # measured 0.236
    assert abs(sq.cos_sim(o_tl.float(), o_ref) - sq.cos_sim(o_orig.float(), o_ref)) < 1e-4
    assert abs(sq.rel_l1(o_tl.float(), o_ref) - sq.rel_l1(o_orig.float(), o_ref)) < 5e-3

    # the original's Triton block mean through our kernels == our kernel on the original's preprocessing
    means, quant, attn, _ = e2e._kernels(b, h, n, D)
    _, qm_tr = sage_api.triton_group_mean(q.clone())
    _, km = means(q, k)
    qp, kp, vtp, sfq, sfk, sfv, _ = quant(q, k, v, qm_tr, km)
    ds16 = torch.matmul(qm_tr, (k - km).transpose(-2, -1))
    o_tr = attn(
        qp.view(torch.int8),
        kp.view(torch.int8),
        vtp.view(torch.int8),
        sfq.view(torch.uint32),
        sfk.view(torch.uint32),
        sfv.view(torch.uint32),
        ds16,
    )
    q_p, k_p, v_p, ds = sage_api.preprocess_qkv(q.clone(), k.clone(), v.clone(), True)
    (qc, qs), (kc, ks), (vc, vs) = (
        sage_api.scale_and_quant_fp4(q_p),
        sage_api.scale_and_quant_fp4_permute(k_p),
        sage_api.scale_and_quant_fp4_transpose(v_p),
    )
    t = sq.export_for_tilelang(sq.canonical_from_sage(qc, kc, vc, qs, ks, vs))
    o_kern = ws.sm120_sageattn3_fwd(b, h, n, D)(
        t["q"].view(torch.int8), t["k"].view(torch.int8), t["vt"].view(torch.int8), t["sfq"], t["sfk"], t["sfv"], ds
    )
    assert torch.equal(o_tr, o_kern)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_tl_leaves_inputs_alone_and_validates():
    q, k, v = _qkv(1, 2, 256, outliers=False)
    k_copy = k.clone()
    e2e.sageattn3_tl(q, k, v)
    assert torch.equal(k, k_copy)  # the original subtracts the K mean in place; this API does not
    with pytest.raises(ValueError, match="multiple of 128"):
        e2e.sageattn3_tl(q[:, :, :200], k[:, :, :200], v[:, :, :200])
    with pytest.raises(ValueError, match="bfloat16"):
        e2e.sageattn3_tl(q.float(), k.float(), v.float())
    with pytest.raises(ValueError, match="same shape"):
        e2e.sageattn3_tl(q, k[:, :, :128], v[:, :, :128])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_tl_perf_smoke():
    """Runs and reports end-to-end TOPS (no threshold)."""
    from tilelang.profiler import do_bench

    b, h, n = 4, 32, 2048
    q, k, v = _qkv(b, h, n, outliers=False)
    ms = do_bench(lambda: e2e.sageattn3_tl(q, k, v), warmup=10, rep=30, backend="cudagraph", return_mode="median")
    tops = 4.0 * b * h * n * n * D / (ms * 1e-3) / 1e12
    print(f"\nsageattn3 TileLang e2e B={b} H={h} N={n}: {ms:.4f} ms, {tops:.1f} TOPS")
    assert tops > 0


if __name__ == "__main__":
    tilelang.testing.main()
