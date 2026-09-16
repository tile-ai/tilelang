"""Contract tests for the SageAttention3 NVFP4 quantization module (examples/attention_sm120)."""

import pytest

torch = pytest.importorskip("torch")
tilelang_testing = pytest.importorskip("tilelang.testing")

from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402


def test_e2m1_rn_even_ties_and_saturation():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.0, -0.3, 6.5, 0.0])
    # ties -> even code; > 6 saturates to 6 (code 7); -0.0 keeps its sign bit
    assert sq.e2m1_rn_codes(x).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 8, 9, 7, 0]
    vals = sq.e2m1_decode(torch.arange(16, dtype=torch.uint8)).tolist()
    assert vals == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def test_e4m3_rn_round_trip_and_saturation():
    b, v = sq.e4m3_rn(torch.tensor([1.0625, 1.1875, 448.0, 1000.0, 0.0]))
    assert v.tolist() == [1.0, 1.25, 448.0, 448.0, 0.0]
    assert torch.equal(sq.ue4m3_decode(b), v)


def test_permutation_and_layout_round_trips():
    assert sq.PERM32.tolist()[:8] == [0, 1, 8, 9, 16, 17, 24, 25]
    assert sorted(sq.PERM32.tolist()) == list(range(32))
    c = torch.randint(0, 16, (2, 3, 256, 128), dtype=torch.uint8)
    assert torch.equal(sq.unpermute_keys(sq.permute_keys(c)), c)
    assert torch.equal(sq.unpack_nibbles_to_codes(sq.pack_codes_to_nibbles(c)), c)
    sf = torch.randint(0, 127, (2, 3, 256, 8), dtype=torch.uint8)
    assert torch.equal(sq.sf_from_sage_layout(sq.sf_to_sage_layout(sf)), sf)
    words = sq.sf_to_rowmajor_words(sf)
    assert words.dtype == torch.uint32 and words.shape == (2, 3, 256, 2)
    assert int(words[0, 0, 0, 0]) == int(sf[0, 0, 0, 0]) | (int(sf[0, 0, 0, 1]) << 8) | (int(sf[0, 0, 0, 2]) << 16) | (
        int(sf[0, 0, 0, 3]) << 24
    )


def test_quantize_rows_matches_reference_formula():
    x = torch.randn(4, 64) * 3
    codes, sf = sq.quantize_rows_nvfp4(x)
    amax = x.reshape(4, 4, 16).abs().amax(-1)
    _, sfv = sq.e4m3_rn(amax / 6.0)
    ref_codes = sq.e2m1_rn_codes(x.reshape(4, 4, 16) / sfv.unsqueeze(-1)).reshape(4, 64)
    assert torch.equal(codes, ref_codes)
    deq = sq.dequantize_rows_nvfp4(codes, sf)
    # e2m1 spacing is at most 2 (between 4 and 6): |error| <= scale * 1.0
    assert float((deq - x).abs().max()) <= float(sfv.max()) * 1.0 + 1e-6


def test_golden_matches_exact_softmax_when_quantization_is_lossless():
    # Inputs that are exactly representable (e2m1 codes x power-of-two scales) and a flat
    # softmax: P is uniform so the two-level P quantization is exact -> golden == fp64 softmax(QK)V.
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 256, 128
    q = torch.zeros(b, h, n, d)  # S == 0 -> P uniform -> exact
    k = torch.randn(b, h, n, d)
    v = sq.e2m1_decode(torch.randint(0, 16, (b, h, n, d), dtype=torch.uint8))
    v[..., 0::16, :] = 6.0  # V scale groups run along KEYS: every 16-key group has amax 6 -> scale 1.0 -> exact
    canon = sq.quantize_canonical(q, k, v)
    ds = torch.zeros(b, h, n // 128, n)
    o = sq.golden_attention(canon, ds, d**-0.5)
    ref = sq.reference_attention(q, k, v, d**-0.5)
    torch.testing.assert_close(o, ref, rtol=0, atol=1e-6)


@tilelang_testing.requires_cuda
@tilelang_testing.requires_cuda_compute_version_eq(12, 0)
def test_quantizer_byte_exact_vs_original_kernels():
    """Byte-exact against thu-ml/SageAttention's fp4quant_cuda; golden aligns with fp4attn_cuda."""
    pytest.importorskip("fp4quant_cuda")
    fp4attn_cuda = pytest.importorskip("fp4attn_cuda")
    from sageattn3 import api as sage_api

    torch.manual_seed(0)
    b, h, n, d = 1, 2, 1024, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q_o, k_o, v_o, ds = sage_api.preprocess_qkv(q.clone(), k.clone(), v.clone(), True)
    q_pk, q_sf = sage_api.scale_and_quant_fp4(q_o)
    k_pk, k_sf = sage_api.scale_and_quant_fp4_permute(k_o)
    v_pk, v_sf = sage_api.scale_and_quant_fp4_transpose(v_o)
    theirs = sq.canonical_from_sage(q_pk, k_pk, v_pk, q_sf, k_sf, v_sf)
    ours = sq.quantize_canonical(q_o, k_o, v_o)
    for key in theirs:
        assert torch.equal(ours[key], theirs[key]), key
    exp = sq.export_for_sage(ours)
    assert torch.equal(exp["k"], k_pk) and torch.equal(exp["sfk"].view(torch.uint8), k_sf.view(torch.uint8))
    o_sage, _ = fp4attn_cuda.fwd(q_pk, k_pk, v_pk, q_sf, k_sf, v_sf, ds, n, None, d**-0.5, False, True, True)
    o_ref = sq.reference_attention(q, k, v, d**-0.5)
    o_gold = sq.golden_attention(ours, ds, d**-0.5)
    # Golden vs original: residual comes from ex2.approx / fast-math div / fp32 MMA order
    # (measured 0.16-0.18 on this GPU); the round-trip variant is provably further (~0.33).
    assert sq.alignment_ratio(o_gold, o_sage.float(), o_ref) < 0.25
    assert sq.alignment_ratio(sq.golden_attention(ours, ds, d**-0.5, round_trip_p_scale=True), o_sage.float(), o_ref) > sq.alignment_ratio(
        o_gold, o_sage.float(), o_ref
    )


if __name__ == "__main__":
    tilelang_testing.main()
