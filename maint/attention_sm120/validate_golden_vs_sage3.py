"""S1 acceptance: (i) our quantizer semantics are byte-exact vs the original fp4quant_cuda
kernels; (ii) the golden reference reproduces the original fp4attn_cuda kernel output
(alignment ratio << 1 against the fp64 reference error); (iii) our torch preprocess matches
the original's (Triton) preprocess.

Requires the original package (thu-ml/SageAttention sageattention3_blackwell) installed.
"""

import argparse
import sys

import torch

sys.path.insert(0, "/root/qutao/tilelang_dev")
from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402

import fp4attn_cuda  # noqa: E402
import fp4quant_cuda  # noqa: E402
from sageattn3 import api as sage_api  # noqa: E402


def sage_quant(q_sm, k_sm, v):
    q_pk, q_sf = sage_api.scale_and_quant_fp4(q_sm)
    k_pk, k_sf = sage_api.scale_and_quant_fp4_permute(k_sm)
    v_pk, v_sf = sage_api.scale_and_quant_fp4_transpose(v)
    return dict(q=q_pk, k=k_pk, v=v_pk, sfq=q_sf, sfk=k_sf, sfv=v_sf)


def sage_fwd(inp, delta_s, kl, d):
    softmax_scale = d ** (-0.5)
    out, _ = fp4attn_cuda.fwd(
        inp["q"], inp["k"], inp["v"], inp["sfq"], inp["sfk"], inp["sfv"], delta_s, kl, None, softmax_scale, False, True, True
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,4096")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = args.dim
    for n in (int(s) for s in args.sizes.split(",")):
        q = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=torch.bfloat16)
        o_ref = sq.reference_attention(q, k, v, d ** (-0.5))

        # --- original pipeline ---
        q_o, k_o, v_o, ds_o = sage_api.preprocess_qkv(q.clone(), k.clone(), v.clone(), True)
        inp_o = sage_quant(q_o, k_o, v_o)
        o_sage = sage_fwd(inp_o, ds_o, n, d).float()

        # --- (iii) our preprocess vs original ---
        q_t, k_t, v_t, qm_t, ds_t = sq.preprocess_qkv(q.clone(), k.clone(), v.clone())
        print(
            f"[N={n}] preprocess: k exact={torch.equal(k_t, k_o)} q max|diff|={float((q_t.float() - q_o.float()).abs().max()):.3e} "
            f"delta_s max|diff|={float((ds_t - ds_o).abs().max()):.3e} (rel {float((ds_t - ds_o).abs().max() / ds_o.abs().max()):.2e})"
        )

        # --- (i) quantizer byte-exactness on the ORIGINAL's preprocessed tensors ---
        canon_o = sq.canonical_from_sage(**inp_o)
        canon_t = sq.quantize_canonical(q_o, k_o, v_o)
        for key in ("q_codes", "q_sf", "k_codes", "k_sf", "vt_codes", "vt_sf"):
            a, b = canon_t[key], canon_o[key]
            neq = int((a != b).sum())
            print(
                f"[N={n}] quant {key:8s}: mismatching bytes = {neq} / {a.numel()}"
                + (
                    ""
                    if neq == 0
                    else f"  e.g. ours={a.flatten()[(a != b).flatten().nonzero()[0]].item()} theirs={b.flatten()[(a != b).flatten().nonzero()[0]].item()}"
                )
            )
        # exporter round trip: export_for_sage(canonical) must equal their tensors byte-for-byte
        exp = sq.export_for_sage(canon_o)
        for key in ("q", "k", "v"):
            print(f"[N={n}] export {key}: equal={torch.equal(exp[key], inp_o[key])}")
        for key in ("sfq", "sfk", "sfv"):
            print(f"[N={n}] export {key}: equal={torch.equal(exp[key].view(torch.uint8), inp_o[key].view(torch.uint8))}")

        # --- (ii) golden vs original kernel ---
        for rt in (False, True):
            o_gold = sq.golden_attention(canon_o, ds_o, d ** (-0.5), round_trip_p_scale=rt)
            print(
                f"[N={n}] golden(round_trip={rt}) vs sage: align_ratio={sq.alignment_ratio(o_gold, o_sage, o_ref):.4f} "
                f"max|diff|={float((o_gold - o_sage).abs().max()):.3e}  | golden vs ref: cos={sq.cos_sim(o_gold, o_ref):.6f} L1={sq.rel_l1(o_gold, o_ref):.4f}"
            )
        print(
            f"[N={n}] sage vs ref: cos={sq.cos_sim(o_sage, o_ref):.6f} L1={sq.rel_l1(o_sage, o_ref):.4f} rmse={sq.rmse(o_sage, o_ref):.4e}"
        )


if __name__ == "__main__":
    main()
