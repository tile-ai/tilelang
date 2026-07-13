import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.meta_path = [finder for finder in sys.meta_path if finder.__class__.__module__ != "_tilelang_editable"]

import torch
import tilelang.language as T
from tilelang.profiler import do_bench

import deep_gemm
from mqa_logits_sm100 import (
    MQALogitsConfig,
    _tilelang_logits_dtype,
    _torch_logits_dtype,
    calc_diff,
    generate_ks_ke,
    mqa_logits_fp4_persistent_ws_kernel,
    mqa_logits_fp8_persistent_ws_kernel,
    ref_mqa_logits,
)


def prepare_deepgemm_data(config: MQALogitsConfig, dtype: str):
    config.validate()
    torch.manual_seed(config.seed)
    q = torch.randn(config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = generate_ks_ke(config)

    if dtype == "fp8":
        q_in = q.to(torch.float8_e4m3fn).contiguous()
        kv_in = deep_gemm.utils.per_custom_dims_cast_to_fp8(kv, (0,), False)
        q_sim = q_in.to(torch.bfloat16)
        kv_sim = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
        return {
            "q": q_sim,
            "kv": kv_sim,
            "q_in": q_in,
            "kv_in": kv_in,
            "weights": weights,
            "ks": ks,
            "ke": ke,
        }

    if dtype != "fp4":
        raise ValueError(f"unsupported dtype: {dtype}")
    if config.logits_dtype != "float32":
        raise ValueError("the FP4 SOTA kernel currently stores float32 logits")
    if config.seq_len_kv % 256 != 0:
        raise ValueError("seq_len_kv must be divisible by 256 for the FP4 SOTA tile")

    q_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        q.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        kv.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_sim = deep_gemm.utils.cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(
        config.seq_len, config.num_heads, config.head_dim
    )
    kv_sim = deep_gemm.utils.cast_back_from_fp4(kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True).view(
        config.seq_len_kv, config.head_dim
    )
    q_sf_deepgemm = q_fp4[1].view(config.seq_len, config.num_heads).contiguous()
    kv_sf_deepgemm = kv_fp4[1].view(config.seq_len_kv).contiguous()
    q_in_deepgemm = (
        q_fp4[0].view(config.seq_len, config.num_heads, config.head_dim // 2).contiguous(),
        q_sf_deepgemm,
    )
    kv_in_deepgemm = (
        kv_fp4[0].view(config.seq_len_kv, config.head_dim // 2).contiguous(),
        kv_sf_deepgemm,
    )
    q_in = (q_in_deepgemm[0], q_sf_deepgemm.view(torch.uint32))
    kv_in = (kv_in_deepgemm[0], kv_sf_deepgemm.view(torch.uint32))
    return {
        "q": q_sim.to(torch.bfloat16),
        "kv": kv_sim.to(torch.bfloat16),
        "q_in": q_in,
        "kv_in": kv_in,
        "q_in_deepgemm": q_in_deepgemm,
        "kv_in_deepgemm": kv_in_deepgemm,
        "weights": weights,
        "ks": ks,
        "ke": ke,
    }


def make_tilelang_bench(config: MQALogitsConfig, dtype: str, data):
    logits = torch.full(
        (config.seq_len, config.seq_len_kv),
        float("-inf"),
        device=data["weights"].device,
        dtype=_torch_logits_dtype(config.logits_dtype),
    )
    if dtype == "fp8":
        kv_fp8, kv_scale = data["kv_in"]
        q_fp8_2d = data["q_in"].reshape(config.seq_len * config.num_heads, config.head_dim)

        def fn():
            mqa_logits_fp8_persistent_ws_kernel(
                q_fp8_2d,
                kv_fp8,
                kv_scale,
                data["weights"],
                data["ks"],
                data["ke"],
                logits,
                config.seq_len,
                config.seq_len_kv,
                heads=config.num_heads,
                head_dim=config.head_dim,
                logits_stride=config.seq_len_kv,
                compressed_logits=False,
                logits_dtype=_tilelang_logits_dtype(config.logits_dtype),
            )

        return logits, fn

    q_fp4, q_scale = data["q_in"]
    kv_fp4, kv_scale = data["kv_in"]
    q_fp4_2d = q_fp4.reshape(config.seq_len * config.num_heads, config.head_dim // 2)

    def fn():
        mqa_logits_fp4_persistent_ws_kernel(
            q_fp4_2d,
            q_scale.reshape(-1),
            kv_fp4,
            kv_scale.reshape(-1),
            data["weights"],
            data["ks"],
            data["ke"],
            logits,
            config.seq_len,
            config.seq_len_kv,
            heads=config.num_heads,
            head_dim=config.head_dim,
            logits_stride=config.seq_len_kv,
            compressed_logits=False,
            logits_dtype=T.float32,
        )

    return logits, fn


def make_deepgemm_bench(config: MQALogitsConfig, dtype: str, data):
    dg_logits_dtype = _torch_logits_dtype(config.logits_dtype)
    if dtype == "fp8":
        return lambda: deep_gemm.fp8_fp4_mqa_logits(
            q=(data["q_in"], None),
            kv=data["kv_in"],
            weights=data["weights"],
            cu_seq_len_k_start=data["ks"],
            cu_seq_len_k_end=data["ke"],
            clean_logits=True,
            max_seqlen_k=0,
            logits_dtype=dg_logits_dtype,
        )
    return lambda: deep_gemm.fp8_fp4_mqa_logits(
        q=data["q_in_deepgemm"],
        kv=data["kv_in_deepgemm"],
        weights=data["weights"],
        cu_seq_len_k_start=data["ks"],
        cu_seq_len_k_end=data["ke"],
        clean_logits=True,
        max_seqlen_k=0,
        logits_dtype=dg_logits_dtype,
    )


def bench_case(config: MQALogitsConfig, dtype: str, check: bool = True, warmup: int = 20, rep: int = 100) -> None:
    data = prepare_deepgemm_data(config, dtype)
    ref = ref_mqa_logits(data["q"], data["kv"], data["weights"], data["ks"], data["ke"])
    out, tl_fn = make_tilelang_bench(config, dtype, data)
    tl_fn()
    torch.cuda.synchronize()

    observed = out.float().masked_fill(ref == float("-inf"), 0)
    ref_cmp = ref.masked_fill(ref == float("-inf"), 0)
    diff = calc_diff(observed, ref_cmp)
    if check:
        threshold = 2e-3 if dtype == "fp4" else 1e-4
        assert diff < threshold, f"{dtype} diff {diff} >= {threshold}"

    tilelang_ms = do_bench(tl_fn, warmup=warmup, rep=rep)
    deepgemm_ms = do_bench(make_deepgemm_bench(config, dtype, data), warmup=warmup, rep=rep)
    label = f"{dtype} s{config.seq_len}_skv{config.seq_len_kv}_h{config.num_heads}_d{config.head_dim}_{config.logits_dtype}"
    print(
        f"{label}: tilelang={tilelang_ms * 1000:.3f} us diff={diff:.3e} "
        f"deepgemm={deepgemm_ms * 1000:.3f} us ratio={deepgemm_ms / tilelang_ms:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM100 MQA logits SOTA kernels against DeepGEMM.")
    parser.add_argument("--dtype", choices=("fp8", "fp4", "both"), default="both")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seq-len-kv", type=int, default=4096)
    parser.add_argument("--logits-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    config = MQALogitsConfig(
        seq_len=args.seq_len,
        seq_len_kv=args.seq_len_kv,
        logits_dtype=args.logits_dtype,
        seed=args.seed,
    )
    dtypes = ("fp8", "fp4") if args.dtype == "both" else (args.dtype,)
    for dtype in dtypes:
        bench_case(config, dtype, check=not args.no_check, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
