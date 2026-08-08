"""Compare the TileLang SageAttention3 core against the official CUDA core.

This follows the local-reference pattern used by Mayfly's nunchaku alignment
harness: preprocess and quantize one deterministic input, run both native
implementations on those exact packed tensors, and report machine-readable
error metrics. Only ``Out`` is compared because both raw ABIs reserve ``LSE``
without materializing it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

import tilelang

from examples.sage_attention_sm120.sageattn3_fp4 import sage3_packed_fp4_attention_raw_kernel


_DEFAULT_CASES = (
    (128, 128, 128, 1),
    (128, 256, 192, 1),
    (256, 256, 256, 1),
    (128, 128, 128, 2),
)


def _parse_case(text: str) -> tuple[int, int, int, int]:
    try:
        case = tuple(int(value) for value in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid case {text!r}: expected integers") from err
    if len(case) != 4:
        raise argparse.ArgumentTypeError(f"invalid case {text!r}: expected query_tokens,kv_tokens,valid_k_tokens,heads")
    query_tokens, kv_tokens, valid_k_tokens, heads = case
    if query_tokens <= 0 or kv_tokens <= 0 or valid_k_tokens <= 0 or heads <= 0:
        raise argparse.ArgumentTypeError(f"invalid non-positive case {text!r}")
    if query_tokens % 128 or kv_tokens % 128:
        raise argparse.ArgumentTypeError("query_tokens and kv_tokens must be multiples of 128")
    if valid_k_tokens > kv_tokens:
        raise argparse.ArgumentTypeError("valid_k_tokens must not exceed kv_tokens")
    return case


def _import_official_extensions(root: Path):
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    search_paths = [root]
    for module_name in ("fp4attn_cuda", "fp4quant_cuda"):
        matches = list(root.rglob(f"{module_name}*.pyd")) + list(root.rglob(f"{module_name}*.so"))
        search_paths.extend(path.parent for path in matches)
    for path in dict.fromkeys(search_paths):
        sys.path.insert(0, str(path))

    return importlib.import_module("fp4attn_cuda"), importlib.import_module("fp4quant_cuda")


def _pad_tokens(x: torch.Tensor, tokens: int) -> torch.Tensor:
    if x.shape[-2] > tokens:
        raise ValueError(f"cannot pad {x.shape[-2]} tokens to the smaller extent {tokens}")
    return F.pad(x, (0, 0, 0, tokens - x.shape[-2])).contiguous()


def _preprocess(
    query_tokens: int,
    kv_tokens: int,
    valid_k_tokens: int,
    heads: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (1, heads, query_tokens, 128)
    shape_kv = (1, heads, valid_k_tokens, 128)
    q = torch.randn(shape_q, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(shape_kv, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(shape_kv, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    # Match sageattn3.api.preprocess_qkv: K is centered before padding, while Q
    # is centered independently in each 128-token block after padding.
    k = k - k.mean(dim=-2, keepdim=True)
    k = _pad_tokens(k, kv_tokens)
    v = _pad_tokens(v, kv_tokens)
    q_blocks = q.reshape(1, heads, query_tokens // 128, 128, 128)
    q_mean = q_blocks.mean(dim=3)
    q = (q_blocks - q_mean.unsqueeze(3)).reshape(shape_q).contiguous()
    delta_s = torch.matmul(q_mean, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def _quantize(x: torch.Tensor, quant_module, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, rows, cols = x.shape
    if kind == "normal":
        packed = torch.empty((batch, heads, rows, cols // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((batch, heads, rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
        quant_module.scaled_fp4_quant(x, packed, scales, 1)
    elif kind == "permute":
        packed = torch.empty((batch, heads, rows, cols // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((batch, heads, rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
        quant_module.scaled_fp4_quant_permute(x, packed, scales, 1)
    elif kind == "transpose":
        packed = torch.empty((batch, heads, cols, rows // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((batch, heads, cols, rows // 16), device=x.device, dtype=torch.float8_e4m3fn)
        quant_module.scaled_fp4_quant_trans(x, packed, scales, 1)
    else:
        raise ValueError(f"unknown quantization kind: {kind}")
    return packed, scales


def _metrics(actual: torch.Tensor, reference: torch.Tensor, max_abs_tolerance: float, min_cosine: float) -> dict[str, Any]:
    actual_f32 = actual.to(torch.float32)
    reference_f32 = reference.to(torch.float32)
    error = (actual_f32 - reference_f32).abs()
    cosine = float(F.cosine_similarity(actual_f32.flatten(), reference_f32.flatten(), dim=0).item())
    max_abs = float(error.max().item())
    mean_abs = float(error.mean().item())
    finite = bool(torch.isfinite(actual_f32).all().item())
    return {
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "cosine_similarity": cosine,
        "max_abs_tolerance": max_abs_tolerance,
        "min_cosine_similarity": min_cosine,
        "finite": finite,
        "pass": finite and max_abs <= max_abs_tolerance and cosine >= min_cosine,
    }


def _run_case(
    case: tuple[int, int, int, int],
    *,
    seed: int,
    arch: str,
    max_abs_tolerance: float,
    min_cosine: float,
    fp4attn_cuda,
    fp4quant_cuda,
) -> dict[str, Any]:
    query_tokens, kv_tokens, valid_k_tokens, heads = case
    q, k, v, delta_s = _preprocess(query_tokens, kv_tokens, valid_k_tokens, heads, seed)
    q_packed, q_scales = _quantize(q, fp4quant_cuda, "normal")
    k_packed, k_scales = _quantize(k, fp4quant_cuda, "permute")
    vt_packed, vt_scales = _quantize(v, fp4quant_cuda, "transpose")

    softmax_scale = 128**-0.5
    reference = fp4attn_cuda.fwd(
        q_packed,
        k_packed,
        vt_packed,
        q_scales,
        k_scales,
        vt_scales,
        delta_s,
        valid_k_tokens,
        None,
        softmax_scale,
        False,
        True,
        True,
    )[0]

    factory = sage3_packed_fp4_attention_raw_kernel
    old_mode = factory.func.mode
    try:
        factory.func.mode = "lazy"
        program = factory.func(query_tokens, kv_tokens, valid_k_tokens, heads, heads, 128)
    finally:
        factory.func.mode = old_mode
    kernel = tilelang.compile(
        program,
        out_idx=[7, 8],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": arch},
    )
    actual, _ = kernel(q_packed, k_packed, vt_packed, q_scales, k_scales, vt_scales, delta_s)
    torch.cuda.synchronize()

    correctness = _metrics(actual, reference, max_abs_tolerance, min_cosine)
    return {
        "case": {
            "query_tokens": query_tokens,
            "kv_tokens": kv_tokens,
            "valid_k_tokens": valid_k_tokens,
            "heads": heads,
            "head_dim": 128,
        },
        "status": "ok" if correctness["pass"] else "fail",
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sageattention-root",
        type=Path,
        required=True,
        help="Directory containing the built fp4attn_cuda and fp4quant_cuda extensions.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="query_tokens,kv_tokens,valid_k_tokens,heads; repeat for multiple cases.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", default="sm_120a")
    # Multi-tile traversal mismatches are about 1e-2 in BF16 output, while the
    # aligned kernels differ by at most one BF16 ULP in the exercised cases.
    parser.add_argument("--max-abs", type=float, default=0.001)
    parser.add_argument("--min-cosine", type=float, default=0.9995)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.getLogger("tilelang").setLevel(logging.WARNING)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    fp4attn_cuda, fp4quant_cuda = _import_official_extensions(args.sageattention_root)
    tilelang.disable_cache()

    cases = tuple(args.case) if args.case else _DEFAULT_CASES
    results = [
        _run_case(
            case,
            seed=args.seed,
            arch=args.arch,
            max_abs_tolerance=args.max_abs,
            min_cosine=args.min_cosine,
            fp4attn_cuda=fp4attn_cuda,
            fp4quant_cuda=fp4quant_cuda,
        )
        for case in cases
    ]
    payload = {
        "target": "sage3_sm120_cuda_alignment",
        "reference": "SageAttention3 fp4attn_cuda.fwd on identical official CUDA-quantized tensors",
        "lse_compared": False,
        "status": "ok" if all(result["status"] == "ok" for result in results) else "fail",
        "results": results,
    }
    print(json.dumps(payload, sort_keys=True) if args.json else json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
