"""Autoregressive model benchmark: TTFT (prefill) + TPOT (decode).

Measures three causal-LM models through three backends:

    eager     — plain PyTorch, no compilation
    inductor  — torch.compile(backend="inductor")
    tilelang  — torch.compile(backend="tilelang")

Models:

    meta-llama/Llama-2-7b-hf
    Qwen/Qwen2.5-7B
    google/gemma-7b

Metrics:

    TTFT  (Time To First Token)  — one prefill forward, batch=1,
          seq_len configurable (default 128).  Latency in ms.
    TPOT  (Time Per Output Token) — decode-phase latency with a
          StaticCache pre-populated with ``prefill_len`` slots.
          batch=1, one new token per step.  Latency in ms/tok.

The script downloads models on first run and caches them via the
normal HuggingFace hub mechanism.  All models are loaded in fp16
with ``attn_implementation="sdpa"``.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings

warnings.filterwarnings("ignore")
import argparse
import csv
from pathlib import Path

import torch
import torch._dynamo

import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "gemma-7b": "google/gemma-7b",
}


# ---------------------------------------------------------------------------
# Build runner
# ---------------------------------------------------------------------------


def _build_runner(model, backend: str):
    if backend == "eager":
        return model
    if backend == "inductor":
        torch._dynamo.reset()
        return torch.compile(model, backend="inductor", mode="default")
    if backend == "tilelang":
        torch._dynamo.reset()
        clear_cache()
        backend_config.vm_clone_output = False
        return torch.compile(model, backend="tilelang")
    raise ValueError(f"unknown backend: {backend}")


# ---------------------------------------------------------------------------
# TTFT (prefill) measurement
# ---------------------------------------------------------------------------


def measure_ttft(runner, model, seq_len: int, n_warmup: int = 3, n_bench: int = 10) -> float:
    """Return mean prefill latency (ms) for a single sequence."""
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, seq_len),
        dtype=torch.int64,
        device="cuda",
    )

    with torch.no_grad():
        for _ in range(n_warmup):
            runner(input_ids=input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    with torch.no_grad():
        for _ in range(n_bench):
            torch.cuda.synchronize()
            start.record()
            runner(input_ids=input_ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# TPOT (decode) measurement
# ---------------------------------------------------------------------------


def _build_static_cache(model, prefill_len: int, max_cache_len: int):
    """Allocate a StaticCache and pre-fill the first ``prefill_len``
    slots with random data so the shapes are realistic."""
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    cache.early_initialization(
        batch_size=1,
        num_heads=num_kv,
        head_dim=head_dim,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        if k is None:
            k = getattr(layer, "key_cache", None)
        v = getattr(layer, "values", None)
        if v is None:
            v = getattr(layer, "value_cache", None)
        if k is not None:
            k[:, :, :prefill_len, :].normal_(0, 0.02)
        if v is not None:
            v[:, :, :prefill_len, :].normal_(0, 0.02)
    return cache


def measure_tpot(runner, model, prefill_len: int, n_warmup: int = 5, n_bench: int = 32) -> float:
    """Return mean time-per-output-token (ms/tok) during decode."""
    max_cache = prefill_len + n_bench + 8
    cache = _build_static_cache(model, prefill_len, max_cache)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, 1),
        dtype=torch.int64,
        device="cuda",
    )
    cache_position = torch.tensor(
        [prefill_len],
        dtype=torch.int64,
        device="cuda",
    )

    def _step():
        return runner(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )

    with torch.no_grad():
        for _ in range(n_warmup):
            _step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(n_bench):
            _step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_bench


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "model",
    "metric",
    "seq_len",
    "eager_ms",
    "inductor_ms",
    "tilelang_ms",
    "speedup_vs_eager",
    "speedup_vs_ind",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", default=",".join(MODELS.keys()), help="Comma-separated model keys to benchmark.")
    ap.add_argument("--backends", default="eager,inductor,tilelang", help="Comma-separated backends.")
    ap.add_argument("--prefill-len", type=int, default=128, help="Sequence length for TTFT prefill benchmark.")
    ap.add_argument("--decode-steps", type=int, default=32, help="Number of decode steps for TPOT.")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    ap.add_argument("--csv", type=Path, default=None, help="Append results to this CSV file.")
    args = ap.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"Models:       {model_keys}")
    print(f"Backends:     {backends}")
    print(f"Prefill len:  {args.prefill_len}")
    print(f"Decode steps: {args.decode_steps}")
    print()

    hdr = (
        f"{'model':<14s} {'metric':<6s} {'seq':<5s} "
        f"{'eager(ms)':>10s}  {'ind(ms)':>10s}  {'tl(ms)':>10s}  "
        f"{'TL/Eager':>10s}  {'TL/Ind':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    csv_rows: list[dict] = []

    for model_key in model_keys:
        hf_name = MODELS[model_key]

        metrics_to_run = [
            (
                "TPOT",
                "1",
                measure_tpot,
                dict(
                    prefill_len=args.prefill_len,
                    n_warmup=args.warmup,
                    n_bench=args.decode_steps,
                ),
            ),
        ]

        for metric, seq_label, measure_fn, measure_kwargs in metrics_to_run:
            # Reload model for each metric to avoid state pollution
            # (TTFT modifies internal caches that break TPOT's StaticCache).
            print(f"\n--- Loading {hf_name} ({metric}) ---", flush=True)
            from transformers import AutoModelForCausalLM

            model = (
                AutoModelForCausalLM.from_pretrained(
                    hf_name,
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa",
                )
                .cuda()
                .eval()
            )

            results: dict[str, float] = {}
            for backend in backends:
                print(f"  [{model_key}/{metric}/{backend}] running...", end="", flush=True)
                try:
                    runner = _build_runner(model, backend)
                    ms = measure_fn(runner, model, **measure_kwargs)
                    results[backend] = ms
                    print(f"  {ms:.2f} ms", flush=True)
                except Exception as e:
                    results[backend] = float("nan")
                    msg = str(e).split("\n")[-1][:80]
                    print(f"  FAIL: {msg}", flush=True)

            eager = results.get("eager", float("nan"))
            ind = results.get("inductor", float("nan"))
            tl = results.get("tilelang", float("nan"))

            print(
                f"{model_key:<14s} {metric:<6s} {seq_label:<5s} "
                f"{eager:>10.2f}  {ind:>10.2f}  {tl:>10.2f}  "
                f"{tl / eager:>10.3f}  {tl / ind:>10.3f}"
            )
            csv_rows.append(
                {
                    "model": model_key,
                    "metric": metric,
                    "seq_len": seq_label,
                    "eager_ms": f"{eager:.4f}",
                    "inductor_ms": f"{ind:.4f}",
                    "tilelang_ms": f"{tl:.4f}",
                    "speedup_vs_eager": f"{eager / tl:.4f}",
                    "speedup_vs_ind": f"{ind / tl:.4f}",
                }
            )

            del model
            torch.cuda.empty_cache()

    if args.csv is not None and csv_rows:
        mode = "a" if args.csv.exists() else "w"
        with args.csv.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if mode == "w":
                writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nAppended {len(csv_rows)} rows → {args.csv}")


if __name__ == "__main__":
    main()
