"""Check and benchmark the fused K-pool Hadamard paths on CUDA or ROCm.

Run from the repository root with ``python -m examples.kpool.benchmark_glm53_kpool_hadamard``.
Optionally pass ``--baseline-root /path/to/checkout`` to compare example sources
using the same installed TileLang compiler. Kernel timing uses graph replay;
wrapper timing includes metadata validation, launch, and device completion.
Peak temporary bytes exclude allocations already live before the wrapper call.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path
import sys

import torch

from examples.kpool import example_glm53_kpool_compress as compress
from examples.kpool import example_glm53_kpool_decode_tail as decode
from tilelang.language.fp8 import determine_torch_fp8_type
from tilelang.profiler import do_bench


def load_baseline(root: Path):
    modules = []
    previous_compress = sys.modules[compress.__name__]
    try:
        for name in ("compress", "decode_tail"):
            path = root / "examples" / "kpool" / f"example_glm53_kpool_{name}.py"
            spec = importlib.util.spec_from_file_location(f"kpool_baseline_{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
            if name == "compress":
                sys.modules[compress.__name__] = module
    finally:
        sys.modules[compress.__name__] = previous_compress
    return tuple(modules)


def check_cache(actual, expected, *, exact=False):
    """Check the existing finite, one-FP8-ULP and separate FP32 scale contract."""
    actual_k, actual_scale = actual
    expected_k, expected_scale = expected
    if exact:
        if not torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8)) or not torch.equal(actual_scale, expected_scale):
            raise RuntimeError("baseline/candidate cache mismatch")
        return
    if not torch.isfinite(actual_k.float()).all() or not torch.isfinite(expected_k.float()).all():
        raise RuntimeError("non-finite FP8 cache value")
    codes = []
    for values in (actual_k, expected_k):
        bits = values.contiguous().view(torch.uint8).to(torch.int16)
        magnitude = bits & 0x7F
        codes.append(torch.where((bits & 0x80) != 0, 0x80 - magnitude, 0x80 + magnitude))
    distance = (codes[0] - codes[1]).abs()
    if torch.any(distance > 1):
        raise RuntimeError(f"FP8 distance exceeds one ULP: {int(distance.max().item())}")
    if not torch.allclose(actual_scale, expected_scale, rtol=2e-3, atol=1e-6):
        raise RuntimeError("FP32 scale mismatch")


def make_cache(num_slots):
    shape = (max(1, (num_slots + 63) // 64), 64)
    dtype = determine_torch_fp8_type()
    return (
        torch.full((*shape, 128), 1.0, dtype=dtype, device="cuda"),
        torch.full(shape, -7.0, dtype=torch.float32, device="cuda"),
    )


def prepare_compress(modules, batch, round_scale, *, basis=False):
    torch.manual_seed(0)
    key = torch.randn(batch, 4, 128, device="cuda", dtype=torch.bfloat16)
    score = torch.randn_like(key)
    ape = torch.randn(4, 128, device="cuda")
    if basis:
        key.copy_(torch.eye(128, device="cuda", dtype=torch.bfloat16)[:, None, :])
        score.zero_()
        ape.zero_()
    loc = torch.randperm(batch, device="cuda", dtype=torch.int64)
    mask = torch.arange(batch, device="cuda") % 7 != 6
    loc[~mask] = -123
    arms = {}
    expected = make_cache(batch + 1)
    reference_k, reference_scale = compress.glm53_kpool_reference(key, score, ape, round_scale=round_scale)
    expected[0].view(-1, 128)[loc[mask]] = reference_k[mask]
    expected[1].view(-1)[loc[mask]] = reference_scale[mask]
    for name, (module, _) in modules.items():
        cache = make_cache(batch + 1)
        kernel = module.glm53_kpool_compress_kernel(round_scale=round_scale)
        launch = partial(kernel, key, score, ape, loc, mask, *cache)
        wrapper = partial(
            module.glm53_kpool_compress_and_write_cache, key, score, ape, loc, *cache, write_mask=mask, round_scale=round_scale
        )
        wrapper()
        check_cache(cache, expected)
        untouched = torch.ones(cache[1].numel(), device="cuda", dtype=torch.bool)
        untouched[loc[mask]] = False
        check_cache(
            (cache[0].view(-1, 128)[untouched], cache[1].view(-1)[untouched]),
            (expected[0].view(-1, 128)[untouched], expected[1].view(-1)[untouched]),
            exact=True,
        )
        arms[name] = (launch, wrapper, cache)
    return arms


def prepare_decode(modules, batch, next_n, round_scale, *, reference=False):
    torch.manual_seed(1)
    key = torch.randn(batch, next_n, 128, device="cuda", dtype=torch.bfloat16)
    score = torch.randn_like(key)
    ape = torch.randn(4, 128, device="cuda")
    # Complete pools make repeated benchmark launches idempotent. For plain
    # decode, only offset 3 changes and the other three tail slots stay fixed.
    start = 3 if next_n == 1 else 0
    positions = torch.arange(start, start + next_n, device="cuda", dtype=torch.int32).expand(batch, -1).contiguous()
    blocks = torch.randperm(batch, device="cuda", dtype=torch.int32)
    slots = blocks[:, None] * 4 + positions % 4
    loc = torch.full_like(positions, -1)
    closes = positions % 4 == 3
    loc[closes] = torch.randperm(int(closes.sum().item()), device="cuda", dtype=torch.int32)
    # Retain a padded row in correctness scenarios; benchmark all requests.
    if reference and batch > 1:
        positions[-1] = -1
        slots[-1] = -1
        loc[-1] = -1
    initial_tail = torch.randn(max(1, batch), 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    expected_tail = initial_tail.clone()
    expected = make_cache(batch * next_n + 1)
    if reference:
        decode.glm53_kpool_decode_tail_reference(expected_tail, slots, key, score, ape, loc, positions, *expected, round_scale=round_scale)
    arms = {}
    previous_tail = None
    for name, (_, module) in modules.items():
        tail = initial_tail.clone()
        cache = make_cache(batch * next_n + 1)
        kernel = module.glm53_kpool_decode_tail_kernel(next_n=next_n, round_scale=round_scale)
        args = (tail, slots, key, score, ape, loc, positions, *cache)
        launch = partial(kernel, *args)
        wrapper = partial(module.glm53_kpool_decode_update_and_maybe_write_cache, *args, round_scale=round_scale)
        wrapper()
        if reference:
            if not torch.equal(tail, expected_tail):
                raise RuntimeError("decode tail differs from sequential reference")
            check_cache(cache, expected)
            untouched = torch.ones(cache[1].numel(), device="cuda", dtype=torch.bool)
            untouched[loc[loc >= 0]] = False
            check_cache(
                (cache[0].view(-1, 128)[untouched], cache[1].view(-1)[untouched]),
                (expected[0].view(-1, 128)[untouched], expected[1].view(-1)[untouched]),
                exact=True,
            )
        if previous_tail is not None and not torch.equal(tail, previous_tail):
            raise RuntimeError("baseline/candidate decode tail mismatch")
        previous_tail = tail
        arms[name] = (launch, wrapper, cache)
    return arms


def check_decode_rejections(modules, round_scale):
    arms = prepare_decode(modules, 2, 4, round_scale)
    for name, (_, wrapper, _) in arms.items():
        values = wrapper.args
        tail_slots = values[0].shape[0] * 4
        cache_slots = values[7].shape[0] * values[7].shape[1]
        cases = (
            (((6, (0, 0), -1),), "positions and tail_slot_mapping must use matching negative padding"),
            (
                ((6, (1, slice(None)), -1), (1, (1, slice(None)), -1)),
                "padded tokens must use a negative cache_loc",
            ),
            (
                ((6, (0, 0), -1), (1, (0, 0), -1)),
                "valid decode tokens must form a prefix in each request row",
            ),
            (((6, (0, 1), 8),), "valid positions must be consecutive within each request"),
            (((1, (0, 0), tail_slots),), f"active tail slots must be in [0, {tail_slots})"),
            (((1, (0, 0), values[1][0, 0] + 1),), "tail slot phase must equal position modulo pool_size"),
            (((1, (0, 0), values[1][1, 0]),), "all tokens for one request must use the same tail block"),
            (((1, (1, slice(None)), values[1][0]),), "active requests must use distinct tail blocks"),
            (((5, (0, 3), -1),), "cache_loc must be nonnegative exactly when a valid token closes a pool"),
            (((5, (0, 3), cache_slots),), f"active cache locations must be in [0, {cache_slots})"),
            (((5, (1, 3), values[5][0, 3]),), "active cache locations must be unique to avoid concurrent writes"),
        )
        before_tail = values[0].clone()
        before_cache = tuple(value.clone() for value in values[7:])
        for edits, expected_error in cases:
            changed = list(values)
            for index in (1, 5, 6):
                changed[index] = values[index].clone()
            for index, coordinate, replacement in edits:
                changed[index][coordinate] = replacement
            try:
                wrapper.func(*changed, round_scale=round_scale)
            except ValueError as error:
                if str(error) != expected_error:
                    raise RuntimeError(f"{name}: {error!s} != {expected_error}") from error
            else:
                raise RuntimeError(f"{name}: accepted invalid decode metadata: {expected_error}")
            if not torch.equal(values[0], before_tail):
                raise RuntimeError(f"{name}: rejected metadata changed the tail cache")
            check_cache(values[7:], before_cache, exact=True)
        wrapper()
    compare_arms(arms)


def compare_arms(arms):
    values = list(arms.values())
    for arm in values[1:]:
        check_cache(arm[2], values[0][2], exact=True)


def measure(arms, case, rounds, rep):
    compare_arms(arms)
    names = list(arms)
    for index in range(rounds):
        for name in names[:: 1 if index % 2 == 0 else -1]:
            launch, wrapper, _ = arms[name]
            kernel_ms = do_bench(launch, backend="cudagraph", rep=rep, return_mode="median")
            wrapper_ms = do_bench(wrapper, backend="wall", device="cuda", warmup=5, rep=rep, return_mode="median")
            torch.cuda.synchronize()
            allocated_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            wrapper()
            torch.cuda.synchronize()
            peak_temporary_bytes = torch.cuda.max_memory_allocated() - allocated_before
            print(
                json.dumps(
                    {
                        **case,
                        "arm": name,
                        "round": index,
                        "kernel_us": kernel_ms * 1000,
                        "wrapper_us": wrapper_ms * 1000,
                        "peak_temporary_bytes": peak_temporary_bytes,
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256, 2048])
    parser.add_argument("--operations", choices=("compress", "decode"), nargs="+", default=["compress", "decode"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep", type=float, default=50, help="milliseconds per timing sample")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches) or args.rounds <= 0 or args.rep <= 0:
        parser.error("batch sizes, rounds, and rep must be positive")
    modules = {}
    if args.baseline_root:
        modules["baseline"] = load_baseline(args.baseline_root)
    modules["candidate"] = (compress, decode)
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip}
        )
    )
    for round_scale in (False, True):
        check_decode_rejections(modules, round_scale)
        compare_arms(prepare_compress(modules, 7, round_scale))
        compare_arms(prepare_compress(modules, 128, round_scale, basis=True))
        for next_n in (1, 4, 8):
            compare_arms(prepare_decode(modules, 3, next_n, round_scale, reference=True))
    print(json.dumps({"checks": "passed", "round_scale": [False, True], "decode_next_n": [1, 4, 8]}), flush=True)
    if args.check_only:
        return
    for round_scale in (False, True):
        for batch in args.batches:
            if "compress" in args.operations:
                case = {"operation": "compress", "batch": batch, "round_scale": round_scale}
                measure(prepare_compress(modules, batch, round_scale), case, args.rounds, args.rep)
            if "decode" in args.operations:
                for next_n in (1, 8):
                    case = {"operation": "decode", "batch": batch, "next_n": next_n, "round_scale": round_scale}
                    measure(prepare_decode(modules, batch, next_n, round_scale), case, args.rounds, args.rep)


if __name__ == "__main__":
    main()
