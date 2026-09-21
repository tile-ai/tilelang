"""Dynamic MoE scale permutation benchmark, adapted from TileLang PR #3267.

Kernel and reference: https://github.com/tile-ai/tilelang/pull/3267
The original three regression shapes are retained without added assumptions.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import tilelang


import tilelang.language as T


@tilelang.jit
def permute_moe_mxfp4_scales(
    in_scales, out_scales, use_full_perm: int, use_quad_shuffle: int, block: int = 256, shape: tuple | None = None
):
    num_experts = T.dynamic("num_experts") if shape is None else shape[0]
    size_n = T.dynamic("size_n") if shape is None else shape[1]
    num_groups = T.dynamic("num_groups") if shape is None else shape[2]
    in_scales: T.Tensor[(num_experts, size_n, num_groups), T.uint8]
    out_scales: T.Tensor[(num_experts, num_groups, size_n), T.uint8]
    total = num_experts * num_groups * size_n

    with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
        for i in T.Parallel(block):
            idx = bx * block + i
            if idx < total:
                e = idx // (num_groups * size_n)
                f = idx - e * num_groups * size_n
                g = f // size_n
                n = f - g * size_n
                if use_quad_shuffle:
                    q = f % 4
                    v = (f - q) + (q % 2) * 2 + q // 2
                else:
                    v = f
                if use_full_perm:
                    p = v % 64
                    u = (v - p) + (p % 8) * 8 + p // 8
                else:
                    p = v % 32
                    u = (v - p) + 2 * (p // 8) + ((p % 8) // 2) * 8 + (p % 8) % 2
                out_scales[e, g, n] = in_scales[e, u % size_n, u // size_n]


def ref_permute(scales: torch.Tensor, size_n: int, use_full_perm: bool, use_quad_shuffle: bool) -> torch.Tensor:
    num_experts, _, num_groups = scales.shape
    f = torch.arange(num_groups * size_n, device=scales.device)
    v = f
    if use_quad_shuffle:
        q = f % 4
        v = (f - q) + (q % 2) * 2 + q // 2

    tile = 64 if use_full_perm else 32
    p = v % tile
    if use_full_perm:
        u = (v - p) + (p % 8) * 8 + p // 8
    else:
        u = (v - p) + 2 * (p // 8) + ((p % 8) // 2) * 8 + (p % 8) % 2

    src = scales.reshape(num_experts, -1)[:, (u % size_n) * num_groups + u // size_n]
    return src.view(num_experts, num_groups, size_n).contiguous()


def _flags(size_k: int, group_size: int, is_a8: bool) -> tuple[bool, bool]:
    use_full_perm = group_size < size_k and group_size != -1 and not is_a8
    use_quad_shuffle = not is_a8
    return use_full_perm, use_quad_shuffle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".cache/launch_plan/permute_scales"))
    parser.add_argument("--key", default="tl.enable_invariant_arithmetic")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    records = []
    cases = [
        ("dsv3_w4a16", 256, 2048, 7168, 32, False),
        ("dsv3_w4a8", 256, 2048, 7168, 32, True),
        ("llama4_w4a16", 8, 4096, 8192, 128, False),
    ]
    for name, e, n, k, g, a8 in cases:
        ng = k // g
        full, quad = _flags(k, g, a8)
        func = permute_moe_mxfp4_scales.get_tir(use_full_perm=int(full), use_quad_shuffle=int(quad))
        torch.manual_seed(42)
        x = torch.randint(0, 256, (e, n, ng), device="cuda", dtype=torch.uint8)
        y = torch.empty((e, ng, n), device="cuda", dtype=torch.uint8)
        expected = ref_permute(x, n, full, quad)
        variants = []
        for mode in ["off", "on", "const"]:
            enabled = mode == "on"
            selected_func = (
                permute_moe_mxfp4_scales.get_tir(use_full_perm=int(full), use_quad_shuffle=int(quad), shape=(e, n, ng))
                if mode == "const"
                else func
            )
            print(name, mode, "compiling", flush=True)
            t = time.perf_counter()
            kernel = tilelang.compile(
                selected_func,
                target="cuda",
                target_host="c",
                execution_backend="tvm_ffi",
                pass_configs={"tl.disable_warp_specialized": True, "tl.enable_fast_math": True, args.key: enabled},
            )
            elapsed = time.perf_counter() - t
            src = kernel.get_kernel_source()
            (root / f"{name}_{mode}.cu").write_text(src)
            (root / f"{name}_{mode}.host").write_text(kernel.get_host_source())

            def call(kernel=kernel, x=x, y=y):
                kernel(x, y)

            call()
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.count):
                    call()
            y.copy_(expected ^ 255)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            variants.append((mode, call, graph, elapsed, len(src)))
        samples = {mode: [] for mode, *_ in variants}
        rng = random.Random(3267)
        for _, _, graph, _, _ in variants:
            for _ in range(5):
                graph.replay()
        torch.cuda.synchronize()
        for _ in range(args.rounds):
            order = list(variants)
            rng.shuffle(order)
            for mode, call, graph, _, _ in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    graph.replay()
                end.record()
                end.synchronize()
                gpu = start.elapsed_time(end) * 1000 / (5 * args.count)
                t = time.perf_counter_ns()
                for _ in range(args.count):
                    call()
                torch.cuda.synchronize()
                wall = (time.perf_counter_ns() - t) / (1000 * args.count)
                samples[mode].append([gpu, wall])
        for mode, _, _, compile_s, source_bytes in variants:
            row = dict(
                case=name,
                mode=mode,
                compile_s=compile_s,
                source_bytes=source_bytes,
                gpu_us=statistics.median(x[0] for x in samples[mode]),
                wall_us=statistics.median(x[1] for x in samples[mode]),
                samples=samples[mode],
            )
            records.append(row)
            print(row, flush=True)
        (root / "results.json").write_text(
            json.dumps(
                dict(
                    sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    device=torch.cuda.get_device_name(),
                    rounds=args.rounds,
                    count=args.count,
                    results=records,
                ),
                indent=2,
            )
        )
        del variants, call, graph, kernel, x, y, expected
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
