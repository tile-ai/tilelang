"""Tune a tiled elementwise kernel with the TileIR backend.

The search is deliberately staged:

1. Tune an ordinary TileLang tile-size parameter with TileIR hints unset.
2. Tune the TileIR entry, load/store, and semantic compiler knobs one factor
   at a time for the best tile size.

Every candidate is checked against the same PyTorch reference before its
latency can be selected.

The candidate values keep this example short. They are illustrative, not a
recommended search space for every kernel or GPU.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.tileir.checks import check_tileir_available
from tilelang.transform import PassConfigKey


def tiled_divide_factory(n: int):
    """Return a kernel exposing every TileIR hint relevant to this copy-heavy op."""

    def kernel(
        block_size: int = 256,
        num_ctas: int | None = None,
        occupancy: int | None = None,
        num_worker_warps: int | None = None,
        copy_latency: int | None = None,
        disable_tma: bool = False,
    ):
        @T.prim_func
        def tiled_divide(
            A: T.Tensor((n,), "float32"),
            C: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(n, block_size),
                threads=256,
                num_ctas=num_ctas,
                occupancy=occupancy,
                num_worker_warps=num_worker_warps,
            ) as bx:
                shared = T.alloc_shared((block_size,), "float32")
                fragment = T.alloc_fragment((block_size,), "float32")
                T.copy(
                    A[bx * block_size],
                    shared,
                    latency=copy_latency,
                    disable_tma=disable_tma,
                )
                T.copy(shared, fragment)
                for i in T.Parallel(block_size):
                    fragment[i] = fragment[i] / 3.0
                T.copy(
                    fragment,
                    C[bx * block_size],
                    latency=copy_latency,
                    disable_tma=disable_tma,
                )

        return tiled_divide

    return kernel


def target_for_current_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA CUDA GPU.")

    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("The TileIR backend requires a GPU with compute capability 9.0 or newer.")
    return f"tileir -arch=sm_{major}{minor}"


def tileir_tuning_configs(block_size: int) -> list[dict[str, Any]]:
    """Return a small, illustrative one-factor-at-a-time tuning space.

    `tileir_hints` is not a separate search dimension. It is the
    per-architecture packaging form of the three entry hints below and is most
    useful after tuning, when selected values are shipped for multiple SMs.
    Real kernels should choose candidates from their algorithm and target GPU.
    """

    baseline: dict[str, Any] = {
        "block_size": block_size,
        "num_ctas": None,
        "occupancy": None,
        "num_worker_warps": None,
        "copy_latency": None,
        "disable_tma": False,
    }

    def with_changes(**changes: Any) -> dict[str, Any]:
        return {**baseline, **changes}

    return [
        baseline,
        # A bounded subset of entry optimization_hints on T.Kernel.
        with_changes(num_ctas=1),
        with_changes(num_ctas=2),
        with_changes(num_ctas=4),
        with_changes(occupancy=1),
        with_changes(occupancy=2),
        with_changes(occupancy=4),
        with_changes(num_worker_warps=4),
        with_changes(num_worker_warps=8),
        # A bounded subset of load/store hints on global-memory T.copy calls.
        with_changes(copy_latency=1),
        with_changes(copy_latency=4),
        with_changes(copy_latency=8),
        with_changes(disable_tma=True),
        # Per-config compiler directive. Fast math changes numerical semantics,
        # so include it only when the operation's tolerance permits it.
        with_changes(pass_configs={PassConfigKey.TL_ENABLE_FAST_MATH: True}),
        # Keep TL_TILEIR_OPT_LEVEL at its default 3. Lower levels are useful for
        # compiler diagnosis, not a normal performance search. The deprecated
        # global TL_DISABLE_TMA_LOWER switch is also intentionally omitted in
        # favor of the per-copy disable_tma candidate above.
    ]


def make_tuner(
    n: int,
    configs: list[dict[str, Any]],
    target: str,
    a: torch.Tensor,
) -> AutoTuner:
    return (
        AutoTuner.from_kernel(tiled_divide_factory(n), configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target=target,
            execution_backend="tileir",
        )
        .set_profile_args(
            supply_prog=lambda _: [a],
            ref_prog=lambda value: value / 3.0,
            rtol=1e-3,
            atol=1e-5,
            skip_check=False,
            cache_input_tensors=False,
        )
    )


def main(n: int = 1 << 20, warmup: int = 5, rep: int = 20, timeout: int = 60) -> None:
    if n % 1024 != 0:
        raise ValueError("n must be a multiple of 1024 so every trial has a cluster-aligned grid.")

    check_tileir_available()
    target = target_for_current_device()

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    reference = a / 3.0

    baseline_config = {"block_size": 256}
    baseline_kernel = tilelang.compile(
        tiled_divide_factory(n)(**baseline_config),
        out_idx=[-1],
        target=target,
        execution_backend="tileir",
    )
    torch.testing.assert_close(baseline_kernel(a), reference, rtol=1e-5, atol=1e-5)
    baseline_latency = baseline_kernel.get_profiler().do_bench(
        input_tensors=[a],
        n_warmup=warmup,
        n_repeat=rep,
    )

    algorithm_configs = [{"block_size": 128}, {"block_size": 256}]
    print(f"Stage 1 TileLang candidates (TileIR hints unset): {algorithm_configs}")
    algorithm_result = make_tuner(n, algorithm_configs, target, a).run(
        warmup=warmup,
        rep=rep,
        timeout=timeout,
    )
    best_block_size = algorithm_result.config["block_size"]

    tileir_configs = tileir_tuning_configs(best_block_size)
    print(f"Stage 2 TileIR candidates: {tileir_configs}")
    best_result = make_tuner(n, tileir_configs, target, a).run(
        warmup=warmup,
        rep=rep,
        timeout=timeout,
    )

    tuned_output = best_result.kernel(a)
    torch.testing.assert_close(tuned_output, reference, rtol=1e-3, atol=1e-5)

    print("\nTileIR autotuning complete")
    print(f"Target: {target}")
    print(f"Baseline config: {baseline_config}")
    print(f"Baseline latency: {baseline_latency:.6f} ms")
    print(f"Best config: {best_result.config}")
    print(f"Best latency: {best_result.latency:.6f} ms")
    print(f"Speedup over baseline: {baseline_latency / best_result.latency:.3f}x")
    if best_result.ref_latency is not None:
        print(f"PyTorch reference latency: {best_result.ref_latency:.6f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1 << 20, help="Number of elements; must be a multiple of 1024")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per candidate")
    parser.add_argument("--rep", type=int, default=20, help="Timed iterations per candidate")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds per candidate")
    args = parser.parse_args()
    main(n=args.n, warmup=args.warmup, rep=args.rep, timeout=args.timeout)
