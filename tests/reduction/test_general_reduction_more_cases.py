"""Multi-case GeneralReduction coverage/performance suite.

This script expands GeneralReduction testing beyond the current single
`sum->epilogue` and softmax-like tests. It covers multiple graph families to
evaluate:
1. schedule applicability (generality),
2. lowering/compilation stability,
3. numerical correctness against PyTorch,
4. runtime and speedup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Callable
from collections.abc import Iterable, Sequence

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import s_tir, te, tirx as tir


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

from tilelang.schedule.gpu.general_reduction import GeneralReduction
from tilelang.schedule.gpu.layernorm_like import LayerNormLike


def _shape_to_tuple(shape: Sequence[tir.PrimExpr]) -> tuple[int, ...]:
    result = []
    for dim in shape:
        if isinstance(dim, tir.IntImm):
            result.append(int(dim.value))
        else:
            raise ValueError(f"Dynamic shape dim is unsupported in this suite: {dim}")
    return tuple(result)


@dataclass
class CaseBuild:
    inputs: list[te.Tensor]
    output: te.Tensor
    ref_fn: Callable[..., torch.Tensor]
    description: str


@dataclass
class CaseResult:
    name: str
    status: str
    tilelang_ms: float | None
    torch_ms: float | None
    speedup: float | None
    note: str


def _build_sum_relu_3d(args: argparse.Namespace) -> CaseBuild:
    m, n, k = args.m, args.n, args.k
    a = te.placeholder((m, n, k), name="a")
    rk = te.reduce_axis((0, k), name="rk")
    red = te.compute((m, n), lambda i, j: te.sum(a[i, j, rk], axis=rk), name="red")
    out = te.compute((m, n), lambda i, j: te.max(red[i, j], tir.const(0.0, "float32")), name="out")

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sum(a_t, dim=2))

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="sum-reduction with ReLU epilogue (3D -> 2D)",
    )


def _build_max_bias_relu_3d(args: argparse.Namespace) -> CaseBuild:
    m, n, k = args.m, args.n, args.k
    a = te.placeholder((m, n, k), name="a")
    bias = te.placeholder((m, n), name="bias")
    rk = te.reduce_axis((0, k), name="rk")
    red = te.compute((m, n), lambda i, j: te.max(a[i, j, rk], axis=rk), name="red")
    out = te.compute(
        (m, n),
        lambda i, j: te.max(red[i, j] + bias[i, j], tir.const(0.0, "float32")),
        name="out",
    )

    @torch.compile
    def ref_fn(a_t: torch.Tensor, bias_t: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.max(a_t, dim=2).values + bias_t)

    return CaseBuild(
        inputs=[a, bias],
        output=out,
        ref_fn=ref_fn,
        description="max-reduction with bias+ReLU epilogue (multi-input epilogue)",
    )


def _build_keepdim_sum_sigmoid_3d(args: argparse.Namespace) -> CaseBuild:
    m, n, k = args.m, args.n, args.k
    a = te.placeholder((m, n, k), name="a")
    rk = te.reduce_axis((0, k), name="rk")
    red = te.compute((m, n, 1), lambda i, j, _: te.sum(a[i, j, rk], axis=rk), name="red")
    one = tir.const(1.0, "float32")
    out = te.compute(
        (m, n, 1),
        lambda i, j, t: one / (one + te.exp(-red[i, j, t])),
        name="out",
    )

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.sum(a_t, dim=2, keepdim=True))

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="keepdim reduction + sigmoid epilogue (tests keepdim pattern)",
    )


def _build_rmsnorm_like_2d(args: argparse.Namespace) -> CaseBuild:
    m, k = args.m2d, args.k2d
    eps = tir.const(args.eps, "float32")
    inv_k = tir.const(1.0 / float(k), "float32")

    a = te.placeholder((m, k), name="a")
    rk = te.reduce_axis((0, k), name="rk")
    sqsum = te.compute((m,), lambda i: te.sum(a[i, rk] * a[i, rk], axis=rk), name="sqsum")
    inv_rms = te.compute((m,), lambda i: tir.const(1.0, "float32") / te.sqrt(sqsum[i] * inv_k + eps), name="inv_rms")
    out = te.compute((m, k), lambda i, j: a[i, j] * inv_rms[i], name="out")

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(torch.mean(a_t * a_t, dim=1, keepdim=True) + args.eps)
        return a_t * inv

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="RMSNorm-like single reduction + broadcast epilogue",
    )


def _build_layernorm_like_2d(args: argparse.Namespace) -> CaseBuild:
    m, k = args.m2d, args.k2d
    eps = tir.const(args.eps, "float32")
    inv_k = tir.const(1.0 / float(k), "float32")

    a = te.placeholder((m, k), name="a")
    rk0 = te.reduce_axis((0, k), name="rk0")
    mean_sum = te.compute((m,), lambda i: te.sum(a[i, rk0], axis=rk0), name="mean_sum")
    mean = te.compute((m,), lambda i: mean_sum[i] * inv_k, name="mean")
    center = te.compute((m, k), lambda i, j: a[i, j] - mean[i], name="center")
    rk1 = te.reduce_axis((0, k), name="rk1")
    var = te.compute((m,), lambda i: te.sum(center[i, rk1] * center[i, rk1], axis=rk1), name="var")
    inv_std = te.compute((m,), lambda i: tir.const(1.0, "float32") / te.sqrt(var[i] * inv_k + eps), name="inv_std")
    out = te.compute((m, k), lambda i, j: center[i, j] * inv_std[i], name="out")

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        mean_t = torch.mean(a_t, dim=1, keepdim=True)
        center_t = a_t - mean_t
        var_t = torch.mean(center_t * center_t, dim=1, keepdim=True)
        return center_t * torch.rsqrt(var_t + args.eps)

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="LayerNorm-like two-reduction chain",
    )


def _build_softmax_like_2d(args: argparse.Namespace) -> CaseBuild:
    m, k = args.m2d, args.k2d
    a = te.placeholder((m, k), name="a")
    rk0 = te.reduce_axis((0, k), name="rk0")
    rmax = te.compute((m,), lambda i: te.max(a[i, rk0], axis=rk0), name="rmax")
    expv = te.compute((m, k), lambda i, j: te.exp(a[i, j] - rmax[i]), name="expv")
    rk1 = te.reduce_axis((0, k), name="rk1")
    rsum = te.compute((m,), lambda i: te.sum(expv[i, rk1], axis=rk1), name="rsum")
    out = te.compute((m, k), lambda i, j: expv[i, j] / rsum[i], name="out")

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(a_t, dim=1)

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="Softmax-like two-reduction chain (max + sum(exp))",
    )


def _build_logsumexp_like_2d(args: argparse.Namespace) -> CaseBuild:
    m, k = args.m2d, args.k2d
    a = te.placeholder((m, k), name="a")
    rk0 = te.reduce_axis((0, k), name="rk0")
    rmax = te.compute((m,), lambda i: te.max(a[i, rk0], axis=rk0), name="rmax")
    expv = te.compute((m, k), lambda i, j: te.exp(a[i, j] - rmax[i]), name="expv")
    rk1 = te.reduce_axis((0, k), name="rk1")
    rsum = te.compute((m,), lambda i: te.sum(expv[i, rk1], axis=rk1), name="rsum")
    out = te.compute((m,), lambda i: te.log(rsum[i]) + rmax[i], name="out")

    @torch.compile
    def ref_fn(a_t: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(a_t, dim=1)

    return CaseBuild(
        inputs=[a],
        output=out,
        ref_fn=ref_fn,
        description="LogSumExp-like two-reduction chain with scalar output",
    )


CASE_BUILDERS: dict[str, Callable[[argparse.Namespace], CaseBuild]] = {
    "sum_relu_3d": _build_sum_relu_3d,
    "max_bias_relu_3d": _build_max_bias_relu_3d,
    "keepdim_sum_sigmoid_3d": _build_keepdim_sum_sigmoid_3d,
    "rmsnorm_like_2d": _build_rmsnorm_like_2d,
    "layernorm_like_2d": _build_layernorm_like_2d,
    "softmax_like_2d": _build_softmax_like_2d,
    "logsumexp_like_2d": _build_logsumexp_like_2d,
}


def _select_cases(case_arg: str) -> list[str]:
    if case_arg == "all":
        return list(CASE_BUILDERS.keys())
    selected = [item.strip() for item in case_arg.split(",") if item.strip()]
    unknown = [name for name in selected if name not in CASE_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Available: {list(CASE_BUILDERS.keys())}")
    return selected


def _lower_module(mod: tvm.IRModule) -> tvm.IRModule:
    mod = tir.transform.Simplify()(mod)
    mod = s_tir.transform.LowerInitBlock()(mod)
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)
    return mod


def _run_case(case_name: str, args: argparse.Namespace) -> CaseResult:
    try:
        build = CASE_BUILDERS[case_name](args)
        func = te.create_prim_func([*build.inputs, build.output])
        rule = LayerNormLike() if case_name == "layernorm_like_2d" else GeneralReduction()
        sch = rule.apply(func, tvm.target.Target({"kind": "cuda", "arch": args.arch}), False)
        if sch is None:
            return CaseResult(
                name=case_name,
                status="schedule_none",
                tilelang_ms=None,
                torch_ms=None,
                speedup=None,
                note=f"{type(rule).__name__}.apply returned None",
            )

        mod = _lower_module(sch.mod)
        if args.print_ir:
            print(f"\n=== Case: {case_name} / Scheduled IR ===")
            print(sch.mod)
            print(f"\n=== Case: {case_name} / Lowered IR ===")
            print(mod)

        if args.check_only:
            return CaseResult(
                name=case_name,
                status="lowered",
                tilelang_ms=None,
                torch_ms=None,
                speedup=None,
                note=build.description,
            )

        kernel = tilelang.compile(mod["main"], pass_configs=PASS_CONFIGS)

        input_shapes = [_shape_to_tuple(inp.shape) for inp in build.inputs]
        out_shape = _shape_to_tuple(build.output.shape)

        torch.manual_seed(args.seed)
        inputs_torch = [torch.randn(shape, device="cuda", dtype=torch.float32) for shape in input_shapes]
        out_torch = torch.empty(out_shape, device="cuda", dtype=torch.float32)

        kernel(*inputs_torch, out_torch)

        ref_fn = torch.compile(build.ref_fn)
        ref_out = ref_fn(*inputs_torch)
        torch.testing.assert_close(out_torch, ref_out, rtol=args.rtol, atol=args.atol)

        bench_note = build.description
        try:
            tilelang_ms = do_bench(
                lambda: kernel(*inputs_torch, out_torch),
                backend=args.bench_backend,
            )
            torch_ms = do_bench(
                lambda: ref_fn(*inputs_torch),
                backend=args.bench_backend,
            )
        except Exception as bench_err:  # pylint: disable=broad-except
            if "Unknown profiler backend" not in str(bench_err):
                raise
            tilelang_ms = do_bench(lambda: kernel(*inputs_torch, out_torch))
            torch_ms = do_bench(lambda: ref_fn(*inputs_torch))
            bench_note = f"{build.description}; fallback benchmark backend=default (requested={args.bench_backend})"

        return CaseResult(
            name=case_name,
            status="passed",
            tilelang_ms=float(tilelang_ms),
            torch_ms=float(torch_ms),
            speedup=float(torch_ms / tilelang_ms) if tilelang_ms > 0 else None,
            note=bench_note,
        )
    except Exception as err:  # pylint: disable=broad-except
        return CaseResult(
            name=case_name,
            status="failed",
            tilelang_ms=None,
            torch_ms=None,
            speedup=None,
            note=str(err),
        )


def _print_summary(results: Iterable[CaseResult], check_only: bool) -> None:
    print("\n=== GeneralReduction Case Summary ===")
    if check_only:
        print(f"{'case':28} {'status':14} note")
        print("-" * 110)
        for r in results:
            print(f"{r.name:28} {r.status:14} {r.note}")
    else:
        print(f"{'case':28} {'status':10} {'tilelang(ms)':>12} {'torch(ms)':>10} {'speedup':>10} note")
        print("-" * 130)
        for r in results:
            t_ms = "-" if r.tilelang_ms is None else f"{r.tilelang_ms:.3f}"
            p_ms = "-" if r.torch_ms is None else f"{r.torch_ms:.3f}"
            spd = "-" if r.speedup is None else f"{r.speedup:.3f}x"
            print(f"{r.name:28} {r.status:10} {t_ms:>12} {p_ms:>10} {spd:>10} {r.note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple GeneralReduction workloads to evaluate generality and performance.")
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated case names, or 'all'.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument("--m", type=int, default=1024, help="3D workloads: first dim.")
    parser.add_argument("--n", type=int, default=16, help="3D workloads: second dim.")
    parser.add_argument("--k", type=int, default=16384, help="3D workloads: reduction dim.")
    parser.add_argument("--m2d", type=int, default=8192, help="2D workloads: first dim.")
    parser.add_argument("--k2d", type=int, default=16384, help="2D workloads: reduction/second dim.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for norm-like workloads.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for torch input generation.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for correctness check.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for correctness check.")
    parser.add_argument(
        "--bench-backend",
        type=str,
        default="event",
        choices=["event", "cupti"],
        help="Profiler backend used by tilelang.profiler.do_bench.",
    )
    parser.add_argument("--check-only", action="store_true", help="Only run schedule/lowering checks.")
    parser.add_argument("--print-ir", action="store_true", help="Print scheduled and lowered IR per case.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any selected case is not successful.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = _select_cases(args.cases)

    if not args.check_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for correctness/performance execution.")

    print("Selected cases:", ", ".join(selected_cases))
    results: list[CaseResult] = []
    for case_name in selected_cases:
        print(f"\n[Case] {case_name}")
        result = _run_case(case_name, args)
        print(f"  status: {result.status}")
        if result.tilelang_ms is not None and result.torch_ms is not None:
            print(f"  tilelang: {result.tilelang_ms:.4f} ms")
            print(f"  torch   : {result.torch_ms:.4f} ms")
            if result.speedup is not None:
                print(f"  speedup : {result.speedup:.4f}x")
        if result.note:
            print(f"  note    : {result.note}")
        results.append(result)

    _print_summary(results, check_only=args.check_only)

    if args.strict:
        ok_status = {"lowered"} if args.check_only else {"passed"}
        failures = [r for r in results if r.status not in ok_status]
        if failures:
            raise RuntimeError("Strict mode failed. Non-successful cases: " + ", ".join(f"{r.name}({r.status})" for r in failures))


if __name__ == "__main__":
    main()
