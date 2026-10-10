"""Reproduce FlashMLA PR #229 with its input generator and msprof harness."""

from functools import partial
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import warnings

warnings.filterwarnings("ignore", message="Permission mismatch.*")

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "correctness", "bench"], default="smoke")
    parser.add_argument("--implementation", choices=["both", "reference", "manual"], default="both")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--nk", type=int, help="Restrict the performance sweep to one KV length")
    parser.add_argument("--detail", action="store_true", help="Also capture msprof pipe statistics")
    parser.add_argument("--case", type=int, help="Run only this zero-based case number")
    args = parser.parse_args()
    sys.path[:0] = [str(args.reference.resolve()), str(args.reference.resolve() / "tests")]
    import lib
    import ref
    import kernelkit as kk
    from examples.ascend.flashmla import manual

    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    torch.set_default_device("npu:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    if args.mode == "bench":
        cases = [
            lib.TestParam(4096, nk, 640, h_q=64, have_attn_sink=True, seed=42) for nk in ([args.nk] if args.nk else [4096, 8192, 32768])
        ]
    elif args.mode == "smoke":
        cases = [lib.TestParam(32, 256, 128, h_q=64, have_attn_sink=True, seed=42)]
    else:
        cases = [
            lib.TestParam(
                nq,
                nk,
                topk,
                h_q=64,
                seed=42,
                have_attn_sink=sink,
                have_topk_length=lengths,
                k_amplifier_portion=0.02 if amplify else 0.0,
                k_amplifier_ratio=256 if amplify else 1,
            )
            for nq, nk, topk in [(1, 128, 128), (62, 95, 128), (213, 1521, 512), (65, 4096, 640)]
            for sink, lengths, amplify in [(False, False, False), (True, True, False), (True, True, True)]
        ]
        cases += [
            lib.TestParam(65, 128, 64, h_q=64, seed=42, is_all_indices_invalid=True),
            lib.TestParam(33, 95, 192, h_q=64, seed=42, have_attn_sink=True, have_topk_length=True),
        ]
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_id, param in enumerate(cases):
        if args.case is not None and case_id != args.case:
            continue
        print(f"CASE {param}", flush=True)
        inputs = lib.generate_testcase(param)
        if param.is_all_indices_invalid:
            inputs.indices[::2].fill_(-1)
        if args.mode == "correctness" and param.have_topk_length:
            inputs.topk_length[0] = 0
        torch.npu.synchronize()
        funcs = {}
        if args.implementation in ("reference", "both"):
            funcs["reference"] = partial(lib.run_flash_mla_sparse_fwd, param, inputs)
        if args.implementation in ("manual", "both"):
            kernel = manual.compile_prefill(
                param.s_q,
                param.s_kv,
                param.topk,
                sink=param.have_attn_sink,
                variable_lengths=param.have_topk_length,
                scale=inputs.sm_scale,
                q_strides=tuple(inputs.q.stride()),
                index_stride=inputs.indices.stride(0),
            )
            source_path = args.output.parent / args.output.stem / f"manual_{param.s_q}_{param.s_kv}_{param.topk}.asc"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(kernel.get_kernel_source())
            print("Compiled cache:", getattr(kernel, "_tilelang_cache_path", None), flush=True)
            sink_arg = inputs.attn_sink if inputs.attn_sink is not None else torch.zeros(64, dtype=torch.float32)
            lengths_arg = inputs.topk_length if inputs.topk_length is not None else torch.full((param.s_q,), param.topk, dtype=torch.int32)
            funcs["manual"] = partial(kernel, inputs.q, inputs.kv.view(param.s_kv, 512), inputs.indices.squeeze(1), sink_arg, lengths_arg)
        outputs = {}
        for name, fn in funcs.items():
            print(f"Launching {name}", flush=True)
            outputs[name] = fn()
            torch.npu.synchronize()
            print(f"Finished {name}", flush=True)
        case = {
            "nq": param.s_q,
            "nk": param.s_kv,
            "topk": param.topk,
            "sink": param.have_attn_sink,
            "lengths": param.have_topk_length,
            "amplified": param.k_amplifier_portion > 0,
            "all_invalid_rows": param.is_all_indices_invalid,
            "q_strides": list(inputs.q.stride()),
            "index_strides": list(inputs.indices.stride()),
            "correctness": {},
        }
        if args.mode != "bench":
            expected_bf16, expected, expected_max, expected_lse = ref.ref_sparse_attn_fwd(param, inputs)
            for name, (out, maximum, lse) in outputs.items():
                checks = [
                    kk.check_is_allclose(
                        name + " out",
                        out.float(),
                        expected,
                        abs_tol=8e-4 if not param.k_amplifier_portion else 1.0,
                        rel_tol=3.01 / 128 if not param.k_amplifier_portion else 1.0,
                        cos_diff_tol=1e-5,
                        dtype_for_cos_diff_calc=torch.float,
                    ),
                    kk.check_is_allclose(name + " max", maximum, expected_max, abs_tol=1e-6, rel_tol=2.01 / 65536),
                    kk.check_is_allclose(name + " lse", lse, expected_lse, abs_tol=1e-5, rel_tol=4.01 / 65536),
                ]
                case["correctness"][name] = all(checks)
                print(name, "max_abs_out", (out.float() - expected).abs().max().item(), flush=True)
            if not all(case["correctness"].values()):
                result["cases"].append(case)
                args.output.write_text(json.dumps(result, indent=2))
                raise AssertionError("Correctness failure")
        elif "reference" in outputs:
            for name in outputs.keys() - {"reference"}:
                for k in range(3):
                    torch.testing.assert_close(
                        outputs[name][k], outputs["reference"][k], rtol=0.03 if k == 0 else 1e-4, atol=8e-4 if k == 0 else 1e-5
                    )
                case["correctness"][name + "_vs_reference"] = True
        if args.mode == "bench":
            flop = lib.count_flop_and_mem_vol(param, inputs).fwd_flop
            samples = {name: [] for name in funcs}
            launches = {name: [] for name in funcs}
            for round_idx in range(args.rounds):
                order = list(funcs) if round_idx % 2 == 0 else list(reversed(funcs))
                for name in order:
                    profile = kk.bench(funcs[name], num_tests=args.repeat, flush_l2=True)
                    needle = "sparse_attn_fwd" if name == "reference" else "main_kernel"
                    latency = profile.get_kernel_time(needle)
                    matching = [ranges for key, ranges in profile.time_ranges.items() if needle in key]
                    count = sum(len(ranges) for ranges in matching)
                    assert count == args.repeat, (name, count, profile.time_ranges.keys())
                    samples[name].append(latency * 1e6)
                    launches[name].append([(end - start) * 1e6 for ranges in matching for start, end in ranges])
                    print(f"{name} round {round_idx}: {latency * 1e6:.3f} us, {flop / latency / 1e12:.2f} TFLOPS", flush=True)
            case["performance"] = {
                name: {
                    "round_mean_us": values,
                    "launch_us": launches[name],
                    "median_us": statistics.median(values),
                    "tflops": flop / statistics.median(values) / 1e6,
                }
                for name, values in samples.items()
            }
            if args.detail:
                from dataclasses import asdict
                from tilelang.profiler import do_bench

                case["pipe_profile"] = {}
                for name, fn in funcs.items():
                    profile = do_bench(fn, backend="msprof_detail", early_stop_baseline=None, _n_warmup=5, _n_repeat=10)
                    case["pipe_profile"][name] = asdict(profile)
                    print(name, profile, flush=True)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
