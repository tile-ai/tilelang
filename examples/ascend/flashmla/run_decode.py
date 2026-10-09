"""Same-input comparison against FlashMLA PR #229, including packed KV caches."""

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
    parser.add_argument("--output", type=Path, default=Path("decode-results.json"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--case", type=int, help="Run only this zero-based case number")
    args = parser.parse_args()
    sys.path[:0] = [str(args.reference.resolve()), str(args.reference.resolve() / "tests")]
    import lib
    import ref
    import quant
    import flash_mla
    import kernelkit as kk
    from examples.ascend.flashmla.decode import prepare_decode

    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    torch.set_default_device("npu:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    fp8 = quant.KVCacheLayout.V41_FP8Sparse
    fp4 = quant.KVCacheLayout.V41_FP4Sparse
    raw = lib.RawTestParamForDecode
    if args.mode == "bench":
        cases = [
            raw(
                b,
                64,
                4,
                1,
                256,
                True,
                128,
                extra_s_k=2048,
                extra_topk=512,
                block_size=256,
                extra_block_size=64,
                kvcache_layout=fp8,
                extra_kvcache_layout=fmt,
                seed=42,
            )
            for fmt in [fp8, fp4]
            for b in [64, 128, 256, 512]
        ]
        cases += [
            raw(
                1,
                64,
                4096,
                1,
                4096,
                False,
                128,
                extra_s_k=4096,
                extra_topk=512,
                block_size=64,
                extra_block_size=64,
                kvcache_layout=fp8,
                extra_kvcache_layout=fmt,
                seed=42,
            )
            for fmt in [fp8, fp4]
        ]
    else:
        cases = [raw(4, 64, 1, 1, 256, True, 128, kvcache_layout=fp8, seed=42)]
        cases += [
            raw(4, 64, 3, 1, 256, True, 128, extra_s_k=512, extra_topk=128, kvcache_layout=fp8, extra_kvcache_layout=fmt, seed=42)
            for fmt in [fp8, fp4]
        ]
        if args.mode == "correctness":
            cases += [
                raw(
                    b,
                    64,
                    3,
                    1,
                    650,
                    True,
                    576,
                    extra_s_k=512,
                    extra_topk=64,
                    block_size=53,
                    extra_block_size=61,
                    have_topk_length=lengths,
                    have_extra_topk_length=lengths,
                    have_zero_seqlen_k=True,
                    enable_attn_sink=sink,
                    is_all_indices_invalid=invalid,
                    kvcache_layout=fp8,
                    extra_kvcache_layout=fmt,
                    seed=42,
                )
                for fmt in [fp8, fp4]
                for b, lengths, sink, invalid in [(1, False, False, True), (74, True, True, False), (35, False, False, False)]
            ]
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_id, rp in enumerate(cases):
        if args.case is not None and case_id != args.case:
            continue
        p = rp.to_test_param()
        print(f"CASE {case_id}: {p}", flush=True)
        inputs = lib.generate_testcase_for_decode(p)
        torch.npu.synchronize()
        metadata, _ = flash_mla.get_mla_metadata()
        funcs = {"reference": partial(lib.run_flash_mla_decode, p, inputs, metadata, None)}
        kernel, funcs["manual"] = prepare_decode(p, inputs)
        source_path = args.output.parent / args.output.stem / f"manual_decode_{case_id}.asc"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(kernel.get_kernel_source())
        print("Compiled cache:", getattr(kernel, "_tilelang_cache_path", None), flush=True)
        outputs = {name: fn() for name, fn in funcs.items()}
        torch.npu.synchronize()
        expected = ref.ref_sparse_attn_decode(p, inputs) if args.mode != "bench" else outputs["reference"]
        case = {
            "id": case_id,
            "b": rp.b,
            "sq": rp.s_q,
            "topk": rp.topk,
            "extra_topk": rp.extra_topk,
            "extra_format": "fp4" if rp.extra_kvcache_layout == fp4 else "fp8",
            "correctness": {},
        }
        for name, (out, lse) in outputs.items():
            checks = [
                kk.check_is_allclose(name + " out", out, expected[0], abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6),
                kk.check_is_allclose(name + " lse", lse, expected[1], abs_tol=1e-6, rel_tol=8.01 / 65536),
            ]
            case["correctness"][name] = all(checks)
            print(
                name,
                "out error",
                (out - expected[0]).abs().max().item(),
                "lse error",
                (lse - expected[1]).abs().nan_to_num().max().item(),
                flush=True,
            )
        assert all(case["correctness"].values()), case
        if args.mode == "bench":
            flop = lib.count_flop_and_mem_vol_for_decode(p, inputs).flop
            samples = {name: [] for name in funcs}
            launches = {name: [] for name in funcs}
            for round_idx in range(args.rounds):
                for name in list(funcs) if round_idx % 2 == 0 else list(reversed(funcs)):
                    profile = kk.bench(funcs[name], num_tests=args.repeat, flush_l2=True)
                    needle = "sparse_attn_fwd" if name == "reference" else "main_kernel"
                    latency = profile.get_kernel_time(needle)
                    matching = [ranges for key, ranges in profile.time_ranges.items() if needle in key]
                    assert sum(len(ranges) for ranges in matching) == args.repeat
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
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2))
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
