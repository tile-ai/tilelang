"""Verify one BF16 case before optional warm-cache NPU-event timing."""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from examples.ascend.hy4_mla.prefill import compile_prefill
from examples.ascend.hy4_mla.testing import check_outputs, make_inputs, sparse_mla_reference


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--nq", type=int, default=33)
    parser.add_argument("--nk", type=int, default=257)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=1 / 16)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--num-cores", type=int, help="Override the launch core count for an A/B comparison")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_op_timeout_ms(30000)
    inputs = make_inputs(args.nq, args.nk, args.topk)
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    device_cores = int(properties.cube_core_num)
    num_cores = device_cores if args.num_cores is None else args.num_cores
    expected = sparse_mla_reference(*inputs, scale=args.scale)
    device_inputs = tuple(t.npu() for t in inputs)
    kernel = compile_prefill(
        args.nq,
        args.nk,
        args.topk,
        scale=args.scale,
        variable_lengths=True,
        pipeline_stages=not args.no_pipeline,
        num_cores=num_cores,
    )
    actual = kernel(*device_inputs)
    torch.npu.synchronize()
    result = vars(args).copy()
    result["accuracy"] = check_outputs(actual, expected)
    result["device"] = torch.npu.get_device_name()
    result["device_aic"] = device_cores
    result["launch_cores"] = min(args.nq, num_cores)
    result["dqk"] = 576
    result["dv"] = 512
    if args.repeat > 0:
        for _ in range(5):
            kernel(*device_inputs)
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeat):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            kernel(*device_inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        result["warm_cache_ms"] = times
        result["median_ms"] = statistics.median(times)
        indices, _, lengths = inputs[2:]
        slots = int(((torch.arange(args.topk) < lengths[:, None]) & (indices >= 0) & (indices < args.nk)).sum())
        result["valid_slots"] = slots
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
