# TileLang/PTO Correctness and Performance Gates

## Implementation Order

1. Prefer deriving from a similar operator implementation in the current repository. Before using bundled code, check [Template Maturity](template_status.md).
2. Locate TileLang through its actual import path and verify that every API exists in PTO lowering. Do not use CUDA-only schedules, targets, or pass configurations.
3. Establish an fp32 or higher-precision PyTorch reference and implement the complete shape/dtype fallback first.
4. Run targeted correctness tests and fix every numerical, out-of-bounds, compilation, and unimplemented-path failure.
5. Change only one variable at a time among tile, core count, stage, residency, execution domain, and fusion.
6. After every change, rerun targeted correctness tests and then benchmark in a fixed environment.
7. After representative shapes run without fallback, execute the relevant complete test suite; continue with project-required expanded tests when the interface requires them.

## Correctness Matrix

Read the public-interface contract first, then cover minimum/common/maximum shapes, permitted remainder classes, dynamic dimensions, noncontiguous strides, every dtype, forward/backward, zero, positive and negative extremes, cancellation, NaN/Inf, and empty tasks. When the interface promises general tail-block support, add tile-1/tile/tile+1. When the interface permits only specific alignments, do not treat an unimplemented fallback as an existing capability. Reduction, Norm, Softmax, Scan carry, and GEMM L0C use fp32 by default; convert output only at the final boundary.

Record the test collection count, pass count, first failure, shape, dtype, tolerance, exception category, and root cause. `NotImplementedError` means unimplemented, not passed. Do not relax tolerances, lower reference precision, skip cases, or report only a previously passing subset.

## Performance Matrix

Keep the input distribution, shape, dtype, output semantics, warmup, repeat, device, frequency, and concurrency identical. Report kernel latency, end-to-end latency, effective GM bytes/bandwidth, Cube/Vector/MTE time, core utilization, UB/L1/L0/workspace bytes, stage count, and generated-code size.

For multiple kernels and host pipelines, include every launch, event, workspace, and synchronization operation. Without measured data from the current device, report only a design, not a performance gain.

## Commands

```bash
TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
TILELANG_DEFAULT_TARGET=pto pytest <test_file>
```

Bundled-template regression:

```bash
python scripts/validate_templates.py
TILELANG_DEFAULT_TARGET=pto python scripts/validate_templates.py --npu --num-cores <available_aiv_cores>
```

Run correctness tests concurrently according to the number of available devices only after confirming the device-binding and isolation mechanisms; otherwise, run serially. On OOM, reduce concurrency without dropping cases. Benchmarks require exclusive device access.
