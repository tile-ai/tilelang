# FP32 CUDA clamp validation

Run these scripts from a source build on an otherwise idle NVIDIA GPU. They
write raw samples and compiler/runtime metadata beside the results. CUDA events measure
device-stream elapsed time; compilation, correctness checks, and host transfers
are outside the timed region.
Both scripts record `cuda_math_sha256`, the hash of the CUDA math header used
for compilation, so results identify the helper even in an uncommitted checkout.

## NVCC helper comparison

```bash
python benchmark/clamp/benchmark_nvcc.py --output /tmp/clamp-nvcc
```

This compiles the current `tl::clamp` and a frozen copy of the old helper in the
same translation unit, with default and fast math. Each mode checks 10,648
operand triples with constant and dynamic bounds. Non-NaN results must match
bit for bit; NaN classification must match, while NaN payload/sign may differ.

The workloads are one dependent FMA+clamp chain, eight independent chains, and
fused affine+clamp at 1M and 16M elements. The chains use four blocks per SM,
256 threads, and 512 iterations. Fifteen samples alternate old/new ordering;
each sample contains five host-issued launches, so short kernels can include
launch gaps in the event interval. Outputs are checked outside timing. The
TileLang script below uses graph replay to reduce those gaps.

## TileLang NVCC/NVRTC and issue #3515

Run the same script in separate baseline and candidate source builds, copying
this benchmark directory to the baseline checkout if needed. Choose `tvm_ffi`
for NVCC or `nvrtc` for NVRTC. Disable the persistent kernel cache so a header
change cannot reuse a previously compiled kernel.

```bash
TILELANG_DISABLE_CACHE=1 python benchmark/clamp/benchmark_tilelang.py \
  --execution-backend nvrtc --no-fast-math --output /tmp/clamp-before
TILELANG_DISABLE_CACHE=1 python benchmark/clamp/benchmark_tilelang.py \
  --execution-backend nvrtc --no-fast-math --output /tmp/clamp-after \
  --compare /tmp/clamp-before
```

Run the first command on the baseline and the second on the candidate, with
the same GPU, toolchain, and backend. The output directories must be distinct;
the script rejects paths that resolve to the same directory before writing any
files or running GPU work, including relative paths and symlink aliases.
`--compare` checks the recorded GPU
capability, Python, PyTorch, and PyTorch CUDA build versions as well as the
workload settings. Keep the driver and compiler toolkit fixed too. Repeat with `--fast-math` and with
`--execution-backend tvm_ffi`. The script saves 42,592 semantic outputs across
constant/dynamic bounds and default/fast math; `--compare` verifies non-NaN bits
and NaN classification against the corresponding baseline.

The timed kernel retains the dependent-bound workload from issue #3515:
65,536 elements, 128 iterations, three clamps per iteration. Nine samples
alternate clamp/minmax order, each using a CUDA Graph of 100 launches. Capture
uses an explicit stream, NVRTC launches receive that stream, and compiled
kernels remain alive through all replays. The
finite-input outputs must match exactly, and a NaN probe verifies their
intentionally different NaN behavior. Min/max is a performance control, not a
semantically equivalent replacement. Baseline/candidate runs are sequential;
inspect raw samples and repeat if clocks or competing workloads change.

Speedups depend on clamp density, dependency chains, memory traffic, register
pressure, GPU, and compiler. A short pilot or a chain microbenchmark does not
predict an application's speedup. NVCC and NVRTC may apply fast-math FTZ options
differently; compare each backend against itself instead of treating one
backend's subnormal behavior as the other backend's oracle.
