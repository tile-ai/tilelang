---
name: run-tests
description: How to run tests in this repo when debugging a kernel - tiered testing via TK_TEST_LEVEL (level 0 core first, then level 1), pytest with -x and -n based on visible GPU/NPU count
license: MIT
compatibility: opencode
metadata:
  audience: kernel-developers
  workflow: testing
---

## What I do

Run the repo's tests for a kernel under development, in two levels:

- **Level 0 (core)** — `TK_TEST_LEVEL=0`: most frequently used param coverage. Fast; catches compile errors and gross correctness bugs.
- **Level 1 (default)** — `TK_TEST_LEVEL=1` (or unset): regular param coverage.

Only proceed to level 1 after level 0 fully passes.

## How to run

1. Determine the number of visible devices:
   - GPU: count comma-separated entries in `CUDA_VISIBLE_DEVICES`
   - NPU: count comma-separated entries in `ASCEND_RT_VISIBLE_DEVICES`
   - If the relevant variable is unset, fall back to `nvidia-smi --query-gpu=index --format=csv,noheader | wc -l` (or `npu-smi info -l` on NPU)
2. Compute worker count:
   - Correctness tests: `num_workers = 3 * num_gpus` (or `3 * num_npus`). If this causes OOM, reduce the worker count (e.g. `2 * num_gpus`, then `1 * num_gpus`) and rerun.
   - Benchmark runs (`--run-benchmark`): `num_workers = num_gpus` (or `num_npus`) — exactly one worker per device, since sharing a device skews timing results
3. Run level 0:

   ```bash
   TK_TEST_LEVEL=0 pytest tests/<module>/test_<kernel>.py -x -n ${num_workers}
   ```

4. If (and only if) level 0 passes, run level 1:

   ```bash
   pytest tests/<module>/test_<kernel>.py -x -n ${num_workers}
   ```

Example with 4 GPUs visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`):

```bash
TK_TEST_LEVEL=0 pytest tests/quant/test_swiglu_forward.py -x -n 12
pytest tests/quant/test_swiglu_forward.py -x -n 12
```

## Flag reference

- `-x` — always add it: stop on the first failure. The repo's `tests/pytest_xdist_failfast_plugin.py` makes `-x` work correctly under xdist (interrupts all workers, exit code 2).
- `-n <workers>` — parallelize via pytest-xdist. Use `3 * num_gpus/npus` for correctness tests. `tests/conftest.py` binds each xdist worker to one device (round-robin over `CUDA_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES`) and caps per-worker GPU memory, so oversubscribing 3 workers per device is safe for correctness runs. For benchmarks use `-n num_gpus/npus` instead (one worker per device).
- Test files live under `tests/<module>/` mirroring `tile_kernels/<module>/` (e.g. `tests/quant/`, `tests/moe/`, `tests/mhc/`).

## Notes

- `TK_TEST_LEVEL` must be 0, 1 or 2 (`tile_kernels/testing/generator.py:get_test_level`); it defaults to 1. Level 2 (full Cartesian product plus corner cases) is for CI / final validation, not routine debugging.
- Tests marked `benchmark` are skipped unless `--run-benchmark` is passed. When benchmarking, run with `-m benchmark` and `-n num_gpus/npus` (one worker per device), e.g. with 4 GPUs: `pytest tests/quant/test_swiglu_forward.py --run-benchmark -m benchmark -x -n 4`.
- `large_gpu_mem` tests are auto-marked from `tests/gpu_mem_profiles/` and are normally run separately with fewer workers (see `scripts/run_test.sh`); not needed for routine kernel debugging.
- Do not change `CUDA_VISIBLE_DEVICES` or `ASCEND_RT_VISIBLE_DEVICES`.
