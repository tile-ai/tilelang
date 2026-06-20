# Metal GEMM

TileLang provides two Metal GEMM paths targeting Apple Silicon GPUs.

## Overview

| Path | `T.gemm` instruction | micro tile | hardware | status |
|------|---------------------|------------|----------|--------|
| simdgroup | `metal.simdgroup` | 8 × 8 × 8 | M1–M5 | stable |
| cooperative tensor | `metal.cooperative_tensor` | 16 × 32 × 16 | M5+ | experimental |

Both paths are available through the same `T.gemm(A_shared, B_shared, C)` call. The compiler
automatically selects the cooperative tensor path when the tile shape and C scope permit;
otherwise it falls back to the simdgroup path.

## Cooperative Tensor Path

On M5+ devices the cooperative tensor path uses Apple's **Metal 4 MPP**
(`mpp::tensor_ops::matmul2d`) to access the Neural Accelerator hardware. This can
deliver significantly higher throughput than the 8×8 simdgroup path.

### Selection rules

TileLang picks the cooperative tensor path when **all** of the following hold:

- `C` is placed in **shared memory** (not `local.fragment` / `metal.simdgroup`).
- M % 16 == 0, N % 32 == 0, K % 16 == 0 (16×32×16 micro tile).
- The number of warps can be evenly partitioned into `M/16 × N/32` tile groups.

If any condition fails the compiler falls back to `metal.simdgroup` without user action.

### Current limitations

- **Shared-C only.** The `local.fragment` C path always uses simdgroup today. Direct
  fragment-C cooperative tensor still needs the fragment accumulator storage remap to
  become `metal.cooperative_tensor`; otherwise the lowered fragment becomes vectorized
  local storage that cannot be used as an MPP destination.
- **float32 accumulation only in MPP path.** MPP matmul2d loads `half`/`bfloat` inputs
  but the destination `cooperative_tensor` is always `float32`.
- **No software pipelining (`num_stages=0`).** Cooperative tensor shared-memory loads
  are emitted inside the inner loop; the pipeline planner does not yet interleave them.
- **No transpose flags.** The lowering assumes `trans_A=False, trans_B=False`.

### Benchmarking

Use the Metal matmul benchmark to compare TileLang against PyTorch MPS and MLX:

```bash
python benchmark/matmul_metal/benchmark_matmul_metal.py --m 4096 --n 4096 --k 4096 --warmup 3 --repeats 10 --sweep
```

The benchmark includes both legacy `metal.simdgroup` and shared-C
`metal.cooperative_tensor` configurations. MLX is reported when `mlx` is installed.

## Future Work

- Support direct fragment-C cooperative tensor by remapping eligible fragment
  accumulators to `metal.cooperative_tensor` storage.
- Enable software pipelining for cooperative tensor loads.
- Extend to transposed operand layouts.
- Add direct global-to-global GEMM (GG) lowering through MPP.
- Performance tuning: broader block-shape search, shared-memory usage reduction,
  and fragment-C persistent accumulators.
