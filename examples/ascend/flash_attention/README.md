# FlashAttention (Ascend NPU)

Online-softmax FlashAttention forward and GQA backward kernels. MHA and GQA
forward share the `flash_attention_fwd` builder in `core.py`. MHA is a special
case with `q_len == kv_len` and one query tile per core. GQA flattens query groups
into `q_len = S1 * G` and distributes the work across cores using `num_blocks`.

GQA backward consumes the optional LSE output from forward and retains the
fastest current path, using `T.Stage` with AutoSchedule. First,
`flash_attention_bwd_preprocess` computes `Delta = sum(O * dO, axis=-1)` for all
query rows. A frontend-staged, KV-centric mixed kernel then performs five GEMMs
in one launch, accumulates dK/dV privately, and reduces dQ into GM using bfloat16
atomic adds. Fusing Delta into the kernel partitioned by KV tile would repeat
the `O * dO` reduction 64 times for the full shape, so both launches remain part
of this implementation.

```
core.py          # Shared flash_attention_fwd builder + FwdTiling
core_bwd.py      # Delta and T.Stage AutoSchedule fused builders
example_mha.py   # MHA wrapper + reference + benchmark
example_gqa.py   # GQA wrapper + reference + benchmark
example_gqa_bwd.py
example_gqa_manual_schedule.py  # GQA with a validated fixed-stage schedule
test_mha.py / test_gqa.py / test_gqa_bwd.py
```

## Limitations

MHA/GQA forward share the same SIMD softmax-packing implementation. Backward
packing also uses a fixed 128x128 tile:

- **`head_dim` is fixed at 128.** Softmax writes probabilities directly in NZ
  layout, using a 128-column `vsstb` stride and `uint16x128` merging. This path
  hardcodes D=128, enforced by `assert head_dim == 128` in the builder. Other
  values of D are unsupported until the implementation is generalized.
- **Tile shapes are constrained.** `block_q` must be even, and
  `block_kv == 2 * VL == 128` (64 fp32 SIMD lanes). Both
  `q_len % block_q == 0` and `kv_len % block_kv == 0` must hold.
- **Inputs must use bfloat16**, with float32 accumulation. The output dtype is
  configurable: MHA writes float32 directly without a cast; GQA writes bfloat16
  through an additional `vcvt` cast into a byte alias of `O_ub`, requiring no
  extra UB storage.
- **No causal mask.** Dropout, attention bias, variable lengths, and padding
  masks are unsupported.
- **Backward currently supports only the flattened GQA layout.** Inputs, `dO`,
  and output gradients use bfloat16. dQ uses BF16 atomic adds across KV tiles.
  Tiles are fixed at 128x128. There is no MHA backward wrapper yet, although its
  mathematical formulation is equivalent to `G=1`.
- **Staged backward requires at least three query tiles:** `q_len / 128 >= 3`.
  Four Q/dO buffer versions do not increase the minimum input size.
- **GQA `num_blocks` must divide `q_len / block_q`**, the number of M tiles,
  to avoid an uneven workload across cores.

## Performance (current shapes, Ascend 950 / dav-3510, bf16)

| Kernel | Shape | TileLang | Torch SDPA | Comparison |
|---|---|---|---|---|
| MHA | `SEQ_LEN=4096, D=128` | 26.8 us · 320.1 TFLOPS | 28.94 us · 296.8 TFLOPS | **1.08x speedup with TileLang** |
| GQA | `S1=8192, G=32, S2=8192, D=128` | 3030 us · 362.8 TFLOPS | 2864 us · 383.9 TFLOPS | 1.06x speedup with Torch |
| GQA backward (T.Stage fused) | `S1=8192, G=32, S2=8192, D=128` | **6.418 ms · 428.3 effective TFLOPS** | — | See the comparison below, measured in the same environment |
| GQA (manual) | Same as above | 3125 us · 351.8 TFLOPS | — | 92.9% of automatic scheduling throughput in the same measurement round |

Forward relative error is below 0.5%. For the full backward shape, the T.Stage
fused BF16 outputs have relative L2 errors of approximately
1.024% / 0.316% / 0.037% for dQ/dK/dV against Torch BF16 gradients. The larger dQ
error comes from repeated BF16 rounding during atomic accumulation across
64 KV tiles.

Backward measurements were collected on 2026-09-24 using Ascend950DT,
CANN/Bisheng 9.2.0, and a local build of TileLang `8121f415` with fast-math
enabled. The baseline and optimized versions used identical inputs and
preallocated outputs over five rounds, alternating A/B and B/A order. Each
round measured cold-L2 `msprof_detail` FFTS kernel durations with 5 warmups and
20 repetitions. The table reports the median and range of the five round means.
Timing includes **dQ zeroing + fused backward**, but excludes forward, Delta,
and cache flush. FLOPs are calculated as `10 * Q * K * D` for the five GEMMs.

| Backward implementation | Median | Range over five rounds | Effective TFLOPS |
|---|---:|---:|---:|
| Baseline (`4347e52e`) | 6834.581 us | 6830.059–6837.689 us | 402.19 |
| UnitFlag + FixPipe dQ + buffer reallocation | 6418.273 us | 6417.254–6418.779 us | 428.27 |

Latency decreased by 6.09%, and throughput increased by 6.49%. All ten raw
profiles were verified to contain 20 fused AIC executions, 20 fused AIV
executions, 20 zeroing operations, and 20 cache flushes. Mixed-kernel duration
was counted only once. Zeroing took approximately 15.4 us; the fused kernel
itself decreased from approximately 6819.2 us to 6402.9 us. The earlier
430.6 TFLOPS result was not reproduced in this environment. All comparisons in
the table use the same toolchain; the difference is not attributed to any
particular compiler commit.

The optimization changes L0C handoffs, the dQ data path, and buffer allocation
together. Adding only UnitFlag took approximately 6.753 ms. Using FixPipe for
dQ with the original three buffer versions took approximately 6.755 ms. Reaching
6.418 ms required four Q/dO buffer versions and two intermediate buffer versions
as well. AIC MAD active increased from approximately 96.86% to 99.80%. With
UnitFlag, FixPipe active includes time spent waiting for data readiness, so a
value near 100% does not indicate exhausted bandwidth.

The earlier cannsim RVEC estimates for `SimdVF` latency are retained:
1053 cycles for Delta and 706 cycles for P/dS packing. The optimized version
removes the dQ Vector cast.

### T.Stage AutoSchedule fused backward implementation snapshot

The entry point is `core_bwd.py::flash_attention_bwd_fused_dq_atomic_staged`.
The grid is partitioned by KV tile. Each AIC exclusively owns a 128x128 K/V tile
and performs the following operations across 2048 query tiles:

| Order | GEMM | Result and ownership |
|---:|---|---|
| 1 | `K @ Q.T` | Scores, split between two AIVs through FixPipe |
| 2 | `V @ dO.T` | dP, split between two AIVs through FixPipe |
| 3 | `P @ dO` | dV, accumulated in FP32 in the KV core's L0C |
| 4 | `dS @ Q` | dK, accumulated in FP32 in the KV core's L0C |
| 5 | `dS.T @ K` | dQ contribution, converted to BF16 and atomically added to GM |

`T.Stage(0)` produces scores/dP and packs P/dS. `T.Stage(1)` consumes P/dS from
the previous query tile. Q/dO in L1 uses four buffer versions; P/dS in L1,
scores/dP in UB, and LSE/Delta in UB use two. L0A/L0B retain two slots, alternating
between GEMMs, while P/dS packing scratch retains one slot. Total L1 usage
remains 448 KiB, and UB usage decreases from 243.5 KiB to 162.5 KiB.

The score, dP, and dQ GEMMs and their output copies each specify
`unit_flag_ctrl=3` as a pair to protect temporary L0C readiness and reuse. dQ
still borrows the drained `dp_l0c`. FixPipe converts it directly to BF16 and
atomically writes it back to GM, eliminating L0C-to-UB transfer, the Vector cast,
and UB-to-GM transfer. Each partial gradient is converted to BF16 before
accumulation. The caller must still zero dQ, and atomic reduction does not
guarantee bitwise determinism.

The kernel specifies buffer version counts. AutoSchedule generates version
indices, pipeline expansion, and the remaining local and cross-core
synchronization. A variant without `T.Stage` passed correctness checks but took
approximately 6.897 ms, so the stage constraints are retained.

This PrimFunc keeps AutoSchedule and shared-memory reuse enabled by default.
Only fast-math needs to be specified:

```python
{
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
```

## Running the examples

Running a forward or backward example directly checks correctness and reports
latency and TFLOPS. Backward Delta preprocessing reports its latency and
effective bandwidth separately, calculated as logical input and output bytes
divided by elapsed time. Use `--no-perf` for GQA backward or `--no-benchmark`
for manually scheduled GQA to run correctness checks only.

```bash
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_mha.py
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa.py
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa_bwd.py
# Check GQA backward correctness on small inputs only
ASCEND_NPU_ARCH=dav-3510 python example_gqa_bwd.py --no-perf
# Fixed stages for 13 tasks; runs the full benchmark shape by default
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa_manual_schedule.py
# Or run the correctness tests
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python -m pytest test_mha.py test_gqa.py test_gqa_bwd.py -v
```
