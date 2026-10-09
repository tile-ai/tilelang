# Measurement report

Measured on 2026-10-09 with one Ascend950DT_9581 (32 AIC + 64 AIV), physical
device 5, CANN 9.2.0, Bisheng clang 15.0.5 (2026-08-27), PyTorch 2.13.0+cpu,
and torch_npu 2.13.0rc1. TileLang base: `a35f8ddf45eba16c21211ec8822d56ce5363036f`.
FlashMLA reference: `a123d0b0191e0da7aa1e044f8644e0989cc63220`.
The unchanged reference extension and manual kernel ran on the same NPU.

## Performance

Each entry is the median of three round means, ten launches per round, using
the PR's msprof harness and 8 GB L2 flush. A/B order alternates by round.
Speedup is reference latency divided by manual latency. Small differences
near 1.00x should be interpreted together with the recorded round variation.
These are attention-kernel times; input reshapes, allocations and other kernels
are excluded. Decode applies the same per-call flattening as the PR wrapper,
so copies of strided inputs affect the initial cache state in both variants.

All prefill cases use nq=4096, H=64, D=512, topk=640 and attention sinks.

| KV length | Reference μs | Manual μs | Reference TFLOPS | Manual TFLOPS | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 858.04 | 848.67 | 400.44 | 404.86 | 1.011x |
| 8192 | 853.78 | 851.15 | 402.44 | 403.68 | 1.003x |
| 32768 | 862.15 | 862.05 | 398.54 | 398.58 | 1.000x |

Decode uses FP8 main KV, topk=128, extra topk=512, H=64, D=512 and attention
sinks. The first eight cases have sq=4, variable cache sequence lengths,
main sequence length 256 and extra sequence length 2048. The last two are the
PR's shared-cache prefill-as-decode cases, with b=1, sq=4096 and both sequence
lengths 4096. They have no topk-length tensors.

| Extra cache | Batch | Queries/batch | Reference μs | Manual μs | Reference TFLOPS | Manual TFLOPS | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 64 | 4 | 78.44 | 76.70 | 273.78 | 279.99 | 1.023x |
| FP8 | 128 | 4 | 144.46 | 142.34 | 297.32 | 301.75 | 1.015x |
| FP8 | 256 | 4 | 274.83 | 271.30 | 312.56 | 316.63 | 1.013x |
| FP8 | 512 | 4 | 539.45 | 533.56 | 318.47 | 321.99 | 1.011x |
| FP4 | 64 | 4 | 70.60 | 71.38 | 304.19 | 300.84 | 0.989x |
| FP4 | 128 | 4 | 131.34 | 131.99 | 327.00 | 325.40 | 0.995x |
| FP4 | 256 | 4 | 247.43 | 248.10 | 347.17 | 346.23 | 0.997x |
| FP4 | 512 | 4 | 489.66 | 497.20 | 350.85 | 345.53 | 0.985x |
| FP8 | 1 | 4096 | 1045.89 | 1032.69 | 328.52 | 332.72 | 1.013x |
| FP4 | 1 | 4096 | 936.84 | 953.41 | 366.76 | 360.39 | 0.983x |

The FP4 shared-cache case had one anomalous round in the full sweep: its
manual mean was 6749.84 μs, with eight launches between 6.25 and 10.57 ms.
The other two round means were 953.41 and 953.17 μs. All samples remain in the
JSON and the table follows the same median-of-round-means rule as every case.
A separate five-round recheck gave manual round means of
952.22, 953.05, 952.43, 953.13, 952.65 μs, with median
952.65 μs, versus 939.97 μs
for the reference. The cause of the anomalous full-sweep capture is unresolved;
see [the full recheck](results/decode-shared-fp4-recheck.json).

## Correctness and scope

The correctness runner checks both variants against the PR's PyTorch
reference, using its output cosine/error tolerances and logits/LSE checks.
Prefill includes 14 cases: sinks, strided inputs, per-query lengths including
zero, invalid indices, partial index groups, and amplified KV values that
trigger rescaling. Decode includes 9 cases: FP8-only, FP8+FP8, FP8+FP4,
zero/all-invalid requests, per-query main/extra lengths, and odd page sizes.
Performance cases additionally compare the two implementations' outputs.

The supported example scope is H=64, D=512 and compact KV pages. Each quantized
cache must fit in 32-bit byte offsets. Runtime topk lengths are implemented as
masks over the static topk loop, while the reference shortens the loop. Thus
variable-topk performance is not claimed by these fixed-topk measurements.

## Controlled changes

The diagnostic case uses nq=4096, nk=8192, topk=640. Intermediate measurements
below use one ten-launch round, except the initial measurement (three rounds).
They guide tuning; the final comparison above uses three rounds for every case.

| Manual implementation | Latency μs |
| --- | ---: |
| Initial explicit pipeline | 1625.56 |
| Match gather / skip-scale notification ordering | 1251.82 |
| Match softmax row grouping and packed conversion | 1242.92 |
| Rely on explicit masks and zero-burst DMA | 1206.52 |
| Hoist scalar indices out of VF | 1206.47 |
| Specialize first-block / accumulation VFs and process two output rows together | 850.94 |

The last experiment combines VF specialization with the reference's paired-row
output epilogue; it does not isolate the contribution of either change alone.
Dynamic-UB alignment had no material effect in the measured case.

## Unroll and code size

The source explicitly keeps persistent and SIMD row-group loops rolled, while
unrolling the same small QK/PV, gather and ND-to-NZ groups as the reference.
The native flags include `-Ofast -mllvm -enable-hiipu-vf-loop-unroll`.
The ND-to-NZ VF is 536 bytes in both binaries. For the representative prefill
specialization, total device `.text` is 14,440 bytes for the reference and
15,560 bytes for the manual version. All manual SIMD VFs report zero stack
usage; the largest reported vector-register count is 12. This inspection did
not collect hardware ICache-miss counters.

## Evidence and commands

[README.md](README.md) contains the exact reproduction commands. Raw launch
samples and round means are saved in [results/prefill-bench-final.json](results/prefill-bench-final.json)
and [results/decode-bench-final.json](results/decode-bench-final.json). The
correctness JSON files, source hashes, compiler version and function sizes
are in [results](results).

These hardware results were collected on the pinned TileLang revision above.
The publication branch targets a newer `main`; hardware execution has not been
repeated on that base. `results/environment.json` retains the measurement
source hashes separately from the published source hashes. The attention core,
entry points, dequantization, and intrinsic adapters match the measured sources;
the published runners expose only the reference and manual implementations.

The full local debug evidence is under `.artifacts/flashmla-pr229/` in the
worktree: `final-suite.log`, generated `.asc` files, intermediate experiments,
`prefill-specialized-resources.log`, `prefill-specialized-symbols.txt`,
`reference-prefill-symbols.txt`, and the pinned reference checkout. These large
scratch artifacts are ignored by Git. Transient device initialization
failures were retried before the final suite; failed starts provide no samples.
