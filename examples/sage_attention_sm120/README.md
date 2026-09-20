# SageAttention3 FP4 on SM120

Pure TileLang port of the CUDA SageAttention3 **raw attention core**, using
upstream SM120 NVFP4 block-scaled MMA, TMA, and LDSM intrinsics.
Quantization and preprocessing are supplied by the reference implementation;
this is not an end-to-end replacement for `sageattn3_blackwell`.

Supported inputs: batch 1, head dimension 128, equal Q/KV head counts,
non-causal attention, BF16 output, and Q/KV padded to multiples of 128.
The packed FP4 values, FP8 scales, and block-mean correction must follow
SageAttention3's physical layout. `LSE` is reserved and not written.

Build the CUDA reference under `SageAttention/sageattention3_blackwell`, or
pass its location explicitly. Run from the TileLang repository root:

```bash
python -m examples.sage_attention_sm120.compare_cuda \
  --sageattention-root SageAttention/sageattention3_blackwell --json
python -m examples.sage_attention_sm120.sageattn3_alignment \
  --sage-root SageAttention/sageattention3_blackwell \
  --include-tilelang-raw --check --warmup 20 --rep 100 --json
```

The first command checks identical CUDA-quantized inputs, including a KV tail
and multiple KV tiles. The second checks the default 4128/4608-token,
30-head cases and compares raw-core CUDA Graph timings on the same GPU.
The default performance gate permits at most 5% higher median latency than
CUDA (`--min-speedup` is CUDA latency divided by TileLang latency).
Run without competing GPU workloads and repeat measurements; record the GPU,
CUDA/PyTorch versions, reference commit, and both per-case latencies.
The separately reported end-to-end CUDA timing is not the performance baseline
for the TileLang raw core.
