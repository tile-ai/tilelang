# Attention Sink

We compare with an optimized version of the official Triton implementation [here](https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py).

## Algorithm
### Forward
The only change from vanilla FlashAttention is that `sinks` should be taken into consideration in the softmax, which requires an extra rescaling at the epilogue stage.

### Backward
Based on detailed mathematical derivation, interestingly, the backward computation process of `dQ`, `dK`, `dv` is almost identical to that in vanilla FlashAttention, except for that the specific meanings of `lse` differ. We only need to compute `dsinks` additionally, which is given by:

$$
dsink_h=-\sum_{b}\sum_{q}P_{b, h, q}Delta_{b, h, q}
$$

where $P_{b, h, q}$ is the proportion of $sink_h$ in the softmax in the $b$-th block, $h$-th head and $q$-th query(row).

## Benchmark of forward process

### Benchmark Environment
- **Hardware**: NVIDIA H800
- **CUDA version**: 12.9
- **Triton Version**: 3.4.0

### Results

- dtype=bfloat16
- batch_size=1, heads=64, kv_heads=8 (the setting of GPT-OSS-120B)
- Full attention is adopted.

| SEQ_LEN | headdim | Triton TFLOPs | TileLang TFLOPs      | Speedup |
|---------|---------|---------------|----------------------|---------|
| 2048    |   64    | 232.98        | **281.89**           | 1.21x   |
| 2048    |  128    | 321.55        | **417.98**           | 1.30x   |
|         |         |               |                      |         |
| 4096    |   64    | 280.70        | **349.47**           | 1.25x   |
| 4096    |  128    | 369.61        | **497.13**           | 1.35x   |
|         |         |               |                      |         |
| 8192    |   64    | 299.04        | **385.56**           | 1.29x   |
| 8192    |  128    | 399.39        | **507.93**           | 1.27x   |
|         |         |               |                      |         |
| 16384   |   64    | 309.46        | **400.62**           | 1.29x   |
| 16384   |  128    | 418.99        | **549.11**           | 1.31x   |

> The backward performance will be further optimized in the future.
