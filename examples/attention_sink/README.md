# Attention Sink

We compare with an optimized version of the official Triton implementation at [here](https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py).


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

- dtype=float16
- batch_size=1, heads=64, kv_heads=8 (the setting of GPT-OSS-120B)
- Full attention is adopted.

| SEQ_LEN | headdim | Triton TFLOPs | TileLang TFLOPs      | Speedup |
|---------|---------|---------------|----------------------|---------|
| 2048    |   64    | 231.55        | **277.07**           | 1.20x   |
| 2048    |  128    | 313.55        | **393.98**           | 1.26x   |
|         |         |               |                      |         |
| 4096    |   64    | 272.17        | **337.30**           | 1.24x   |
| 4096    |  128    | 356.35        | **461.54**           | 1.30x   |
|         |         |               |                      |         |
| 8192    |   64    | 289.93        | **353.81**           | 1.22x   |
| 8192    |  128    | 392.18        | **482.50**           | 1.23x   |
|         |         |               |                      |         |
| 16384   |   64    | 299.52        | **377.44**           | 1.26x   |
| 16384   |  128    | 404.64        | **519.02**           | 1.28x   |

> The backward performance will be further optimized via fine-grained manual pipelining of FA3 in the tilelang kernel.