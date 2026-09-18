# KDA kernel implementation with TileLang
## Requirement
- TileLang: 0.1.6.post2+cuda.git729e66ca
- triton: 3.2.0
- FLA: commit 9714c5(used for comparison)

We copy the needed files and function from flash-linear-attention to the FLA_KDA/ for easily comparison.

## Packed safe-gated decode

`example_safe_gated_kda_decode.py` implements a one-token packed KDA decode
primitive for models that use the bounded safe gate:

```text
g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
beta = sigmoid(b)
```

The kernel consumes post-convolution packed QKV, applies Q/K L2 normalization,
updates a slot-indexed `[value_heads, value_dim, key_dim]` recurrent state in
place, and returns `[batch, 1, value_heads, value_dim]`. A state index of `-1`
produces zero output without accessing or modifying the state pool.
Non-negative state indices must be unique within a decode batch, as each active
request owns one mutable state slot.
Positive indices outside the configured state pool are invalid: the reference
rejects them, while the device kernel defensively produces zero output without
reading or writing state so a bypassed host check cannot corrupt GPU memory.

The default activation/state types are BF16/FP32. GLM-5.3-Flash with TP8 uses
eight local Q/K/V heads with `key_dim=value_dim=128` and `lower_bound=-5.0`.
The implementation is model- and GPU-architecture-independent; callers provide
the dimensions and gate bound when specializing the kernel.
