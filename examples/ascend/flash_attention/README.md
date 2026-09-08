# FlashAttention (Ascend NPU)

Online-softmax FlashAttention 前向 kernel。MHA 与 GQA 共享
`core.py` 中的 `flash_attention_fwd` builder；MHA 是
`q_len == kv_len`、单 query tile per core 的特例，GQA 把 query group 展平成
`q_len = S1 * G` 并按 `num_blocks` 切分到多核。

```
core.py          # 通用 flash_attention_fwd builder + FwdTiling
example_mha.py   # MHA wrapper + reference + benchmark
example_gqa.py   # GQA wrapper + reference + benchmark
example_gqa_manual_schedule.py  # GQA with a validated fixed-stage schedule
test_mha.py / test_gqa.py
```

## 局限性

两个 kernel 共享同一条 SIMD softmax-packing 实现，因此约束一致：

- **`head_dim` 固定为 128。** softmax 直接把概率写成 NZ 布局（128 列 `vsstb` stride、
  `uint16x128` 合并），这条路径硬编码了 D=128，builder 里以 `assert head_dim == 128`
  显式限制，未泛化前不接受其它 D。
- **tile 形状固定。** `block_q` 必须偶数，`block_kv == 2 * VL == 128`（fp32 SIMD lane
  数 64）。`q_len % block_q == 0`、`kv_len % block_kv == 0` 必须整除。
- **输入 dtype 固定为 bfloat16**，累加为 float32。输出 dtype 可选：MHA 用 float32
  直出（无 cast），GQA 用 bfloat16（额外一趟 `vcvt` cast，写入 `O_ub` 的 byte-alias，
  不额外占 UB）。
- **仅前向、无 causal mask。** 不支持 dropout、attention bias、变长 / padding mask。
- **KV 长度需 ≥ `num_stages * block_kv`。** pipeline 展开需要足够多的 KV block；小
  shape（如 `kv_len < 3 * 128`）会在 lowering 阶段报 `RewriteFlagToBuf: negative
  event_id`，属于 pipeline 深度不足，不是数值错误。
- **GQA `num_blocks` 必须整除 `q_len / block_q`**（M tile 数），否则各核负载不均。

## 性能（当前 shape，Ascend 950 / dav-3510，bf16）

| kernel | shape | TileLang | Torch SDPA | 对比 |
|---|---|---|---|---|
| MHA | `SEQ_LEN=4096, D=128` | 26.8 us · 320.1 TFLOPS | 28.94 us · 296.8 TFLOPS | **TileLang 快 1.08x** |
| GQA | `S1=8192, G=32, S2=8192, D=128` | 3030 us · 362.8 TFLOPS | 2864 us · 383.9 TFLOPS | Torch 快 1.06x |
| GQA（manual） | 同上 | 3125 us · 351.8 TFLOPS | — | 同轮达到 auto 的 92.9% |

正确性：两者相对误差均 < 0.5%。GQA 仍略慢于 Torch，瓶颈在
softmax VF / MTE3 写回一侧。

## 运行

```bash
source haienv tilelang
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_mha.py
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa.py
# 固定 13 个 task 的 stage；加 --benchmark 运行完整性能 shape
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa_manual_schedule.py
# 或跑正确性测试
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python -m pytest test_mha.py test_gqa.py -v
```
