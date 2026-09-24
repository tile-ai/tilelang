# FlashAttention (Ascend NPU)

Online-softmax FlashAttention 前向与 GQA backward kernel。MHA 与 GQA 前向共享
`core.py` 中的 `flash_attention_fwd` builder；MHA 是
`q_len == kv_len`、单 query tile per core 的特例，GQA 把 query group 展平成
`q_len = S1 * G` 并按 `num_blocks` 切分到多核。

GQA backward 使用 forward 可选输出的 LSE，只保留当前最快的 `T.Stage` +
AutoSchedule 路径：先用
`flash_attention_bwd_preprocess` 计算所有 query row 共享的
`Delta = sum(O * dO, axis=-1)`，再执行 frontend-staged KV-centric mixed kernel。后者在
一个 kernel 内完成五次 GEMM，私有累加 dK/dV，并把 dQ 以 bfloat16 atomic add
归约到 GM。若把 Delta 放进按 KV tile 切分的 fused kernel，完整 shape 会把
`O * dO` reduction 重复 64 次，因此这两个 launch 都是保留路径的一部分。

```
core.py          # 通用 flash_attention_fwd builder + FwdTiling
core_bwd.py      # Delta 与 T.Stage AutoSchedule fused builders
example_mha.py   # MHA wrapper + reference + benchmark
example_gqa.py   # GQA wrapper + reference + benchmark
example_gqa_bwd.py
example_gqa_manual_schedule.py  # GQA with a validated fixed-stage schedule
test_mha.py / test_gqa.py / test_gqa_bwd.py
```

## 局限性

MHA/GQA 前向实例共享同一条 SIMD softmax-packing 实现；backward packing 目前也
沿用相同的 128x128 固定 tile 约束：

- **`head_dim` 固定为 128。** softmax 直接把概率写成 NZ 布局（128 列 `vsstb` stride、
  `uint16x128` 合并），这条路径硬编码了 D=128，builder 里以 `assert head_dim == 128`
  显式限制，未泛化前不接受其它 D。
- **tile 形状固定。** `block_q` 必须偶数，`block_kv == 2 * VL == 128`（fp32 SIMD lane
  数 64）。`q_len % block_q == 0`、`kv_len % block_kv == 0` 必须整除。
- **输入 dtype 固定为 bfloat16**，累加为 float32。输出 dtype 可选：MHA 用 float32
  直出（无 cast），GQA 用 bfloat16（额外一趟 `vcvt` cast，写入 `O_ub` 的 byte-alias，
  不额外占 UB）。
- **无 causal mask。** 不支持 dropout、attention bias、变长 / padding mask。
- **backward 当前仅覆盖 GQA 的 flattened 布局。** 输入、`dO` 与输出梯度均为
  bfloat16，其中 dQ 跨 KV tile 做 BF16 atomic add。tile 固定为 128x128。MHA
  backward 尚未封装，但其数学路径等价于 `G=1`。
- **staged backward 至少需要三个 query tile。** 要求 `q_len / 128 >= 3`；
  四层 Q/dO 缓冲不提高最短输入要求。
- **GQA `num_blocks` 必须整除 `q_len / block_q`**（M tile 数），否则各核负载不均。

## 性能（当前 shape，Ascend 950 / dav-3510，bf16）

| kernel | shape | TileLang | Torch SDPA | 对比 |
|---|---|---|---|---|
| MHA | `SEQ_LEN=4096, D=128` | 26.8 us · 320.1 TFLOPS | 28.94 us · 296.8 TFLOPS | **TileLang 快 1.08x** |
| GQA | `S1=8192, G=32, S2=8192, D=128` | 3030 us · 362.8 TFLOPS | 2864 us · 383.9 TFLOPS | Torch 快 1.06x |
| GQA backward（T.Stage fused） | `S1=8192, G=32, S2=8192, D=128` | **6.418 ms · 428.3 effective TFLOPS** | — | 见下方同环境优化前后对照 |
| GQA（manual） | 同上 | 3125 us · 351.8 TFLOPS | — | 同轮达到 auto 的 92.9% |

前向正确性相对误差 < 0.5%。完整 backward shape 上，T.Stage fused BF16 输出相对
Torch BF16 梯度的 dQ/dK/dV 相对 L2 误差约为 1.024% / 0.316% / 0.037%。dQ 的较大
误差来自跨 64 个 KV tile 的 BF16 atomic 逐次舍入。

Backward 数据于 2026-09-24 在 Ascend950DT、CANN/Bisheng 9.2.0 上测得，使用
TileLang `8121f415` 的本地构建，开启 fast-math。优化前后使用同一组输入和预分配
输出，五轮按 A/B、B/A 交替测量。每轮为 cold-L2 `msprof_detail` FFTS kernel
duration，warmup 5 次、repeat 20 次；下面报告五轮均值的中位数和范围。
计时包含 **dQ 清零 + fused backward**，不含 forward、Delta 或 cache flush。
FLOPs 按五个 GEMM 的 `10 * Q * K * D` 计算。

| backward 实现 | 中位数 | 五轮范围 | effective TFLOPS |
|---|---:|---:|---:|
| 优化前（`4347e52e`） | 6834.581 us | 6830.059–6837.689 us | 402.19 |
| UnitFlag + FixPipe dQ + 缓冲重分配 | 6418.273 us | 6417.254–6418.779 us | 428.27 |

延迟下降 6.09%，吞吐提高 6.49%。十份原始 profile 均核实有 20 次 fused AIC、
20 次 fused AIV、20 次清零和 20 次 cache flush，mixed kernel 的时间只累计一次。
清零约 15.4 us；fused 本身由约 6819.2 us 降至 6402.9 us。当前环境未复现早期
430.6 TFLOPS 的历史数字，表中对照均来自同一工具链，不将差异归因于某个编译器提交。

优化同时调整 L0C 交接、dQ 数据路径和缓冲分配。仅添加 UnitFlag 约为 6.753 ms；
在原三层缓冲上改用 FixPipe dQ 约为 6.755 ms；结合四层 Q/dO、两层中间缓冲才
达到 6.418 ms。AIC MAD active 从约 96.86% 提高到 99.80%。UnitFlag 下 FixPipe
active 包含等待就绪时间，接近 100% 不代表带宽耗尽。

保持此前 cannsim RVEC 估算的 `SimdVF` latency：Delta 1053 cycles、P/dS pack
706 cycles。优化版已移除 dQ Vector cast。

### T.Stage AutoSchedule fused backward 实现快照

入口是 `core_bwd.py::flash_attention_bwd_fused_dq_atomic_staged`。grid 按 KV tile
切分，每个 AIC 独占一个 128x128 K/V tile，并在 2048 个 query tile 上执行：

| 顺序 | GEMM | 结果与所有权 |
|---:|---|---|
| 1 | `K @ Q.T` | score，经 FixPipe 分给两个 AIV |
| 2 | `V @ dO.T` | dP，经 FixPipe 分给两个 AIV |
| 3 | `P @ dO` | dV，在该 KV core 的 L0C 中 FP32 累加 |
| 4 | `dS @ Q` | dK，在该 KV core 的 L0C 中 FP32 累加 |
| 5 | `dS.T @ K` | dQ contribution，转 BF16 后 atomic add 到 GM |

`T.Stage(0)` 产生 score/dP 并打包 P/dS，`T.Stage(1)` 消费上一 query tile 的
P/dS。Q/dO L1 使用四层缓冲，P/dS L1、score/dP UB 和 LSE/Delta UB 使用两层；
L0A/L0B 保持两个槽、逐 GEMM 交替，P/dS packing scratch 保持单槽。L1 总占用
仍为 448 KiB，UB 从 243.5 KiB 降到 162.5 KiB。

score、dP、dQ 的 GEMM 和输出 copy 成对指定 `unit_flag_ctrl=3`，保护临时 L0C
的就绪与复用。dQ 仍借用已排空的 `dp_l0c`，由 FixPipe 直接转换为 BF16 并原子
写回 GM，省去 L0C→UB、Vector cast 和 UB→GM。每个部分梯度先转 BF16 再累加，
调用方仍须清零 dQ，原子归约仍不保证逐位确定性。

版本数由 kernel 指定；版本索引、流水线展开及其余本地/跨核同步由 AutoSchedule
生成。去掉 `T.Stage` 的对照虽通过正确性验证，但约为 6.897 ms，因此保留阶段约束。

该 PrimFunc 保持 AutoSchedule 与 shared-memory reuse 默认开启，只需沿用 fast-math：

```python
{
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
```

## 运行

```bash
source haienv tilelang
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_mha.py
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa.py
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa_bwd.py
# GQA backward 与 Torch SDPA backward 的完整 shape 性能对比
ASCEND_NPU_ARCH=dav-3510 python example_gqa_bwd.py --perf
# 固定 13 个 task 的 stage；加 --benchmark 运行完整性能 shape
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python example_gqa_manual_schedule.py
# 或跑正确性测试
ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 python -m pytest test_mha.py test_gqa.py test_gqa_bwd.py -v
```
