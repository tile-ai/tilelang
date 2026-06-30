# SM120 NVFP4 blockscaled GEMM design note

这份文档记录当前 SM120 NVFP4 blockscaled GEMM PR 的设计边界，重点解释
TMA shared-memory view、per-lane operand copy view、scale-factor swizzle 和后续
TileLang 抽象收束方向。

## 当前 PR 的目标

当前 PR 先建立一个可验证的 SM120a NVFP4 blockscaled GEMM 路径，而不是一次性
完成所有 CUTLASS/CuTe 风格抽象。当前性能路径包含：

- packed FP4 A/B operands 的 TMA load。
- SFA/SFB scaling factors 的 TMA load。
- persistent pingpong 风格的 warp-specialized producer/consumer pipeline。
- 一个 producer warp group 加两个 consumer warp groups。
- `C` accumulator epilogue 走 `reg -> smem -> gmem`。
- profiling 后选定的主要配置：`128x128x256`, `num_stages=2`。
- `cutlass_128x4` scale-factor source swizzle，用于改善 scale-factor load 的访问形态。

当前 8192^3 constant-scale 大形状约为 1.26 PFLOPS，PR 描述中按当前状态写作约
1.3 PFLOPS。更深的抽象清理放到后续 PR。

## 两种 view 需要分开

这个 kernel 里容易混淆的是 shared-memory tile 的连续性和 MMA operand fragment
的连续性。它们不是同一个问题。

```text
Producer/TMA view:
  gmem tile -> smem tile
  主要约束是 bulk copy、128B 对齐、tile/panel 粒度和 TMA descriptor 能表达的布局。

Consumer/fragment view:
  smem tile -> ldmatrix/LDSM -> named registers -> OMMA.SF
  主要约束是每个 lane 给 OMMA.SF 的 A/B fragment register 顺序。
```

A/B shared allocation 本身仍然是一块连续的 shared buffer。TMA 的 128B 粒度会推动
layout 采用 panelized/aligned 的 producer view；对 FP4 来说，128B 等于 256 个 FP4
元素，也刚好贴近 `block_K=256` 的主配置。

但是 consumer 端不是按“某个 thread 从 smem 读一段连续 byte slice”来喂 MMA。
`ldmatrix.sync.aligned.x4.m8n8.shared.b16` 是 warp-level copy atom：每个 lane 给出
自己的 shared address，硬件 collectively load matrix fragment，并把结果落到后续
OMMA.SF 需要的 register operands。也就是说，smem physical buffer 可以连续，
per-lane fragment view 仍然是 lane-dependent、K-swizzled、register-package view。

当前代码里的证据在 `src/tl_templates/cuda/gemm_sm120.h`：

- `sm120_fulltile_compact_a_offset(tx, k_block_idx, row_idx)`
- `sm120_fulltile_compact_b_offset(tx, k_block_idx, panel_idx)`
- `sm120_fulltile_compact_k_swizzle_offset(tx, k_block_idx)`

这些 offset 不是普通 `row * stride + col`。它们依赖 `threadIdx.x`、`k_block_idx`、
row/panel index 和 K swizzle。例如 compact A/B offset 中有 128B panel 步进，也有
`row_idx * 4096` / `panel_idx * 4096` 这样的 package stride。含义是：一个 OMMA.SF
issue 需要的 A/B register package 来自 shared tile 的若干 swizzled positions，而不是
一个 thread-local contiguous chunk。

更精确的说法是：TMA 的 128B 粒度影响 producer shared layout，但 fragment view
非连续的直接原因是 SM120 OMMA.SF operand package contract。

## scale-factor source swizzle

当前性能路径要求 host 侧传入的 SFA/SFB storage 已经按 `cutlass_128x4` source layout
重排。语义上 SFA/SFB 是 `[M or N, K / 64]` 的 `uint32` matrix，每个 word 包含四个
连续 K/16 group 的 scale bytes。性能路径的 source storage 则按 128 rows x 4 words
组织：

```text
semantic row i, semantic K-word k
  -> physical row = k * 32 + (i % 32)
  -> physical word = i / 32
```

这样做的目的不是改变数学语义，而是让 scale-factor TMA load 和 consumer 端的
SM120 full-tile helper 看到更适合 OMMA.SF scale operand 的 packed layout。benchmark
里 reference checking 保留 semantic row-major copy，kernel 输入则使用 swizzled copy。

scale swizzle 只能解决 scale-factor package 的局部性和访问形态。它不能自动消除 A/B
operand 的 per-lane copy/package 需求，因为 A/B 仍然要通过 `ldmatrix/LDSM` 进入
OMMA.SF 规定的 named register operands。

## TileLang 当前缺少的中间层

TileLang 现在已经有外层工具：

- `T.tma_copy` 表达 producer 端 bulk copy。
- `T.Pipelined` 和 `T.ws` 表达 pipeline / warp specialization skeleton。
- `T.alloc_fragment`、`T.gemm`、`T.mma_gemm_blockscaled` 表达 logical tile op。

但是 SM120 NVFP4 极致路径还需要一层更细的 contract：

```text
per-lane copy atom view:
  lane_id
  -> smem logical coord
  -> smem physical byte offset
  -> ldmatrix/LDSM variant
  -> destination register tuple order
  -> OMMA.SF operand binding
```

普通 CuTe GEMM template 中有类似 `thr_copy_A.retile_D(tCrA)`、
`copy(tiled_copy_A, tCsA, tCrA_copy_view)` 的 copy-view 表达。但当前 SM120 NVFP4
blockscaled path 还没有把这个概念提升成 TileLang Python IR 或统一 lowering contract。
因此本 PR 的性能路径把这层逻辑局部放在 SM120 private builtin/template helper 中。

这也是 `sm120_ldmatrix_x4_blockscaled_operand*` 这类 helper 存在的原因。它们不是要
表达“整个 operand tile 在 smem 中不连续”，而是在表达：

- source address 可以是预先计算好的 PTX shared address register。
- destination 是 OMMA.SF 直接消费的 named scalar registers。
- register tuple 的顺序必须匹配 SM120 blockscaled MMA operand package。

## 后续 PR 的抽象方向

当前 PR 先收敛并提交，保持用户侧 API 尽量轻量。后续更扎实的工作应把 SM120
blockscale 支持收束到 TileLang 内部 abstraction，而不是继续把性能策略暴露在 example
kernel 中。建议的下一步是：

- 保留 `T.ws + T.Pipelined + T.tma_copy` 作为外层 warp-specialized pipeline skeleton。
- 在 lowering/policy/codegen 里加入 SM120 blockscaled operand package contract。
- 把 A/B operand `copy_view -> register fragment package -> OMMA.SF` 做成可复用内部抽象。
- 把 `cutlass_128x4` scale source contract、scale TMA layout、scale register selection 明确化。
- 让用户 API 继续保持轻量，只表达 blockscaled GEMM 的 tile-level intent。

这样可以同时保持 TileLang 的低心智负担用户接口，以及 CUTLASS/CuTe 风格的高性能
operand staging 约束。
