# GEMM NV FP4 Feature – 现状 Review 与实现步骤评估

## 一、当前代码库现状

### 1. 已有基础（可直接复用）

| 模块 | 现状 |
|------|------|
| **FP4 类型与转换** | `src/tl_templates/cuda/cuda_fp4.h` 已有 `__nv_fp4_e2m1`、`fp4_e2_t`、fp4↔half/float/bf16 转换，以及 packed load/store 辅助函数。 |
| **Load/Store** | `lower_ldg_stg.cc` 已对 `float4_e2m1fn` 做 4-bit 处理（bits()、total_bits 等）。 |
| **CuTeDSL codegen** | `codegen_cutedsl.cc` 已支持 `Float4E2M1FN`。 |
| **Python dtype** | `tilelang/language/dtypes.py` 已有 `float4_e2m1fn` 及 x2/x4/…/x64 向量类型；与 torch 的映射在 `dtypes.py` / `tensor.py`。 |
| **TCGEN5 窄精度 meta** | `src/op/tcgen5_meta.h` 中 `GetTCGEN5MMAMeta` 已把 `float4_e2m1fn` 与 float8/float6 一起纳入窄精度分支：K%32==0，M/N 与现有 F8/F6 同规则，C 为 float32 或 float16。 |
| **SM100 MMA 模板** | `gemm_sm100.h` 已有 `SM100_MMA_F8F6F4_WS_SS`（PTX `kind::f8f6f4`），支持 ≤8bit 类型；当前仅对 `cute::float_e4m3_t` / `float_e5m2_t` 做了 `DispatchInstruction`。 |
| **Dequant 示例** | `examples/dequantize_gemm/` 下有 FP4 相关示例（如 `example_dequant_gemm_fp4_hopper.py`），用于参考 FP4 数据流与 scaling。 |

### 2. 缺口（需要补齐）

| 模块 | 问题 |
|------|------|
| **指令描述符** | `GetTCGEN5InstrDesc()` 中 `encode_dtype()` 未处理 `float4_e2m1fn`，会走到 `LOG(FATAL)`。需为 FP4 分配正确的 format 编码（需查 CUTLASS/PTX 描述符定义）。 |
| **PTX 类型枚举与字符串** | `src/target/ptx.cc` 的 `DTypeFromString()` 无 `"float4_e2m1fn"`；`common.h` 的 `DataType` 枚举无 `kFloat4_e2m1fn`；`enum_to_str` / `num_bits` 等表也需扩展。 |
| **GEMM 到 codegen 的 dtype 传递** | 当前 GEMM Lower 时把 `ab_dtype` 转成字符串传给后端；需保证 TVM `float4_e2m1fn` → 字符串 `"float4_e2m1fn"` → PTX/codegen 能识别并在 C++ 端选用 FP4 路径。 |
| **C++ 侧 FP4 类型与 Dispatch** | `gemm_sm100.h` 中 `DispatchInstruction` 仅有 `float_e4m3_t` / `float_e5m2_t`；需增加 FP4 的 C++ 类型（如对接 `cuda_fp4.h` 的 `fp4_e2_t` 或 CUTLASS 的 FP4 类型）以及对应的 `DispatchInstruction` 特化。 |
| **CuTe/CUTLASS 对 FP4 的支持** | `UMMA::make_instr_desc`、`sm100_smem_selector` 等是否支持 4-bit 类型需确认；若 CUTLASS 已有 FP4 描述符/布局，需在 TileLang 侧对齐。 |
| **Block-scaled FP4（可选）** | NVIDIA 文档中的 `mxf4.block_scale` / `mxf4nvf4.block_scale` 为单独路径，若要做 block-scaled GEMM，需额外设计 scale 的存储与传入方式，可作二期。 |

---

## 二、推荐实现步骤（按依赖顺序）

### Phase 1：类型与描述符贯通

1. **PTX / common 数据类型**
   - 在 `common.h` 的 `DataType` 中增加 `kFloat4_e2m1fn`（或与现有命名规范一致）。
   - 在 `ptx.cc` 的 `DTypeFromString` 中增加 `"float4_e2m1fn"`（及可选别名）映射到该枚举。
   - 在 `enum_to_str`、`num_bits`、`dtype_str` 等表中为 FP4 填好条目，保证 codegen 能正确生成类型名与位宽。

2. **TCGEN5 指令描述符**
   - 查 CUTLASS/PTX 文档中 tcgen05.mma `kind::f8f6f4` 下 FP4 的 descriptor 编码（与 e4m3/e5m2 的差异）。
   - 在 `tcgen5_meta.h` 的 `GetTCGEN5InstrDesc()` 里为 `float4_e2m1fn` 增加 `encode_dtype` 分支，返回正确的 format 码，保证现有 GEMM  Lower 路径在 A/B 为 FP4 时能生成合法描述符。

3. **验证路径**
   - 写一个最小用例：只做「分配 FP4 buffer + 从 Python 传 dtype 到 codegen」，确认：
     - TVM dtype → 字符串 → `DTypeFromString` → `DTypeEnumToString` 一致；
     - 若已有 FP4 的 LDG/STG，可顺带跑一条简单 load/store 或 dequant 示例，确认无回归。

### Phase 2：C++ GEMM 模板与 Dispatch

4. **C++ FP4 类型与 CuTe 对接**
   - 确定 GEMM 使用的 C++ 类型：`cuda_fp4.h` 的 `fp4_e2_t` 或 CUTLASS 的等价类型。
   - 若 CUTLASS 有 `make_instr_desc` 对 FP4 的重载，在 TileLang 的 `to_cute_type`（或当前用于 F8/F6 的映射处）增加 TVM float4_e2m1fn → 该 C++ 类型的映射。
   - 若 CUTLASS 的 `sm100_smem_selector` 对 4-bit 有特殊布局（例如 packed 2×fp4/byte），在 `GemmTensorOp` 的 A/B layout 路径中接入，保证与 descriptor 一致。

5. **DispatchInstruction 与 SS/WS**
   - 在 `gemm_sm100.h` 中为 FP4 增加 `DispatchInstruction<fp4_type, fp4_type, float, ...>`（以及 C=half 若需要），指向已有的 `SM100_MMA_F8F6F4_SS` / `SM100_MMA_F8F6F4_WS_SS`。
   - 确认 PTX 的 `kind::f8f6f4` 对 FP4 的 M/N/K 与 tile 约束与当前 meta 一致（K=32 等）；必要时对照 CUTLASS 72a_blackwell_nvfp4_bf16_gemm 示例。

6. **端到端 GEMM 调用**
   - 在 `gemm.cc` / `gemm_py.cc` 的 Lower 中，当 `a_`/`b_` 为 float4_e2m1fn 时，传入的 `kind_dtype` 字符串需与 `DTypeFromString` 一致（如 `"float4_e2m1fn"`）。
   - 跑一次 SM100 上 A/B 为 FP4、C 为 float32 的 GEMM kernel，对比参考实现（如 CUTLASS 72a）做数值与正确性检查。

### Phase 3：示例与文档

7. **Example**
   - 在 `examples/gemm_sm100/` 或新建 `examples/gemm_fp4_sm100/` 增加一个 NV FP4 GEMM 示例：T.float4_e2m1fn 作为 A/B dtype，accum float32，与 ref（先 dequant 再 bf16/f32 GEMM）对比。
   - 若后续做 block-scaled，可在同一目录下加另一个示例，区分「无 scale」与「block scale」两种用法。

8. **文档与 CI**
   - 在 `docs/programming_guides/type_system.md` 或 GEMM 相关文档中注明：SM100 TCGEN5 MMA 支持 float4_e2m1fn，以及当前限制（K%32、M/N 与 meta 一致、是否支持 block scale）。
   - CI 中为 SM100 增加一条 FP4 GEMM 的编译/运行（若环境有 SM100 或通过 skip 条件处理）。

---

## 三、风险与依赖

- **CUTLASS 版本**：FP4 描述符与 `make_instr_desc` 的细节可能随 CUTLASS 版本变化，需以当前 submodule 版本为准核对。
- **PTXAS / CUDA**：`cuda_fp4.h` 中已有对 CUDA &lt; 13 的 workaround（如 `__tl_cvt_fp4_to_halfraw_naive`），若运行环境 CUDA 版本较新，可再确认是否仍需要这些分支。
- **Block-scaled FP4**：若产品需要 4× 吞吐的 block-scaled 路径，需单独设计 scale 张量、描述符与 API，工作量大于「仅非 block-scaled FP4 GEMM」。

---

## 四、与 Flash Attention SM100 的优先级

- Flash Attention SM100 的 example（含 wasp）可后续再迭代；当前不作为重点。
- GEMM NV FP4 作为独立 feature，与 FA 的改动解耦；完成上述 Phase 1–2 即可得到可用的 FP4 GEMM 路径，再视需求加 block scale 或集成到更高层算子。

---

## 五、本次已完成的修改（CuTeDSL/TCGEN5 路径）

以下修改已落地，使 **CuTeDSL 生成的 SM100 GEMM kernel** 在 A/B 为 `float4_e2m1fn` 时能正确走 `kind::f8f6f4` 的 FP4 路径。当前未改 `gemm_sm100.h` 的 CUTLASS `DispatchInstruction`（该路径由 CUTLASS 模板实例化，与 CuTeDSL 生成 TIR 再 codegen 的路径分离）。

| 文件 | 修改内容 |
|------|----------|
| **src/target/ptx.h** | `DataType` 枚举增加 `kFloat4_e2m1fn = 23`。 |
| **src/target/ptx.cc** | `enum_to_str` / `dtype_str` / `num_bits` 增加第 24 项；`DTypeFromString` 增加 `"float4_e2m1fn"`、`".e2m1"` → `kFloat4_e2m1fn`。 |
| **src/tl_templates/cuda/common.h** | `DataType` 枚举增加 `kFloat4_e2m1fn = 23`。 |
| **src/op/tcgen5_meta.h** | `GetTCGEN5InstrDesc()` 中 `encode_dtype` 增加 `dtype.is_float4_e2m1fn()` 分支，返回 `2`（FP4 format 编码）。 |
| **src/tl_templates/cuda/instruction/tcgen05mma.h** | 为 `DataType::kFloat4_e2m1fn` 增加 `tcgen05mma_ss`、`tcgen05mma_ts`、`tcgen05mma_ws_ss` 特化，均转发到 `kFloat8_e4m3` 的 f8f6f4 实现。 |
| **tilelang/intrinsics/mma_macro_generator.py** | `dtype_abbrv` 增加 `"float4_e2m1fn": "float4_e2m1fn"`，保证 lowering 时 `a_dtype_abbrv` 传入 `ptx_tcgen05_mma_ss("float4_e2m1fn", ...)`。 |

**后续建议**（在有 SM100 / nvcc 环境时）：

1. 本地或 CI 执行完整构建并跑 `examples/gemm_sm100/` 中 BF16/F8 用例，确认无回归。
2. 新增 FP4 GEMM 示例：`in_dtype=accum_dtype=T.float4_e2m1fn, T.float`，`block_K=128`（K%32 已由 meta 保证），与参考实现做数值对比。
3. 若需走 CUTLASS 模板路径的 FP4 GEMM，再补 `gemm_sm100.h` 的 `to_cute_type<fp4_e2_t>` 与 `DispatchInstruction<fp4_e2_t, fp4_e2_t, float, ...>`。
