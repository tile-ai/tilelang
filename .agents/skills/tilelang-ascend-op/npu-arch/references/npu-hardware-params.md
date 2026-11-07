# NPU hardware parameter true value reference

> **Data source**: `${ASCEND_HOME_PATH}/<arch>/data/platform_config/*.ini` (arch such as `aarch64-linux`, `x86_64-linux`, `arm64-linux`), the parameter value is subject to the return of the `PlatformAscendC` interface at runtime.
>
> **Authoritative source for the DAV_3510 specifications**: [Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf), Table 3-1 (core counts, compute capability, memory, and L2 for each 950PR/950DT SKU) and Table 4-2 (physical capacity of each buffer in the memory hierarchy).
>
> **Data Credibility Convention**: "INI field"/"actual measurement" in this document refers to the static configuration value read from ini. **ini file is a necessary but not sufficient condition for discovering hardware parameters** — not every ini corresponds to a real mass-produced chip, and some configurations may come from engineering samples, simulation platforms, or planned SKUs. The document focuses on architecture-level constants (§1, §2) and key differences (§3). The changing parameters only use typical mass production SKUs as examples.

---

## 0. Product mapping table

List the product series, SocVersion, and specific chip models retained by this reference. **This table is the authoritative source shared by SKILL.md and `npu-arch-guide.md`. Update the chip list only here.**

| Product Series | SocVersion | NpuArch | __NPU_ARCH__ | Chip Model |
|---------|-----------|---------|:---:|---------|
| Atlas Training Series | ASCEND910 | DAV_1001 | 1001 | Ascend910 |
| Atlas reasoning series | ASCEND310P | DAV_2002 | 2002 | Ascend310P1, Ascend310P3, Ascend310P5, Ascend310P7 |
| Atlas A5 Training | ASCEND950 | DAV_3510 | 3510 | Ascend950DT (Decode) |
| Atlas A5 Inference | ASCEND950 | DAV_3510 | 3510 | Ascend950PR (Prefill) |
| Mate80, Pura90 and other series | Kirin9030 | DAV_3113 | 3113 | Kirin9030 |
| MatePad Edge, MateBookPro and other series | KirinX90 | DAV_3003 | 3003 | KirinX90 |

> **Model relationship**: `ASCEND950` / `DAV_3510` covers multiple Ascend950PR and Ascend950DT SKUs. Use `npu_arch=3510` for architecture-level capabilities and the complete `full_soc` for SKU-dependent resources and peaks.
>
> **950 Series Positioning** (950 White Paper §2/§3): **Ascend950PR** is oriented towards high-performance recommendation, large model **Prefill** and multi-modal reasoning; **Ascend950DT** covers the entire process of pre-training, post-training and inference (including **Decode** and Prefill) (see typical SKU examples for Memory specifications). (Decode)/(Prefill) in the above table are scene-focused annotations and are not the only uses.

---

## 1. Server cross-architecture consistent parameters

The following parameters are consistent across all validated **server** Ascend NPU architectures (except Kirin client-side platforms):

| Parameters | INI fields | Values | Description |
|------|---------|:---:|------|
| L0A | `[AICoreSpec] l0_a_size` | 64 KB (65536) | Cube left matrix operand |
| L0B | `[AICoreSpec] l0_b_size` | 64 KB (65536) | Cube right matrix operand |
| Cube MAC array | `cube_m_size / cube_k_size / cube_n_size` | 16×16×16 | Complete 4096 MACs in one cycle |

> **Note**: Even if the above parameters are usually the same, they should still be obtained through `GetCoreMemSize` in the code to avoid hard coding. Kirin end-side platform (DAV_3003/DAV_3113) Cube MAC array is different, and DAV_3113 L0A/L0B is smaller; see §2.4/§2.5 for details.

---

## 2. Each architecture parameter (usually consistent between sub-models)

Generally, the following parameter values are consistent with the sub-models of NpuArch:

### 2.1 DAV_1001 — Ascend910 Series

| Parameters | INI fields | Values |
|------|---------|:---:|
| NpuArch | `NpuArch` | 1001 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 32 MB (33554432) |
| BT | `bt_size` | — (does not exist) |
| sparsity | `sparsity` | — (does not exist) |
| Core types | `core_type_list` | `AICore` (no independent VectorCore) |
| Inter-core relationship | — | VectorCore does not exist, ai_core cohesive Cube+Vector function |

### 2.2 DAV_2002 — Ascend310P Series

| Parameters | INI fields | Values |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2002 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 16 MB (16777216) |
| Memory | `memory_size` | 24 GB (24000000000) |
| BT | `bt_size` | — (does not exist) |
| sparsity | `sparsity` | — (does not exist) |
| Core types | `core_type_list` | `AICore,VectorCore` (no independent CubeCore) |
| Inter-core relationship | — | The Cube function is integrated in AICore, and AICore and VectorCore have a non-1:2 relationship |

### 2.3 DAV_3510 — Ascend950DT / Ascend950PR Series

| Parameters | INI fields | Values |
|------|---------|:---:|
| NpuArch | `NpuArch` | 3510 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 248 KB (253952) |
| BT | `bt_size` | 4 KB (4096) |
| sparsity | `sparsity` | 0 (4:2 no longer supported) |
| Core types | `core_type_list` | `CubeCore,VectorCore` |
| Inter-core relationship | — | CubeCore : VectorCore = 1 : 2 |

> **Contrast with 950 white paper**: The physical capacity in Table 4-2 of the white paper is marked as L1 512KB / L0A/L0B 64KB / L0C 256KB per AI Core, consistent with INI; **UB is marked as 512KB per AI Core**, and the interpretation aligned with INI is the physical capacity (per AI Core group) (1 AIC + 2 AIV) AIV Physics 256KB × 2, "by group" size is extrapolated). INI `ub_size` 248KB is the available value per AIV user - physical 256KB − 8KB reserved = 248KB = 253952B, which exactly matches the INI (all 950PR/950DT SKUs of local CANN have been verified to be consistent). For the complete chain of evidence, see `npu-arch-guide.md` §Buffer capacity.

### 2.4 DAV_3003 — KirinX90 client-side series

> **Important Note**: KirinX90 client-side platform is used, the architecture code is `dav-l300`.

| Parameters | INI fields | Values | Differences from Server Edition |
|------|---------|:---:|------|
| NpuArch | `NpuArch` | 3003 | New schema, not in legacy mapping table |
| AIC_version | `AIC_version` | AIC-L-300 | End-side specific version identification |
| Number of cores | `ai_core_cnt` | 1 | **Single core** (server version multi-core) |
| VectorCore | `vector_core_cnt` | 1 | Single VectorCore |
| L1 | `l1_size` | 1 MB (1048576) | Client-side local buffer |
| L0A | `l0_a_size` | 64 KB (65536) | **Same as Server Edition** (32 KB for DAV_3113) |
| L0B | `l0_b_size` | 64 KB (65536) | **Same as server version** (32 KB for DAV_3113) |
| L0C | `l0_c_size` | 128 KB (131072) | Client-side capacity |
| UB | `ub_size` | 128 KB (131072) | **Smaller than DAV_3510 (248KB)** |
| L2 | `l2_size` | 0 | **No L2 Cache** |
| BT | `bt_size` | 1 KB (1024) | Client-side capacity |
| Cube MAC array | `cube_m/n/k_size` | 16×8×16 | **N dimension is reduced by half, the server version is 16×16×16** |
| Sparse | `sparsity` | 1 (supports 4:2) | Architecture field |
| Core types | `core_type_list` | `AICore,VectorCore` | Cube functions are integrated in AICore |
| vec_calc_size | `vec_calc_size` | 128 | Vector calculation unit size |

> **Development Notes**:
> - The UB capacity is only 128 KB. Special attention should be paid to the UB split size when designing Tiling.
> - No L2 Cache, data transfer strategy needs to consider GM pass-through
> - Cube array N=8 (instead of 16), matrix multiplication output dimensions are different
> - Single-core design, no need for multi-core segmentation
> - L0A/L0B is the same as the server version (64KB), but UB/L0C is smaller. Please pay attention to the transportation strategy of Cube output to UB



### 2.5 DAV_3113 — Kirin9030 end-side series

> **Important Note**: The Kirin client-side platform uses the `mobile-station` version of CANN, and development relies on the simulator rather than the actual hardware. Architecture codename `dav-l311`, `__NPU_ARCH__=3113`.

| Parameters | INI fields | Values | Differences from Server Edition |
|------|---------|:---:|------|
| NpuArch | `NpuArch` | 3113 | New schema, not in legacy mapping table |
| AIC_version | `AIC_version` | AIC-L-311 | End-side specific version identification |
| Number of cores | `ai_core_cnt` | 1 | **Single core** (server version multi-core) |
| VectorCore | `vector_core_cnt` | 1 | Single VectorCore |
| L1 | `l1_size` | 512 KB (524288) | Same capacity as DAV_3510 |
| L0A | `l0_a_size` | 32 KB (32768) | **Half the size of server version** |
| L0B | `l0_b_size` | 32 KB (32768) | **Half the size of server version** |
| L0C | `l0_c_size` | 64 KB (65536) | **Half the size of server version** |
| UB | `ub_size` | 128 KB (131072) | **Smaller than DAV_3510 (248KB)** |
| L2 | `l2_size` | 0 | **No L2 Cache** |
| BT | `bt_size` | 1 KB (1024) | Client-side capacity |
| Cube MAC array | `cube_m/n/k_size` | 16×8×16 | **Server version of 16×16×16 variant, N dimension reduced by half** |
| Sparse | `sparsity` | 1 (supports 4:2) | Architecture field |
| Core types | `core_type_list` | `AICore,VectorCore` | Cube functions are integrated in AICore |
| vec_calc_size | `vec_calc_size` | 128 | Vector calculation unit size |

> **Development Notes**:
> - The UB capacity is only 128 KB. Special attention should be paid to the UB split size when designing Tiling.
> - No L2 Cache, data transfer strategy needs to consider GM pass-through
> - Cube array N=8 (instead of 16), matrix multiplication output dimensions are different
> - Single-core design, no need for multi-core segmentation

> **About UB capacity**: The value in the table is the INI `ub_size` field, which is the user-available capacity returned by `GetCoreMemSize(CoreMemType::UB, ...)`. **The runtime is always divided into chunks based on the return value of this interface**, hard coding is prohibited.
>
> **FB (Fix Buffer)**: FixPipe quantization scale storage area. INI fields `fb0_size` / `fb1_size` / `fb2_size` / `fb3_size`; `GetCoreMemSize(FB)` returns `fb0_size`.

---

### Submodel change parameters

> **Within the same NpuArch, parameters such as the number of cores, frequency, L2, and Memory may vary with different sub-models. **
>
> `platform_config/*.ini` covers a variety of configurations (including engineering samples and simulation configurations). The ini file is a necessary but not sufficient condition for discovering hardware parameters, so **the "parameter range" is not enumerated here**. The operator code must be obtained at runtime through the `PlatformAscendC` interface.
>
> For reference, the following is a typical mass production SKU example of an architecture with archXX specialized development path (data source: corresponding ini file):
>
> | Architecture | Example Model | CubeCore | VectorCore | Frequency | L2 | Memory |
> |------|---------|:---:|:---:|:---:|:---:|:---:|
> | DAV_3510 (PCIE) | Ascend950PR_957b | 28 | 56 | 1.65 GHz | 112 MB | 112 GB |
> | DAV_3510 (Server) | Ascend950PR_9589 | 32 | 64 | 1.65 GHz | 128 MB | 128 GB |
>
> > Note: CubeCore : VectorCore = 1 : 2 in DAV_3510.
> >
>
> **Ascend950DT** (950 White Paper Table 3-1): Cube 36/32/28 core, Vector 72/64/56 core three levels; Memory 144/96 GB @ 4TB/s; L2 128MB; Cube BF16 486/432/378T, FP8 family 973/865/756T, MXFP4 1946/1730/1513T; Vector FP16/BF16 60/54/47T (three gears, each gear FP16=BF16). 950PR is 32/28 core two-tier (128/112GB @ 1.6/1.4TB/s, L2 128/112MB), which is consistent with the INI data in the above table.
>
> **Usage constraints**: The complete `full_soc` returned by `get_npu_arch.py` is an index to find specific specifications, and does not mean that the script directly measures the peak bandwidth or theoretical computing power. Core count, frequency, L2, and Memory are prioritized with runtime/INI results that exactly match the model; peak bandwidth and theoretical computing power must match credible specs. When only the PR/DT product family can be confirmed and the gear level cannot be confirmed, the precise bandwidth or computing power utilization cannot be calculated.


---

## 3. Quick review of key architectural differences

| Features | DAV_1001 | DAV_2002 | DAV_3510 | DAV_3003 (Kirin) | DAV_3113 (Kirin) |
|------|:--:|:--:|:--:|:--:|:--:|
| Core Type | AICore | AICore+VectorCore | CubeCore+VectorCore | AICore+VectorCore | AICore+VectorCore |
| Cube:Vec Ratio | N/A (No Vec) | Not 1:2 | 1:2 | N/A (Single Core) | N/A (Single Core) |
| L1 | 1 MB | 1 MB | 512 KB | 1 MB | 512 KB |
| L0A | 64 KB | 64 KB | 64 KB | 64 KB | **32 KB** |
| L0B | 64 KB | 64 KB | 64 KB | 64 KB | **32 KB** |
| L0C | 256 KB | 256 KB | 256 KB | 128 KB | 64 KB |
| UB | 256 KB | 256 KB | 248 KB | 128 KB | 128 KB |
| BT | — | — | 4 KB | 1 KB | 1 KB |
| Sparse 4:2 | — | — | Not supported | Supported | Supported |
| Cube Array | 16³ | 16³ | 16³ | **16×8×16** | **16×8×16** |

---

## 4. Specifications based on public information and experience values

The following information cannot be derived directly from the currently installed INI and instead comes from public sources or engineering experience. **Items confirmed by the [Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) (hereafter, the "950 White Paper") identify the corresponding chapter in the Information Source column**:

> **Position annotation convention**: The "Document Location" column of this table uses chapter anchors (such as `Guide §5`) and does not use line numbers to avoid reference failure caused by document line number drift.

| # | Content | Document location | Source of information |
|---|------|---------|---------|
| 1 | SIMT Register File 128KB | Guide §5 SIMT vs SIMD | Public information / experience value |
| 2 | SIMT DCache Max 128KB | Guide §5 SIMT vs SIMD | Same as above |
| 3 | SSBuffer 256KB | Guide §3 Buffer Capacity / SKILL.md §DAV_3510 hardware summary | Same as above |
| 4 | CV direct path: L0C→UB, UB→L1, SSBuffer message | Guide §4 DAV_3510 data paths / SKILL.md §DAV_3510 hardware summary | L0C→UB (including path quantization) and L1↔UB direct connection: 950 White Paper §4.1.1/§4.1.4; SSBuffer message path: public information / experience value |
| 5 | L1→GM and GM→L0A/L0B paths removed | Guide §4 Key data path changes | Public information/experience points |
| 6 | BufferID replaces set/wait synchronization | Guide §4 Instruction sequence and BufferID synchronization / SKILL.md §DAV_3510 hardware summary | 950 White Paper §4.1.6 |
| 7 | Multi-core simultaneous access to GM with the same address performance optimization | Guide §4 MTE data transfer engine | Public information / experience value |
| 8 | SIMD-Regbase: OOO instruction dual issuance | Guide §6 SIMD-Regbase | 950 White Paper §3/§4.1.2/§4.1.3 |
| 9 | Warp Scheduler 4 per AIV | Guide §5 SIMT vs SIMD | Public Information / Experience Points |
| 10 | NDDMA + ND-DMA Cache Specifications | Guide §8 NDDMA High-dimensional DMA | 950 White Paper §4.1.5 |
| 11 | CCU three communication paradigms and KFC scheduling changes | Guide §9 CCU general computing integration | CCU hardware architecture and algorithm support: 950 white paper §4.6.4; three communication paradigms and KFC scheduling details: public information / experience value |
| 12 | Vector computing power 950PR Server=54T / PCIE=47T: Baseline 27T/23.7T × Dual-Issue 2, but not all Vector instructions support dual-Issue | Guide §3 Computing power and system specifications → Vector computing power derivation | 950 White Paper Table 3-1 + §3/§4.1.2/§4.1.3 |
| 13 | Memory bandwidth Server 1.6 TB/s / PCIE 1.4 TB/s | Guide §3 Computing power and system specifications | 950 White Paper Table 3-1/§4.3.1 |
