# NPU architecture generation description

This document describes the architectural generational division of Ascend NPU and its impact on operator development.

---

## Directory

1. [Architecture Generation Overview](#Architecture Generation Overview)
2. [Complete mapping table](#complete mapping table)
3. [Typical hardware parameters and acquisition methods](#Typical hardware parameters and acquisition methods)
4. [DAV_3510 Microarchitecture and Programming Impact](#dav_3510-Microarchitecture and Programming Impact)
5. [SIMT vs SIMD hardware capability differences](#simt-vs-simd-hardware capability differences)
6. [SIMD-Regbase Architecture](#simd-regbase-architecture)
7. [DAV_3510 Data Format Extension](#dav_3510-Data Format Extension)
8. [NDDMA high-dimensional DMA instruction](#nddma-high-dimensional-dma-instruction)
9. [CCU Tongsuan Integrated Development Model](#ccu-Tongsuan Integrated Development Model)
10. [Architecture Compatibility Checklist](#architecturecompatibilitychecklist)
11. [Reference Information Source](#Reference Information Source)

---

## Architecture Generation Overview

### Core concepts

| Concept | Description |
|-----|------|
| **NpuArch** | Chip architecture number, defines the instruction set and microarchitecture, obtained through `GetCurNpuArch()` at runtime |
| **SocVersion** | System-on-chip version, software naming identifier, obtained through `GetSocVersion()` at runtime |
| **__NPU_ARCH__** | Device side compilation macro, four-digit value, used for conditional compilation |
| **archXX** | The abbreviation of the operator warehouse architecture directory, taking the first two digits of the DAV number, such as `arch35` |

### Architecture directory abbreviation (archXX)

Directories divided by architecture in the operator warehouse are named using `archXX`, taking the first two digits of `DAV_XXXX`:

| Catalog | Corresponding to NpuArch | Chip |
|------|-------------|------|
| **arch35** | DAV_3510 | Ascend950DT / Ascend950PR |

> Naming rule: `archXX` = `arch` + the first two digits of the DAV number. For the supported A5 architecture, DAV_**35**10 → arch35.

### Architecture code alias

| Codename | Corresponds to SocVersion | Corresponds to NpuArch | Description |
|-----|----------------|-------------|------|
| **A5** | ASCEND950 | DAV_3510 | Ascend950DT (Decode) / Ascend950PR (Prefill) |

**Model relationship**

The `ASCEND950` / `DAV_3510` architecture identity covers multiple Ascend950PR and Ascend950DT SKUs. Architecture-level capabilities come from `npu_arch=3510`, while SKU-dependent resources and peaks must be selected with the complete `full_soc` and trusted specifications.

> **Note**: For NPU core operator development, it is usually not necessary to be aware of the specific SocVersion. Using NpuArch to distinguish chips is beneficial to ease of use and maintainability.

### Key details

- `DAV_RESV` is the error return value of `GetCurNpuArch()`: returned when acquisition fails, string conversion fails, or value <= 0
- `RESERVED_VERSION` is the error return value of `GetSocVersion()`; neither failure value may be treated as A5 evidence

---

## Complete mapping table

For complete product series / SocVersion / NpuArch / chip model mapping, see [`npu-hardware-params.md` §0 Product Mapping Table](npu-hardware-params.md#0-Product Mapping Table).

---

## Typical hardware parameters and acquisition methods

> ⚠️ **Core Principle**: This section serves as the upstream true source of NPU architecture knowledge in the skills warehouse. Downstream skills should consume the data in this section. **Reverse self-reference is considered a risk**.
>
> The table shows **typical specification values** for selected A5 SKUs. Other Ascend950PR/DT SKUs and vNPU instances may expose different resources. **The actual value must be obtained through the interface below when the operator is running**. Hard-coding typical values will cause out-of-bounds access or wasted capacity.
>
> For complete parameter reference (architecture constants/typical SKUs/based on public information and experience values), see `npu-hardware-params.md`.

### Computing power and system specifications

| Specifications | Ascend950PR PCIE (DAV_3510) | Ascend950PR Server (DAV_3510) |
|--------|:---:|:---:|
| CubeCore core count | **28** | **32** |
| Frequency (GHz) | 1.65 | 1.65 |
| Cube computing power BF16/FP16 | 378T | 432T |
| Cube computing power FP8/HiF8/MXFP8 | 757T | 865T |
| Cube computing power MXFP4 | 1514T | 1730T |
| Vector computing power FP16 | 47T [²](#whitepaper) | 54T [²](#whitepaper) |
| Memory capacity (GB) | 112 | 128 |
| Memory bandwidth | 1.4 TB/s [²](#whitepaper) | 1.6 TB/s [²](#whitepaper) |

> **Contrast with White Paper Table 3-1**: The official specifications of 950PR are Cube 32/28 core and Vector 64/56 core, which are consistent with the two levels of Server/PCIE in this table (PCIE/Server naming comes from INI SKU, and the white paper is not marked according to the form); the derived values of computing power, memory, and L2 in the table are consistent with the official values, only the PCIE of FP8/MXFP4 Files are officially marked by rounding down (see FP8 derivation note).

> **Note**: The above are typical values for selected submodels (true source: `platform_config/*.ini`). **Core count, frequency, L2, Memory, and peak values may differ among other Ascend950PR/DT SKUs.** For details, see `npu-hardware-params.md` for typical SKU examples.

#### Cube calculation power formula derivation

Cube BF16/FP16 theoretical peak computing power (TFLOPS) is calculated by the following formula:

```
TFLOPS = M × K × N × number of cores × cube_freq(MHz) × 2 ÷ 10⁶
```

**Parameter meaning**:

| Parameter | Value | Source |
|------|:--:|------|
| M × K × N (Cube MAC array) | 16×16×16 = **4096** | INI `cube_m_size=cube_k_size=cube_n_size=16` |
| AICore core number | 28(PCIE) / 32(Server) | INI `[SoCInfo] ai_core_cnt` |
| cube_freq | 1650 MHz for the listed 950PR SKUs | INI `[AICoreSpec] cube_freq` |
| ×2 | FMA counts as 2 floating point operations | 1 MAC = 1 multiply + 1 add |
| ÷10⁶ | MAC/s → TFLOPS | TFLOPS = 10¹² FLOPS = 10⁶ × 10⁶ MAC×2 |

**Derivation Example**:

```
950PR Server:  4096 × 32 × 1650 × 2 ÷ 10⁶ = 432.54 TFLOPS
950PR PCIE:    4096 × 28 × 1650 × 2 ÷ 10⁶ = 378.47 TFLOPS
```

> **Cube computing power and Vector computing power**: The Cube computing power in the above table is pure Cube unit computing power, excluding Vector Core contribution. Vector computing power is listed separately, total chip computing power = Cube + Vector. The total computing power of 950PR BF16/FP16 given in Table 3-1 of the 950 white paper is 486/425T, which is the sum of Cube(432/378) + Vector(54/47) - the Vector Core of DAV_3510 has **native support for BF16** (white paper §4.1.2), and there is no restriction that BF16 can only use Cube.

#### FP8_E4M3FN computing power derivation

Unlike FP16/BF16 which uses `cube_m/n/k_size=16×16×16=4096`, the Cube MAC array of FP8_E4M3FN is larger (INI `[DtypeMKN]` section):

```
DT_FLOAT8_E4M3FN = 16,32,16  →  M×K×N = 16×32×16 = 8192
```

Substitute into the formula:

```
950PR Server: 8192 × 32 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 865 TFLOPS
950PR PCIE: 8192 × 28 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 757 TFLOPS
```

The FP8 family (including HiF8, MXFP8) is regarded as 8192 MAC/cycle. MXFP4 is derived according to the MKN of INT4 (`DT_INT4=16,64,16`): 16384 MAC/cycle, 2 times that of FP8.

```
950PR Server: 16384 × 32 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 1730 TFLOPS
950PR PCIE: 16384 × 28 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 1514 TFLOPS
```

> Note: HiF8 and MXFP8 do not have independent DtypeMKN entries in the INI, but their computing power is equivalent to E4M3FN. 950 White Paper §4.1.1 Confirmation: HiF8/MXFP8/FP8 provides 2 times the FP16 tensor computing power at the same frequency, and MXFP4 provides 4 times; Table 3-1 Official values FP8 family 865/756T, MXFP4 1730/1513T (PCIE file is rounded down, the formula value is 756.9/1513.9).

#### Vector calculation power derivation

Vector computing power formula (FP16):

```
TFLOPS = vec_calc_size × vector_core_cnt × vec_freq(MHz) × 2(FMA) ÷ 10⁶
```

| Parameters | 950PR PCIE | 950PR Server | Source |
|------|:---:|:---:|------|
| vec_calc_size | 128 | 128 | INI `[AICoreSpec] vec_calc_size` |
| vector_core_cnt | 56 | 64 | INI `[SoCInfo] vector_core_cnt` |
| Frequency (MHz) | 1650 | 1650 | INI `[VectorCoreSpec] vec_freq` |

**950PR PCIE**: 128 × 56 × 1650 × 2 ÷ 10⁶ = 23.7T, including Regbase OOO dual-issue (see SIMD-Regbase section) and ×2 = **47 TFLOPS**[²](#whitepaper)
**950PR Server**: 128 × 64 × 1650 × 2 ÷ 10⁶ = 27T, including Regbase OOO dual-issue (see SIMD-Regbase section) and ×2 = **54 TFLOPS**[²](#whitepaper)

> The 950 white paper confirms that Vector Core supports **dual launch + out-of-order execution (OOO)** (see the SIMD-Regbase section for the architecture source). Table 3-1 lists official Vector computing power FP16/BF16 = 54/47T and FP32 = 27/23T for the Server/PCIE tiers, consistent with the above formula. Note that not all Vector instructions support dual issue.

### AIV (Vector) core number

For the supported DAV_3510 `CubeCore,VectorCore` architecture, each Cube Core is paired with 2 Vector Cores.

**The actual value is subject to `GetCoreNumAiv()`**, some SKU or vNPU instances may be cut.

### Buffer capacity (per AI Core)

| Buffer | Ascend950PR | Purpose |
|--------|:---:|------|
| **L1** | 512 KB | Cube input cache |
| **L0A** | 64 KB | Cube left matrix operand |
| **L0B** | 64 KB | Cube right matrix operand |
| **L0C** | **256 KB** | Cube output |
| **UB** | **248 KB** | Vector workspace, separate copy for each AIV |
| **L2** | **128 MB** (Server) / **112 MB** (PCIE) | Cross-core shared cache |
| **BT** (biasSize) | **4 KB** | FixPipe Bias table |
| **SSBuffer** | 256 KB [¹](#unverified) | AIC↔AIV inter-core message path |

> L1/L0A/L0B/L0C/UB/BT are usually consistent within the same generation architecture, L2 and Memory may vary by model. The runtime will always be based on `GetCoreMemSize`.

**About UB capacity**: 248 KB in the table is the return value of `GetCoreMemSize(CoreMemType::UB, ...)` on the listed A5 SKUs. The specific value is subject to the interface return to avoid hard coding.

> **Caliber description ("by group" is an inference, the evidence chain is as follows)**: 950 White Paper Table 4-2 The original text annotation UB = **512 KB per AI Core**; the interpretation aligned with INI (available 248 KB per AIV) is: Table 4-2 is based on **AI Core group (1 AIC + 2 AIV) physical capacity**, that is, physical 256 KB × 2 per AIV. Evidence chain:
> 1. Local CANN INI `[VectorCoreSpec] ub_size = 253952` = **248 KB** for all 950PR/950DT SKUs (available value per AIV user, i.e. `GetCoreMemSize(UB)` return value);
> 2. Arithmetic closed loop: 256 KB − 8 KB reserved per AIV physics = 248 KB = 253952 B, exactly consistent with INI;
> 3. Circumstantial evidence: The same table "L1 512KB per AI Core" is self-consistent only if understood by "group" (L1 is on the AIC side, one copy for each group, INI `l1_size`=512KB per AIC);
> 4. Table 3-1 confirms that Cube:Vector = 1:2 (such as 32/64), and each group of 2 AIVs has an independent UB.
>
> There is no contradiction between the three: **Single AIV physics 256 KB / available 248 KB; group-level physics 512 KB**. SIMT scenarios also need to make way for DCache ≥32KB (see `simt-arch-guide.md`). Tiling is always based on the return value of `GetCoreMemSize`.

### Kernel/Tiling side acquisition (recommended)

```cpp
#include "utils/tiling/platform/platform_ascendc.h"

auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

//Number of cores
uint32_t aicNum = ascendcPlatform.GetCoreNumAic(); // Cube core number
uint32_t aivNum = ascendcPlatform.GetCoreNumAiv(); // Vector core number

// Buffer capacity (user available value)
uint64_t ubSize, l1Size, l0aSize, l0bSize, l0cSize, l2Size, btSize;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,   ubSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,   l1Size);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,   l2Size);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT,   btSize);
```

| Computing unit used by the operator | Core interface |
|---------|------------|
| Vector (Add/Mul/Reduce, etc.) | `GetCoreNumAiv()` |
| Cube (MatMul/Conv, etc.) | `GetCoreNumAic()` |
| Cube + Vector (fusion operator) | Take both and use them for respective blocks |

### Counterexample: hard-coded hardware parameters

```cpp
// ❌ Error: Hard-coded typical values will be out of bounds or wasted across models or cropped SKUs
constexpr uint32_t CORE_NUM = 32;
constexpr uint32_t UB_SIZE  = 248 * 1024;
SetBlockDim(CORE_NUM);
pipe.InitBuffer(queue, 2, UB_SIZE / 2);

// ✅ Correct: Get at runtime
uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
uint64_t ubSize;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
SetBlockDim(coreNum);
pipe.InitBuffer(queue, 2, ubSize / 2);
```

### Host Runtime (ACL) side obtains the number of cores

Called after `aclrtSetDevice` in direct mode:

```cpp
int64_t coreNum;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum); // Pure Vector operator
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &coreNum); // Matrix operator
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM, &coreNum); // Mixed operator
```
---

## DAV_3510 Microarchitecture and Programming Impact

> Only hardware capabilities that have a direct impact on Ascend C operator programming are included. Physical form (Chiplet/POD/Server), interconnection protocol (Lingqu/UBoE), storage media (HiBL/HiZQ) and other contents that do not affect the operator code are not included here.

### AI Core Buffer Level

Determine the LocalTensor allocation location, DataCopy path selection, and pipeline orchestration.

| Buffer | Purpose | Programming Impact |
|--------|-----|---------|
| **L1 Buffer** | Cube input buffer | Matrix multiplication left/right matrix resident |
| **L0A / L0B** | Cube operand | MTE1 move target |
| **L0C Buffer** | Cube output (**Ascend950PR expansion**) | Affects the basic block Tiling upper limit |
| **UB / Unified Buffer** | Vector workspace | Vector computing main battlefield, SIMT/SIMD sharing |
| **SSBuffer** | **DAV_3510 New**: Inter-CV message path | Replaces fine-grained synchronization of GM workspace |
| **BT / FP Buffer** | FixPipe Configuration | Quantization/Rearrangement Parameters |
| **ND-DMA Cache** | NDDMA cache | Discrete transfer optimization |

### MTE data transfer engine

| engine | path |
|-----|------|
| MTE1 | L1 ↔ L0A/B/C, L1 ↔ UB |
| MTE2 | GM → L1 / UB |
| MTE3 | UB → GM / L1 |

> DAV_3510 has optimized the performance of the scenario where multiple cores simultaneously access Global Memory with the same address. The core split template of the matrix multiplication related operator can be simplified accordingly (it is no longer necessary to design complex misalignment strategies to avoid conflicts with the same address). [¹](#unverified)

### DAV_3510 data paths relevant to operators

DAV_3510 exposes three paths that allow Cube and Vector to exchange data directly and avoid GM workspace transfer (white paper §4.1.4 confirms CV direct path):

1. **L0C → UBuffer**: Cube results go directly to Vector, and FixPipe outputs to UB; White Paper §4.1.1 further confirms support for **L0C → UB path quantization** (FP32/INT32 → BF16/FP16/FP8/INT8)
2. **UBuffer ↔ L1**: Vector and Cube are directly connected in two directions (UB→L1 is a new direction) to avoid GM transfer
3. **SSBuffer message channel**: fine-grained synchronization signal between CVs [¹](#unverified)

**Typical benefit scenario**: FA/FIA type fusion operators can completely eliminate GM reading and writing of workspace A/B/C.

> Note: The data paths of L1→GM and GM→L0A/L0B on DAV_3510 have been deleted, and the kernel that relies on these paths needs to be modified to alternative paths (such as L1→UB→GM, GM→L1→L0A/L0B). [¹](#unverified)

### Instruction sequence synchronized with BufferID

Each execution unit on DAV_3510 has an independent instruction queue: Scalar / Cube / FixPipe / MTE1 / MTE2 / MTE3 / SIMD VF or SIMT VF.

**BufferID synchronization mechanism**: Eliminates the original set/wait forced pairing requirement and simplifies the synchronization code of multiple pipeline operators. White paper §4.1.6 confirms: `get_buf()` corresponds to locking, `rel_buf()` corresponds to unlocking, similar to mutex lock semantics, more cohesive than set_flag/wait_flag, and decoupled from other pipelines.

---

## SIMT vs SIMD hardware capability differences

| Dimensions | SIMT | SIMD |
|-----|------|------|
| Programming Paradigms | Scalar Programming (Threading Perspective) | Vector Programming (Continuous Computation within VF) |
| Control logic | Warp Scheduler hardware branch scheduling | Software expansion loop |
| GM discrete access | **Supports direct access** (in-core DCache) | Not supported, needs to be moved into UB first |
| Register type | Scalar register (per-Thread) | Vector register (per-VF) |
| Register organization | 128 KB register file, divided according to the number of thread concurrency [¹](#unverified) | Multiple VL=256 B vector registers |

**SIMT unique hardware** (new in DAV_3510): Warp Scheduler (4 per AIV) [¹](#unverified) / SIMT Register File / SIMT DCache (maximum 128KB, reuse UB as Cacheline, 128B granularity) [¹](#unverified)
**SIMD/SIMT sharing**: ALU / ICache / Unified Buffer

**SIMT applicable scenarios**: Gather/Scatter, Hash insertion, random numbers, sorting (including atomic operations).

**SIMT is not applicable to scenarios**:
- Large block dense BF16/FP16 matrix multiplication/convolution - SIMD + Cube pipeline is more efficient
- Long vector sequential calculations - SIMD vectorized instructions for higher throughput
- A target without confirmed `DAV_3510` evidence is outside this Skill's supported SIMT scope

SIMT on DAV_3510 provides `__global__` kernel function syntax and `<<<...>>>` startup mode (equivalent to CUDA style). Pure SIMT mode can be directly scheduled; SIMD VF and SIMT VF can be mixedly called through `__global__ __aicore__` in the same kernel function. Inline SIMT functions are declared using `__simt_vf__`.

---

## SIMD-Regbase Architecture

DAV_3510 introduces the Regbase architecture on the Vector unit to coexist with the traditional Membase (white paper §3 confirms the "new dual-emission Register-Based SIMD architecture", §4.1.3 confirms out-of-order execution (OOO), §4.1.2 confirms the introduction of the RegFile register between UB and Vector ALU for temporary storage).

**New register group**: VFScalar / Address register / Alignment register / Mask register / Vector register

**Core Advantages**:
- **In-Register Computation** — Reduce UB access bandwidth
- **OOO command double issuance** — Vector performance improvement
- Support LB **non-32B aligned** data processing

**Code form changes**: Membase’s `block` / `repeat` parameters → Regbase’s for-loop explicit loop.


---

## DAV_3510 data format extension

### Data types supported by Cube MMAD

Data types supported by the Cube MMAD computing unit on DAV_3510 (bold fonts are new for DAV_3510):

| Format | Remarks |
|-----|------|
| FP16 / BF16 / HF32 / FP32 / S8 | Universal type |
| **FP8 E5M2 / E4M3** | Static/dynamic quantization |
| **MXFP8 E5M2/E4M3** | 32 Data shared 1 Scale |
| **MXFP4 E2M1/E1M2** | 4-bit floating point |
| **HiF8** | Huawei custom format |

> Note: DAV_3510 no longer supports 4:2 sparse matrix calculations. The kernel that originally relied on this feature to speed up needs to be changed to dense or other supported sparse strategies.

### Type name mapping

| Format | C++ type name |
|-----|----------|
| FP8 E5M2 | `fp8_e5m2_t` |
| FP8 E4M3FN | `fp8_e4m3fn_t` |
| HiFloat8 | `hifloat8_t` |
| INT4 | `int4b_t` (Note: used for Vector/weight storage, **not** supported by Cube MMAD) |

### MXFP type’s special requirements for Tiling

32 Data share 1 Scale. If Scale uses the same StepK as Data, the TileSize of Scale will be too small and the bandwidth utilization will be low. An independent `scaleFactor` cache needs to be used.

---

## NDDMA high-dimensional DMA instruction

Cooperate with ND-DMA Cache to improve discrete access and transposition efficiency. White paper §4.1.5 Confirmation: NDDMA can rearrange global memory data in **up to 5 dimensions** and write it directly to UB. The transfer and rearrangement/transposition are completed in one step; the built-in cache automatically explores data locality and merges reads of multiple data element granularities into 128-byte reads.

**Typical usage**: When the last-axis length < 128B (such as D=16, FP32 calculation efficiency is only 16/64 = 25%), use NDDMA to transpose the D axis to the high axis to achieve sufficient parallelism and then calculate, and finally move it out through Transpose / DataCopyPad.

```text
Move in (NDDMA) → Compute (last axis reduce to non-last axis reduce) → Move out (Transpose + DataCopyPad)
```

---

## CCU Tongsuan Fusion Development Model

DAV_3510 adds a CCU (Collective Communication Unit) dedicated communication engine on IO-Die. **Impact on operator developers: Tongsuan fusion operators have new development options. ** White paper §4.6.4 Confirmation: CCUA integrates MemorySlice (data storage) and Reduce Unit (data calculation). URMA transfer and Reduce calculation are executed by hardware judgment. It supports Broadcast / ReduceScatter / AllGather / AllReduce / All2All / All2Allv typical algorithms. After completion, the status is reported through the Mission task interface.

### Three communication paradigms

| Method | Description | Calculate core occupancy |
|-----|------|----------|
| AIV + UBMem | Classic way, AIV writes UB Mem to trigger communication | Occupy |
| AIV directly drives URMA | Asynchronous communication, AIV directly initiates URMA | Occupy |
| **AIV + CCU** | CCU completes synchronization/Reduce/transportation | **Not occupied** |

### KFC scheduling model changes

- Original: `AICore (KFC) ↔ AICPU (KFC) → SDMA`
- New: `AICore (KFC) ↔ CCU (KFC) → URMA`

### CCU’s advantages for operator development

- Communication does not occupy AI Core computing power
- Communication tasks unfold faster, and the static delay from AICore initiation to the start of communication is smaller
- On-chip buffer + Reduce unit order preservation, natural zero copy

### CCU uses preconditions

- **Architecture**: DAV_3510 only, with CCU on IO-Die
- **Algorithm**: general fusion operator (such as AllReduce/AllGather and MatMul fusion)
- **Topology**: Inter-cluster communication requires hardware interconnection that supports URMA

---

## Architecture Compatibility Checklist

When developing operators, please confirm:

- [ ] Universal implementation tested on all target architectures
- [ ] If there is a special implementation of arch35, it has been tested separately
- [ ] Tiling logic correctly identifies the architecture and selects the implementation
- [ ] Performance meets baseline requirements on target architecture

---

## Reference information sources

### Official White Paper

- [Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) (Huawei, Ascend950PR/950DT): authoritative source for the DAV_3510 specifications and microarchitecture.
  - **Table 3-1**: Number of Cube/Vector cores, Cube/Vector/total computing power (MXFP4/FP8/INT8/BF16/FP16/TF32), Memory capacity and bandwidth, and L2 Cache capacity of each SKU of 950PR/950DT
  - **Table 4-2**: Physical capacity of each Buffer at the Memory level (L1 512KB / L0A/L0B 64KB / L0C 256KB / UB 512KB per AI Core)
  - **§4.1**: Cube/Vector Core microarchitecture (L0C→UB path quantization, CV pass-through, Vector dual-issue Regbase + OOO, Vector native BF16, SIMD/SIMT hybrid programming)
  - **§4.1.5/§4.1.6**: NDDMA (up to 5-dimensional rearrangement, built-in cache, 128B read merging), BufferID synchronization mechanism (`get_buf()`/`rel_buf()`)
  - **§4.3**: High-speed on-chip memory (950PR 128GB/1.6TB/s, 950DT 144GB/4TB/s), L2 Cache (128MB UMA, 512B CacheLine, 4×128B Sector, L2 Hint, CMO)
  - **§4.6.4**: CCU collective communication engine (URMA handling + Reduce Unit + MemorySlice, supports Broadcast/ReduceScatter/AllGather/AllReduce/All2All/All2Allv)

### CANN installation package is visible (`${ASCEND_HOME_PATH}/<arch>/`)

| Documentation | Content |
|------|------|
| `asc/include/utils/tiling/platform/platform_ascendc.h` | `SocVersion` enumeration, `PlatformAscendC` interface (`GetCurNpuArch` / `GetSocVersion` / `GetCoreMemSize`, etc.) |
| `include/platform/soc_spec.h` | `NpuArch` enumeration complete definition |

### Annotation

- <a id="unverified">¹</a> Based on public information and experience values, no directly corresponding verifiable fields have been found in the currently installed INI and 950 white paper. See `npu-hardware-params.md` §4 for details.
- <a id="whitepaper">²</a> has been published by "[Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)" confirmed (original ¹ item upgrade), please see the "Official White Paper" in this section for the corresponding chapters.
