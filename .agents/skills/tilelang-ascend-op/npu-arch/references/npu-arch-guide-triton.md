# NPU architecture generation description

This document describes the architectural generational division of Ascend NPU and its impact on operator development.

---

## Directory

1. [Architecture Generation Overview](#Architecture Generation Overview)
2. [Complete mapping table](#complete mapping table)
3. [Typical hardware parameters and acquisition methods](#Typical hardware parameters and acquisition methods)
4. [DAV_3510 Microarchitecture and Programming Impact](#dav_3510-Microarchitecture and Programming Impact)
5. [SIMT vs SIMD hardware capability differences](#simt-vs-simd-hardware capability differences)
6. [DAV_3510 Data Format Extension](#dav_3510-Data Format Extension)
7. [Architecture Compatibility Checklist](#architecturecompatibilitychecklist)
8. [Reference information source](npu-arch-guide.md#Reference information source)

---

## Architecture Generation Overview

### Core concepts

| Concept | Description |
|-----|------|
| **__NPU_ARCH__** | Device side compilation macro, four-digit value, used for conditional compilation |
| **archXX** | The abbreviation of the operator warehouse architecture directory, taking the first two digits of the DAV number, such as `arch35` |

### Architecture directory abbreviation (archXX)

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

- A target without consistent `Ascend950PR`/`Ascend950DT` and `npu_arch=3510` evidence is outside this Skill's supported workflow

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

> Note: HiF8 and MXFP8 do not have independent DtypeMKN entries in the INI, but their computing power is equivalent to E4M3FN. 950 White Paper §4.1.1 Confirmation: At the same frequency, HiF8/MXFP8/FP8 provides 2 times the FP16 tensor computing power, and MXFP4 provides 4 times; Table 3-1 rounds down the PCIE file Cube computing power to FP8 756 / MXFP4 1513 TFLOPS (the formula value is 756.9/1513.9).

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

**950PR PCIE**: 128 × 56 × 1650 × 2 ÷ 10⁶ = 23.7T, including Regbase OOO dual-issue ×2 = **47 TFLOPS**[²](#whitepaper)
**950PR Server**: 128 × 64 × 1650 × 2 ÷ 10⁶ = 27T, including Regbase OOO dual-issue ×2 = **54 TFLOPS**[²](#whitepaper)

> The 950 white paper confirms that Vector Core uses a **Dual-Issue Register-Based SIMD architecture** (§3) and supports out-of-order execution (OOO, §4.1.3). Table 3-1 lists official Vector computing power FP16/BF16 = 54/47T and FP32 = 27/23T for the Server/PCIE tiers, consistent with the above formula. Note that not all Vector instructions support dual issue.

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

> **UB caliber description ("by group" is an inference, see `npu-arch-guide.md` for the chain of evidence)**: 950 White Paper Table 4-2 Original text annotation UB = **512 KB per AI Core**; the interpretation aligned with INI is based on **AI Core group (1 AIC + 2 AIV) physical capacity** (physical 256 KB × 2 per AIV); INI `[VectorCoreSpec] ub_size = 253952` = **248 KB** is the value available per AIV user** (physical 256 KB − 8 KB reserved, exact). Circumstantial evidence: The same table "L1 512KB per AI Core" is only self-consistent if understood by groups (L1 is on the AIC side, one copy for each group). There is no contradiction between the three. Tiling is subject to the return value of `GetCoreMemSize`.


## DAV_3510 Microarchitecture and Programming Impact

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

DAV_3510 exposes three paths that allow Cube and Vector to exchange data directly and avoid GM workspace transfer (white paper §4.1.4 confirms CV direct path; §4.1.1 confirms that L0C→UB supports path-associated quantization FP32/INT32→BF16/FP16/FP8/INT8):

1. **L0C → UBuffer**: Cube results go directly to Vector, and FixPipe outputs to UB
2. **UBuffer ↔ L1**: Vector and Cube are directly connected in two directions (UB→L1 is a new direction) to avoid GM transfer
3. **SSBuffer message channel**: fine-grained synchronization signal between CVs [¹](#unverified)

**Typical benefit scenario**: FA/FIA type fusion operators can completely eliminate GM reading and writing of workspace A/B/C.

> Note: The data paths of L1→GM and GM→L0A/L0B on DAV_3510 have been deleted, and the kernel that relies on these paths needs to be modified to alternative paths (such as L1→UB→GM, GM→L1→L0A/L0B). [¹](#unverified)

### Instruction sequence synchronized with BufferID

Each execution unit on DAV_3510 has an independent instruction queue: Scalar / Cube / FixPipe / MTE1 / MTE2 / MTE3 / SIMD VF or SIMT VF.

**BufferID synchronization mechanism**: Eliminates the original set/wait forced pairing requirement and simplifies the synchronization code of multiple pipeline operators.

> **CCU General Accounting Fusion**: DAV_3510 adds a new collective communication engine on IO-Die. This article will not be expanded. For details, see `npu-arch-guide.md` §CCU General Accounting Fusion Development Model (three communication paradigms, KFC scheduling model).

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

## Architecture Compatibility Checklist

When developing operators, please confirm:

- [ ] Universal implementation tested on all target architectures
- [ ] If there is a special implementation of arch35, it has been tested separately
- [ ] Tiling logic correctly identifies the architecture and selects the implementation
- [ ] Performance meets baseline requirements on target architecture

---

## Reference information sources

### Official White Paper

- [Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) (Huawei, Ascend950PR/950DT): authoritative source for the DAV_3510 specifications and microarchitecture. See `npu-arch-guide.md` §Reference Information Sources for a chapter map.

### CANN installation package is visible (`${ASCEND_HOME_PATH}/<arch>/`)

| Documentation | Content |
|------|------|
| `asc/include/utils/tiling/platform/platform_ascendc.h` | `SocVersion` enumeration, `PlatformAscendC` interface |
| `include/platform/soc_spec.h` | `NpuArch` enumeration complete definition |

### Annotation

- <a id="unverified">¹</a> Based on public information and experience values, no directly corresponding verifiable fields have been found in the currently installed INI and 950 white paper. See `npu-hardware-params.md` §4 for details.
- <a id="whitepaper">²</a> Confirmed by "Ascend 950 NPU Architecture White Paper" (original ¹ upgrade).
