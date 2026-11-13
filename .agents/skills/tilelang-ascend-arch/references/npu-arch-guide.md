# Ascend 950 (DAV_3510) Architecture Guide

This document describes the architecture characteristics of Ascend950PR / Ascend950DT (DAV_3510) and their impact on operator development.

---

## Contents

1. [Architecture Overview](#architecture-overview)
2. [DAV_3510 Microarchitecture and Programming Impact](#dav_3510-microarchitecture-and-programming-impact)
3. [SIMT vs SIMD Hardware Capability Differences](#simt-vs-simd-hardware-capability-differences)
4. [SIMD-Regbase Architecture](#simd-regbase-architecture)
5. [DAV_3510 Data Format Extensions](#dav_3510-data-format-extensions)
6. [NDDMA High-Dimensional DMA Instructions](#nddma-high-dimensional-dma-instructions)
7. [CCU Compute-Communication Fusion Model](#ccu-compute-communication-fusion-model)
8. [Compute Composition and Derivation](#compute-composition-and-derivation)
9. [Architecture Compatibility Checklist](#architecture-compatibility-checklist)
10. [Reference Sources](#reference-sources)

---

## Architecture Overview

### Parameter Query Principle

Core counts, buffer capacities, frequency, L2, and Memory vary across sub-models and form factors within the same NpuArch, and a vNPU instance sees even lower values after partitioning. **Operators must query actual values at runtime through `PlatformAscendC` / ACL interfaces**; hardcoding typical values causes overflow or waste.

---

## DAV_3510 Microarchitecture and Programming Impact

> Only hardware capabilities that directly affect Ascend C operator programming are included. Physical form (Chiplet/POD/Server), interconnect protocols (Lingqu/UBoE), storage media (HiBL/HiZQ), and other aspects that do not affect operator code are out of scope.

### AI Core Buffer Hierarchy

Determines LocalTensor allocation, DataCopy path selection, and pipeline arrangement.

| Buffer | Purpose | Programming impact |
|--------|---------|-------------------|
| **L1 Buffer** | Cube input cache | Left/right matrix operands of matmul reside here |
| **L0A / L0B** | Cube operands | MTE1 load targets |
| **L0C Buffer** | Cube output (enlarged on 950) | Bounds the base-block tiling upper limit |
| **UB / Unified Buffer** | Vector workspace | Main arena for Vector compute; shared by SIMT/SIMD |
| **SSBuffer** | **New in DAV_3510**: CV message path, 256 KB [¹](#unverified) | Fine-grained synchronization replacing GM workspace |
| **BT / FP Buffer** | FixPipe configuration | Quantization/reordering parameters |
| **ND-DMA Cache** | NDDMA cache | Scattered-movement optimization |

> Buffer capacities are usually consistent within an architecture generation; always use the `GetCoreMemSize` return value at runtime. For the UB capacity caliber (whitepaper 512 KB vs interface 248 KB), see [`npu-hardware-params.md` §2](npu-hardware-params.md#2-ub-capacity-caliber-512-kb-vs-248-kb).

### MTE Data Movement Engines

| Engine | Paths |
|--------|-------|
| MTE1 | L1 ↔ L0A/B/C, L1 ↔ UB |
| MTE2 | GM → L1 / UB |
| MTE3 | UB → GM / L1 |

> DAV_3510 optimizes multi-core concurrent access to the same Global Memory address; partitioning templates for matmul-class operators can be simplified accordingly (no longer requiring elaborate staggering strategies to avoid same-address conflicts). [¹](#unverified)

### Key Data Path Changes on DAV_3510

DAV_3510 adds three paths that let Cube and Vector exchange data directly, avoiding GM workspace round-trips (whitepaper §4.1.4 confirms the CV direct paths):

1. **L0C → UBuffer**: Cube results reach Vector directly, with FixPipe output to UB; whitepaper §4.1.1 further confirms **L0C→UB in-path quantization** (FP32/INT32 → BF16/FP16/FP8/INT8)
2. **UBuffer ↔ L1**: bidirectional direct connection between Vector and Cube (UB→L1 is the new direction), avoiding GM round-trips
3. **SSBuffer message path**: fine-grained CV synchronization signals [¹](#unverified)

**Typical beneficiaries**: FA/FIA-style fused operators can eliminate GM reads/writes of workspaces A/B/C entirely.

> Note: on DAV_3510 the L1→GM and GM→L0A/L0B data paths have been removed; kernels that depend on them must be reworked onto alternative paths (e.g., L1→UB→GM and GM→L1→L0A/L0B). [¹](#unverified)

### Instruction Queues and BufferID Synchronization

On DAV_3510, each execution unit has its own instruction queue: Scalar / Cube / FixPipe / MTE1 / MTE2 / MTE3 / SIMD VF or SIMT VF.

**BufferID synchronization**: removes the original strict set/wait pairing requirement and simplifies synchronization code in multi-pipeline operators. Whitepaper §4.1.6 confirms: `get_buf()` corresponds to locking and `rel_buf()` to unlocking, with mutex-like semantics — more cohesive than set_flag/wait_flag and decoupled from other pipelines.

---

## SIMT vs SIMD Hardware Capability Differences

| Dimension | SIMT | SIMD |
|-----------|------|------|
| Programming paradigm | Scalar programming (thread view) | Vector programming (contiguous computation within a VF) |
| Control flow | Hardware branch scheduling via Warp Scheduler | Software-unrolled loops |
| Scattered GM access | **Direct access supported** (in-core DCache) | Not supported; data must first be moved into UB |
| Register type | Scalar registers (per thread) | Vector registers (per VF) |
| Register organization | 128 KB register file, split by thread concurrency [¹](#unverified) | Multiple VL=256 B vector registers |

**SIMT-only hardware** (new in DAV_3510): Warp Scheduler (4 per AIV) [¹](#unverified) / SIMT Register File / SIMT DCache (up to 128 KB, reusing UB as cachelines at 128 B granularity) [¹](#unverified)
**SIMD/SIMT shared**: ALU / ICache / Unified Buffer

**SIMT sweet spots**: Gather/Scatter, hash insertion, random numbers, sorting (including atomics).

**SIMT is a poor fit for**:
- Large dense BF16/FP16 matmul / convolution — SIMD + Cube pipelines are more efficient
- Long contiguous vector computation — SIMD vectorized instruction throughput is higher
- Pre-DAV_3510 architectures — no SIMT hardware exists there

On DAV_3510, SIMT offers `__global__` kernel syntax and `<<<...>>>` launch (CUDA-style); pure SIMT kernels can be dispatched directly, and a single kernel can mix SIMD VF and SIMT VF calls via `__global__ __aicore__`. Inline SIMT functions are declared with `__simt_vf__`.

---

## SIMD-Regbase Architecture

DAV_3510 introduces the Regbase architecture on the Vector unit, coexisting with the traditional Membase model (whitepaper §3 confirms the "dual-issue Register-Based SIMD architecture"; §4.1.3 confirms out-of-order execution (OOO); §4.1.2 confirms RegFile registers introduced between UB and the Vector ALU as temporary storage).

**New register groups**: VFScalar / address registers / alignment registers / mask registers / vector registers

**Key advantages**:
- **In-register computation** — reduces UB access bandwidth
- **OOO dual-issue** — improves Vector performance
- Supports **non-32B-aligned** LB data processing

**Code shape change**: Membase `block` / `repeat` parameters → explicit for-loops in Regbase.

---

## DAV_3510 Data Format Extensions

### Data Types Supported by Cube MMAD

Data types supported by the Cube MMAD unit on DAV_3510 (bold marks DAV_3510 additions):

| Format | Notes |
|--------|-------|
| FP16 / BF16 / HF32 / FP32 / S8 | General-purpose types |
| **FP8 E5M2 / E4M3** | Static/dynamic quantization |
| **MXFP8 E5M2/E4M3** | 32 data elements share 1 scale |
| **MXFP4 E2M1/E1M2** | 4-bit floating point |
| **HiF8** | Huawei-defined format |

> Note: DAV_3510 no longer supports 4:2 sparse matrix computation; kernels that relied on this for speedups must switch to dense or another supported sparsity strategy.

### MXFP Tiling Requirements

32 data elements share 1 scale. If the scale uses the same StepK as the data, its TileSize becomes too small and bandwidth utilization suffers; use a separate `scaleFactor` buffer.

---

## NDDMA High-Dimensional DMA Instructions

Combined with the ND-DMA Cache, NDDMA improves scattered access and transpose efficiency. Whitepaper §4.1.5 confirms: NDDMA can rearrange global-memory data across **up to 5 dimensions** and write it directly to UB, fusing movement with rearrangement/transpose in one step; its built-in cache automatically exploits locality, merging multiple element-granularity reads into 128-byte reads.

**Typical usage**: when the last axis is shorter than 128 B (e.g., D=16 gives only 16/64 = 25% FP32 compute efficiency), use NDDMA to transpose the D axis to a higher axis to recover parallelism before computing, then write results out via Transpose / DataCopyPad.

```text
Load (NDDMA) → Compute (last-axis reduce becomes non-last-axis reduce) → Store (Transpose + DataCopyPad)
```

---

## CCU Compute-Communication Fusion Model

DAV_3510 adds a dedicated CCU (Collective Communication Unit) engine on the IO-Die. **Impact for operator developers: compute-communication fused operators gain a new implementation option.** Whitepaper §4.6.4 confirms: the CCUA integrates MemorySlice (data storage) and a Reduce Unit (data computation); URMA movement and Reduce computation are dispatched by hardware; Broadcast / ReduceScatter / AllGather / AllReduce / All2All / All2Allv are supported; status is reported through the Mission task interface.

### Three Communication Paradigms

| Approach | Description | Compute-core usage |
|----------|-------------|--------------------|
| AIV + UBMem | Classic approach: the AIV writes UB Mem to trigger communication | Occupied |
| AIV-driven URMA | Asynchronous communication: the AIV issues URMA directly | Occupied |
| **AIV + CCU** | CCU performs synchronization/Reduce/movement | **Not occupied** |

### KFC Scheduling Model Changes

- Before: `AICore (KFC) ↔ AICPU (KFC) → SDMA`
- After: `AICore (KFC) ↔ CCU (KFC) → URMA`

### CCU Benefits for Operator Development

- Communication does not consume AI Core compute
- Communication tasks expand faster, with lower static latency from AICore issue to communication start
- On-chip buffer + Reduce unit preserve ordering: naturally zero-copy

### CCU Prerequisites

- **Architecture**: DAV_3510 only (the CCU sits on the IO-Die); not available on earlier generations
- **Algorithm**: compute-communication fused operators (e.g., AllReduce / AllGather fused with MatMul)
- **Topology**: inter-cluster communication requires URMA-capable interconnect

---

## Compute Composition and Derivation

> The official whitepaper Table 3-1 spec tiers (core counts / compute / Memory / L2) are listed in [`npu-hardware-params.md` §1](npu-hardware-params.md#1-whitepaper-table-3-1-spec-tiers). This section presents the derivation method, for understanding and cross-checking the official values.

### Cube Compute Formula Derivation

Theoretical peak Cube compute (TFLOPS) for BF16/FP16 is calculated as:

```
TFLOPS = M × K × N × cores × cube_freq(MHz) × 2 ÷ 10⁶
```

**Parameter meanings**:

| Parameter | Value | Source |
|-----------|:-----:|--------|
| M × K × N (Cube MAC array) | 16×16×16 = **4096** | INI `cube_m/n/k_size` |
| AICore count | 28 / 32 (950PR two tiers) | Whitepaper Table 3-1 |
| cube_freq | 1650 MHz | INI `[AICoreSpec] cube_freq` |
| ×2 | An FMA counts as 2 floating-point ops | 1 MAC = 1 multiply + 1 add |
| ÷10⁶ | MAC/s → TFLOPS | TFLOPS = 10¹² FLOPS = 10⁶ × 10⁶ MAC×2 |

**Worked examples**:

```
950PR 32-core tier:  4096 × 32 × 1650 × 2 ÷ 10⁶ = 432.54 TFLOPS
950PR 28-core tier:  4096 × 28 × 1650 × 2 ÷ 10⁶ = 378.47 TFLOPS
```

> **Cube vs Vector compute**: the Cube numbers above are pure Cube-unit compute and exclude Vector Core contributions. The 950PR BF16/FP16 totals of 486/425T in whitepaper Table 3-1 equal Cube (432/378) + Vector (54/47) — the DAV_3510 Vector Core **natively supports BF16** (whitepaper §4.1.2); there is no restriction that BF16 can only run on Cube.

### FP8_E4M3FN Compute Derivation

Unlike FP16/BF16, which use `cube_m/n/k_size=16×16×16=4096`, FP8_E4M3FN has a larger Cube MAC array (INI `[DtypeMKN]` section):

```
DT_FLOAT8_E4M3FN = 16,32,16  →  M×K×N = 16×32×16 = 8192
```

Substituting into the formula:

```
950PR 32-core tier:  8192 × 32 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 865 TFLOPS
950PR 28-core tier:  8192 × 28 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 757 TFLOPS
```

The FP8 family (including HiF8 and MXFP8) is all treated as 8192 MAC/cycle. MXFP4 is derived from the INT4 MKN (`DT_INT4=16,64,16`): 16384 MAC/cycle, 2× FP8.

```
950PR 32-core tier:  16384 × 32 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 1730 TFLOPS
950PR 28-core tier:  16384 × 28 cores × 1650 MHz × 2(FMA) ÷ 10⁶ = 1514 TFLOPS
```

> Note: HiF8 and MXFP8 have no independent DtypeMKN entries in the INI, but their compute is equivalent to E4M3FN. Whitepaper §4.1.1 confirms that at the same frequency HiF8/MXFP8/FP8 provide 2× FP16 tensor compute and MXFP4 provides 4×; Table 3-1 official values are FP8 family 865/756T and MXFP4 1730/1513T (the 28-core tier is officially rounded down; formula values are 756.9/1513.9).

### Vector Compute Derivation

Vector compute formula (FP16):

```
TFLOPS = vec_calc_size × vector_core_cnt × vec_freq(MHz) × 2(FMA) ÷ 10⁶
```

**950PR 28-core tier**: 128 × 56 × 1650 × 2 ÷ 10⁶ = 23.7T; with Regbase OOO dual-issue (see the SIMD-Regbase section) ×2 = **47 TFLOPS**[²](#whitepaper)
**950PR 32-core tier**: 128 × 64 × 1650 × 2 ÷ 10⁶ = 27T; with Regbase OOO dual-issue (see the SIMD-Regbase section) ×2 = **54 TFLOPS**[²](#whitepaper)

> The 950 whitepaper confirms: the Vector Core supports **dual-issue + out-of-order execution (OOO)** (architectural source in the SIMD-Regbase section), with single-core FP16/FP32 compute up 100% over the previous generation (§4.1.2); Table 3-1 official Vector compute FP16/BF16 = 54/47T and FP32 = 27/23T (two tiers; FP16 equals BF16 in each), matching the formulas above. Note that not all Vector instructions support dual issue.

---

## Architecture Compatibility Checklist

When developing an operator, confirm that:

- [ ] The generic implementation passes tests on all target architectures
- [ ] Any 950-specific implementation is tested separately
- [ ] Tiling logic correctly identifies the architecture and selects the implementation
- [ ] Performance meets baseline requirements on the target architectures

---

## Reference Sources

### Official Whitepaper

- [Ascend 950 NPU Architecture Whitepaper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) (Huawei, Ascend950PR/950DT): the authoritative source for DAV_3510 official specs and microarchitecture.
  - **Table 3-1**: per-SKU Cube/Vector core counts, Cube/Vector/total compute (MXFP4/FP8/INT8/BF16/FP16/TF32), Memory capacity and bandwidth, L2 cache capacity for 950PR/950DT
  - **Table 4-2**: physical capacities of each buffer in the memory hierarchy (L1 512KB / L0A/L0B 64KB / L0C 256KB / UB 512KB per AI Core)
  - **§4.1**: Cube/Vector Core microarchitecture (L0C→UB in-path quantization, CV direct paths, Vector dual-issue Regbase + OOO, native Vector BF16, SIMD/SIMT mixed programming)
  - **§4.1.5/§4.1.6**: NDDMA (up to 5-dimensional rearrangement, built-in cache, 128B read merging), BufferID synchronization (`get_buf()`/`rel_buf()`)
  - **§4.3**: high-speed on-chip memory (950PR 128GB/1.6TB/s, 950DT 144GB/4TB/s), L2 cache (128MB UMA, 512B cache lines, 4×128B sectors, L2 hints, CMO)
  - **§4.6.4**: CCU collective communication engine (URMA movement + Reduce Unit + MemorySlice, supporting Broadcast/ReduceScatter/AllGather/AllReduce/All2All/All2Allv)

### Visible in the CANN Installation (`${ASCEND_HOME_PATH}/<arch>/`)

| Location | Content |
|----------|---------|
| `include/platform/soc_spec.h` | Complete `NpuArch` enum definition |
| `asc/include/utils/tiling/platform/platform_ascendc.h` | `SocVersion` enum, `PlatformAscendC` interface (`GetCurNpuArch` / `GetSocVersion` / `GetCoreNumAic` / `GetCoreNumAiv` / `GetCoreMemSize`) |
| `<arch>/data/platform_config/Ascend950*.ini` | Per-SKU buffer capacities, core counts, frequency, L2, Memory, sparsity, DtypeMKN for 950PR/950DT |
| `ccec_compiler` (`--cce-aicore-arch`) | Compiler arch-token |
| `include/acl/acl_rt.h` | Host-side core-count queries via `aclrtGetDeviceInfo` |
| `asc/include/simt_api/`, `c_api/`, `basic_api/` | SIMT/SIMD APIs, data type names, special-value macros |

### Annotations

- <a id="unverified">¹</a> Based on public sources and experience; no directly verifiable field has been found in the currently installed INI files or the 950 whitepaper. See [`npu-hardware-params.md` §3](npu-hardware-params.md#3-public-source-and-experience-based-specs).
- <a id="whitepaper">²</a> Confirmed by the [Ascend 950 NPU Architecture Whitepaper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf); chapter references are in the Official Whitepaper section above.
