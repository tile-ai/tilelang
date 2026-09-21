---
name: npu-arch
description: Ascend NPU hardware detection and architecture knowledge query skills. First obtain or reuse the complete SoC/NpuArch evidence of the local machine, and then determine the target platform capabilities, resource specifications, feature support and conditional compilation strategy through chip model mapping, architecture generation division and archXX feature description. A5 hardware identification, resource querying, and performance analysis for this workflow.
---

# Ascend NPU architecture knowledge

## Get local hardware evidence

This Skill is the unified entrance for this workflow to obtain hardware identity. If the caller has passed in the complete detection JSON of the same target device and the same configuration, it will be directly reused; when there is no evidence, incomplete evidence, or device or configuration changes, first parse the real directory where the current `SKILL.md` is located to `NPU_ARCH_SKILL_DIR`, and run the detection script that comes with this Skill:

```bash
python3 "$NPU_ARCH_SKILL_DIR/scripts/get_npu_arch.py" --json
```

Keep the full JSON returned by the script as `evidence`, not just extract the product family name. This workflow will only continue if the following conditions are also met:

- The script exit code is 0;
- `full_soc` completely matches `Ascend950PR` / `Ascend950DT` series;
- `npu_arch` is `3510`;
- `NpuArch inconsistency` does not exist in `warnings`.

Stop and report raw evidence when detection fails, information conflicts, or the model is not supported. A5 is not used by default, and the complete native model must not be inferred from `short_soc`, directory name, static mapping, or the Chip Name of `npu-smi`. The detection script queries the local default device; when multiple models of devices coexist, you must first confirm that the target device corresponds to the evidence, otherwise it will stop.

Pass `full_soc`, `npu_arch` and full `evidence` to subsequent design, generation, profiling and tuning phases. In the same session, it can only be reused if the target device and configuration have not changed; when any hardware-related Skill is called independently and there is no valid evidence, first load the Skill by name and execute the above process.

## Information usage granularity

- The combination of `Ascend950PR` / `Ascend950DT` series and `npu_arch=3510` is only used to confirm the support range of this workflow.
- `npu_arch=3510` is used to query the instruction set, data path and UB/L1/L0 capabilities shared with the same architecture; when runnable, priority is still given to confirming the actual value through the runtime interface.
- Complete `full_soc` is used to match SKU variation parameters such as core number, frequency, L2, Memory, peak bandwidth and theoretical computing power. Values ​​that can be obtained through the runtime interface are given priority by using real machine results; when they cannot match trusted specifications, they are marked as unconfirmed, and the typical values ​​of the same series must not be applied.
- Ascend950PR has 1.4 TB/s and 1.6 TB/s Memory bandwidth profiles, and the recorded specification of Ascend950DT is 4 TB/s; only the complete model that has been matched to the corresponding specification profile can be used for bandwidth utilization calculations.
- `432 TFLOPS` only corresponds to 32 Cube Core, Cube FP16/BF16 peak at 1.65 GHz. The 28/32/36 Cube Core file is about 378/432/486 TFLOPS; select according to the complete model or actual machine core number and frequency, and cannot be mixed with Vector or the total computing power of the whole chip.

## Architecture Generation Overview

| Concept | Description |
|-----|------|
| **NpuArch** | Chip architecture number, defines the instruction set and microarchitecture, obtained through `GetCurNpuArch()` at runtime |
| **SocVersion** | System-on-chip version, software naming identifier, obtained through `GetSocVersion()` at runtime |
| **__NPU_ARCH__** | Device side compilation macro, four-digit value, used for conditional compilation |
| **archXX** | The abbreviation of the operator warehouse directory, take the first two digits of the DAV number (such as DAV_3510 → arch35) |
| **__DAV_C310__** | Build system internal macro, equivalent to `NpuArch::DAV_3510` / `arch35` / `__NPU_ARCH__=3510`, cannot be inferred numerically |

## Complete mapping table

For complete product series / SocVersion / NpuArch / chip model mapping, see [`npu-hardware-params.md` §0 Product Mapping Table](references/npu-hardware-params.md#0-Product Mapping Table).

## Query runtime architecture and resources

```cpp
#include "utils/tiling/platform/platform_ascendc.h"

auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();         // DAV_2201 / DAV_3510 / ...
platform_ascendc::SocVersion socVer = ascendcPlatform.GetSocVersion();
```

`GetCurNpuArch()` returns `NpuArch::DAV_RESV` upon failure, `GetSocVersion()` returns `SocVersion::RESERVED_VERSION` upon failure.

These interfaces are used to confirm the runtime architecture and resources on the identified device; `GetSocVersion()` does not replace the `full_soc` probe described above, nor does it alone differentiate between all Ascend950 SKUs.

## Key changes between DAV_3510 and DAV_2201

> For detailed hardware parameter true values, see `references/npu-hardware-params.md`. The actual value must be obtained through the `PlatformAscendC` interface at runtime, and hard coding is prohibited.

### Buffer (usually consistent within the same architecture)

| Buffer | DAV_2201 | DAV_3510 |
|--------|----------|----------|
| L0C | 128 KB | 256 KB |
| UB | 192 KB | 248 KB |
| BT | 1 KB | 4 KB |

### Frequency/number of cores/L2/Memory (depends on model/form)

| Parameters | Ascend910B2 (DAV_2201) | Ascend950PR PCIE | Ascend950PR Server |
|------|------------------------|------------------|-------------------|
| Cube core count | 24 | 28 | 32 |
| Frequency | 1.8 GHz | 1.65 GHz | 1.65 GHz |
| L2 | 192 MB | 112 MB | 128 MB |
| Memory | 64 GB | 112 GB | 128 GB |

> For details, see [Typical SKU Example](references/npu-hardware-params.md#Sub-model change parameters).
>
> Memory bandwidth: 950PR has two levels of 1.6/1.4 TB/s, and the recorded specification of 950DT is 4 TB/s; it must match the specific specification file with the complete `full_soc`, and cannot be selected only by the PR/DT product family. The 950 specifications are shown in Table 3-1 of the white paper. The 910B2 is still public information and experience values. For details, see `npu-hardware-params.md` §4.

### Instruction set and microarchitecture

| Categories | Changes |
|------|------|
| Data format | Added FP8 / MXFP8 / MXFP4 / HiF8 Cube MMAD |
| CV pass-through | Added L0C→UB, UB→L1, SSBuffer messages |
| Synchronization mechanism | BufferID replaces set/wait strong pairing |
| Programming model | Added SIMT, SIMD-Regbase, NDDMA, CCU general computing integration |

## Detailed document index

`references/` is loaded on demand:

- **`npu-hardware-params.md`** — Hardware parameter reference: consistent parameters for each architecture sub-model, typical SKU examples, specifications based on public information and experience values
- **`simt-arch-guide.md`** — SIMT architecture concept reference: thread hierarchy, Warp scheduling mechanism, memory space, UB partition, data type, function call level, LAUNCH_BOUND and number of registers, SIMT and SIMD core differences
- **`npu-arch-guide.md`**:
  - **Typical hardware parameters and acquisition methods**: Typical values of core number/Buffer capacity, `GetCoreNumA*` / `GetCoreMemSize` / `aclrtGetDeviceInfo` interface usage (including hard-coded counterexamples)
  - **DAV_3510 microarchitecture**: AI Core Buffer level, MTE engine, CV data path changes, BufferID synchronization
  - **SIMT vs SIMD**: Differences in hardware capabilities and applicable scenarios
  - **SIMD-Regbase**: register group, core advantages, code form changes
  - **Data format extension**: complete data type table, C++ type name mapping, MXFP Tiling constraints
  - **NDDMA / CCU**: High-dimensional DMA usage, CCU three communication paradigms, KFC scheduling model
  - **Architecture Compatibility Checklist/Reference Sources**

## Official source

- [Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — authoritative source for the Ascend950PR/950DT (DAV_3510) specifications and microarchitecture. See `references/npu-arch-guide.md` §Reference Information Sources for a chapter map.

> **Scope Boundary**: This skill focuses on architectural judgment and hardware capability identification. Project template contents such as operator directory structure, CMake configuration, file naming convention, etc. are not within the scope of this skill.
