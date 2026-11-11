# Ascend 950 Spec Ground Truth and Experience-Based Values

> **Data source**: the [Ascend 950 NPU Architecture Whitepaper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) (authoritative source for 950PR/950DT official specs: Table 3-1 per-SKU core counts, compute, Memory, L2; Table 4-2 physical buffer capacities across the memory hierarchy) plus public-source/engineering-experience specs (§3).

---

## 1. Whitepaper Table 3-1 Spec Tiers

| Spec | 950PR (two tiers) | 950DT (three tiers) |
|------|-------------------|---------------------|
| Cube cores | 32 / 28 | 36 / 32 / 28 |
| Vector cores | 64 / 56 | 72 / 64 / 56 |
| Memory capacity | 128 / 112 GB | 144 / 96 GB |
| Memory bandwidth | 1.6 / 1.4 TB/s | 4 TB/s |
| L2 cache | 128 / 112 MB | 128 MB |
| Cube compute BF16/FP16 | 432 / 378 T | 486 / 432 / 378 T |
| Cube compute FP8 family | 865 / 757 T | 973 / 865 / 756 T |
| Cube compute MXFP4 | 1730 / 1514 T | 1946 / 1730 / 1513 T |
| Vector compute FP16/BF16 | 54 / 47 T | 60 / 54 / 47 T |

> CubeCore : VectorCore = 1:2. Whitepaper Table 3-1 does not label tiers by form factor; defer to the whitepaper for tier correspondence. 950PR total compute (Cube+Vector) BF16/FP16 = 486/425T (32-core tier / 28-core tier). Derivation methods are in [`npu-arch-guide.md` §Compute Composition and Derivation](npu-arch-guide.md#compute-composition-and-derivation).

---

## 2. UB Capacity Caliber (512 KB vs 248 KB)

The 950 whitepaper Table 4-2 states UB = **512 KB per AI Core**; the CANN INI (`ub_size`) and `GetCoreMemSize(CoreMemType::UB)` return **248 KB usable per AIV**. These do not conflict; the reconciliation is:

1. Table 4-2 counts physical capacity **per AI Core group (1 AIC + 2 AIV)**, i.e., 256 KB physical per AIV × 2;
2. Arithmetic closure: 256 KB physical per AIV − 8 KB reserved = 248 KB = 253952 B, exactly matching the INI (verified consistent across all 950PR/950DT SKUs in the local CANN installation);
3. Corroboration: "L1 512 KB per AI Core" in the same table is only self-consistent under the per-group reading (L1 lives on the AIC side, one per group; the INI `l1_size` is a per-AIC value);
4. Table 3-1 confirms Cube:Vector = 1:2 (e.g., 32/64); the 2 AIVs in each group have independent UBs.

> **★ Conclusion (budgeting rule — follow this)**: **the two AIVs' UBs are physically independent and cannot be shared** — always budget UB at **248 KB per AIV**, never as an "AI Core group 512 KB" or "2 × 248 KB = 496 KB shared pool". The whitepaper's 512 KB is the physical sum of the group's two AIVs, not capacity available to a single AIV; under MIX 1:2, the same kernel holds an independent UB and an independent TPipe on each of the two AIVs. SIMT kernels must additionally reserve ≥32 KB for the DCache (see `simt-arch-guide.md`). Tiling must always use the `GetCoreMemSize` return value; hardcoding is forbidden.

---

## 3. Public-Source and Experience-Based Specs

> **Position annotation convention**: the "Document position" column uses section anchors (e.g., Guide §SIMT vs SIMD), not line numbers, to avoid broken references from line drift. Entries confirmed by the whitepaper carry the chapter in the "Source" column.

| # | Item | Document position | Source |
|---|------|-------------------|--------|
| 1 | SIMT Register File 128 KB | Guide §SIMT vs SIMD | Public sources / experience |
| 2 | SIMT DCache up to 128 KB (reuses UB as cachelines, 128 B granularity) | Guide §SIMT vs SIMD | Public sources / experience |
| 3 | SSBuffer 256 KB | Guide §Microarchitecture Buffer Hierarchy | Public sources / experience |
| 4 | CV direct paths: L0C→UB (with in-path quantization) and L1↔UB direct connection | Guide §Microarchitecture Data Paths | 950 whitepaper §4.1.1/§4.1.4 |
| 5 | SSBuffer message path | Guide §Microarchitecture Data Paths | Public sources / experience |
| 6 | L1→GM and GM→L0A/L0B paths removed | Guide §Microarchitecture Data Paths | Public sources / experience |
| 7 | BufferID replaces set/wait synchronization (get_buf/rel_buf mutex-like semantics) | Guide §Microarchitecture BufferID | 950 whitepaper §4.1.6 |
| 8 | Multi-core same-address GM access optimization | Guide §Microarchitecture MTE | Public sources / experience |
| 9 | SIMD-Regbase: OOO dual-issue | Guide §SIMD-Regbase | 950 whitepaper §3/§4.1.2/§4.1.3 |
| 10 | Warp Scheduler: 4 per AIV | Guide §SIMT vs SIMD / SimT Guide §Warp Scheduling | Public sources / experience |
| 11 | NDDMA + ND-DMA Cache specs | Guide §NDDMA | 950 whitepaper §4.1.5 |
| 12 | CCU three communication paradigms and KFC scheduling changes | Guide §CCU | CCU hardware architecture and algorithm support: 950 whitepaper §4.6.4; paradigm and KFC details: public sources / experience |
| 13 | Dual-issue caliber in the Vector compute derivation (not all Vector instructions support dual issue) | Guide §Compute Composition and Derivation | 950 whitepaper Table 3-1 + §3/§4.1.2/§4.1.3 |
| 14 | Memory bandwidth: 950PR 1.6/1.4 TB/s, 950DT 4 TB/s | §1 Spec Tiers | 950 whitepaper Table 3-1/§4.3.1 |

---

## 4. Sub-Model Variance Notes

Within the same NpuArch (DAV_3510), core count, frequency, L2, and Memory vary by sub-model (§1 tiers are the whitepaper production specs; the CANN INI contains additional SKU configurations, including engineering samples and simulation configs). Operator code **must query values at runtime through the `PlatformAscendC` interface**; see [`npu-arch-guide.md` §Architecture Overview](npu-arch-guide.md#architecture-overview).
