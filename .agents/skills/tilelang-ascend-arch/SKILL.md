---
name: tilelang-ascend-arch
description: Ascend 950 architecture knowledge lookup skill. Uses spec tiers, microarchitecture feature descriptions, and SIMT vs SIMD capability differences to determine target platform capability, feature support, and compute composition. Use when confirming 950PR/950DT spec tiers, microarchitecture features (CV direct paths, BufferID synchronization, NDDMA, CCU), data format support, SIMT/SIMD selection, or compute composition.
---

# Ascend 950 (DAV_3510) Architecture Knowledge

## 950 Series Positioning

| Chip | Positioning |
|------|-------------|
| Ascend950PR | Targets high-performance recommendation, LLM Prefill, and multimodal inference |
| Ascend950DT | Covers the full workflow of pretraining, post-training, and inference (including Decode and Prefill) |

> (Decode)/(Prefill) indicate the primary workload focus, not exclusive usage. Both chips belong to DAV_3510; spec tiers are documented in `references/npu-hardware-params.md` §1.

## DAV_3510 Key Features

| Category | Feature |
|----------|---------|
| Data formats | FP8 / MXFP8 / MXFP4 / HiF8 Cube MMAD |
| CV direct paths | L0C→UB (with in-path quantization), UB↔L1, SSBuffer messaging |
| Synchronization | BufferID replaces strict set/wait pairing |
| Programming models | SIMT, SIMD-Regbase, NDDMA, CCU compute-communication fusion |

## Document Index

`references/` loaded on demand:

- **`npu-arch-guide.md`** — Architecture guide: 950 microarchitecture (buffer hierarchy, MTE, CV data paths, BufferID), SIMT vs SIMD, SIMD-Regbase, data format extensions, NDDMA, CCU, compute composition and derivation, architecture compatibility checklist
- **`npu-hardware-params.md`** — Whitepaper spec ground truth: 950PR/950DT core counts / compute / Memory tiers, UB capacity caliber evidence chain, public-source and experience-based specs
- **`simt-arch-guide.md`** — SIMT architecture concepts reference: thread hierarchy, warp scheduling, memory spaces, UB partitioning and DCache constraints, function call hierarchy, LAUNCH_BOUND and register counts

## Official Sources

- [Ascend 950 NPU Architecture Whitepaper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — the authoritative source for Ascend950PR/950DT official specs and microarchitecture; a chapter map is in `references/npu-arch-guide.md` §Reference Sources

> **Scope boundary**: This skill focuses on Ascend 950 architecture judgment and hardware capability identification. Engineering template content such as operator directory structure, CMake configuration, and file naming conventions is out of scope.
