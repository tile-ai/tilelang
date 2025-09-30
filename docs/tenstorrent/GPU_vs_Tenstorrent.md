# GPU (CUDA‑style) vs. Tenstorrent (TT‑Metalium)  
**Architecture & Programming Model Comparison**

This document contrasts mainstream **GPU** execution (CUDA‑style) with **Tenstorrent**’s **TT‑Metalium** programming model. It focuses on how work is assigned, how kernels are executed, how memory/staging works, and what primitives you use to build high‑performance pipelines.

> Reference for TT concepts: the public **Metalium Guide** — https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md

---

## At‑a‑Glance Comparison

| Aspect | GPU (CUDA‑style) | Tenstorrent (TT‑Metalium) |
|---|---|---|
| **Execution unit** | **Streaming Multiprocessor (SM)** executes many thread blocks; each block has warps/threads. | **Tensix core** executes a **persistent kernel**; per‑core RISC‑V controllers orchestrate **reader / compute / writer** roles. |
| **Work assignment / scheduling** | Launch many blocks; a **hardware scheduler** dynamically assigns blocks to SMs as resources free up (oversubscription is common). | **Static** partitioning on host: each core receives a fixed subset of **tile IDs** and loops over them (**no dynamic scheduler / oversubscription**). |
| **Kernel lifetime on a core** | Short‑lived blocks; SMs pick up new blocks dynamically. | **Long‑lived (“persistent”)** kernels per core; explicit outer loop over assigned work (`start_id`, `count`). |
| **Indexing / grid model** | Kernel body written against `blockIdx.{x,y,z}`, `threadIdx.{x,y,z}`; hardware provides block/thread coordinates. | Keep a **grid‑style body** (e.g., `bx, by`). For TT, a **generated outer loop** recovers `(bx, by)` from linear tile IDs assigned to that core. |
| **Granularity of compute data** | Software‑chosen tiles (e.g., 16×16, 32×32) for blocking; no mandatory DRAM tile format. | **Native 32×32 tiles**; tensors are **tilized** in DRAM to that format; cores operate tile‑by‑tile. |
| **On‑chip scratchpad** | **Shared memory** (per‑SM, user‑managed, banked). | **L1 SRAM** per core, exposed via **Circular Buffers (CBs)** created/configured by the program. |
| **Global↔on‑chip staging** | In‑kernel copies (global→shared), often with `cp.async` to overlap copy/compute. | Separate **reader kernel** streams **DRAM tiles → L1 CBs**; **compute kernel** consumes/produces tiles; **writer kernel** drains **L1 → DRAM**. |
| **Software pipelining** | `cp.async` groups, double/multi‑buffering in shared memory within a thread block. | **CB depth** (double/multi‑buffer) + split reader/compute/writer gives pipeline overlap across DRAM/L1/compute. |
| **Synchronization (intra‑block/core)** | `__syncthreads()`, warp sync primitives. | CB read/write indices, semaphores, and per‑kernel roles coordinate producer/consumer within a core. |
| **Work distribution helpers** | Streams, graphs, cooperative groups; no API to pin blocks to specific SMs. | **CoreRange / CoreRangeSet** select participating cores; helpers (e.g., “split work to cores”) compute `(start_id, count)` per core. |
| **Oversubscription / load balance** | Yes: typically launch more blocks than SMs; hardware balances dynamically. | **No oversubscription**; balance is achieved by the **static partition** of tile ranges across cores. |
| **Core‑to‑core data movement** | Usually via global/L2; limited cross‑SM features (vendor‑specific). | **On‑chip NoC** with **multicast** (e.g., write once, deliver to a rectangle of cores) for reuse patterns. |
| **Memory model (where data lives)** | Global (device DRAM), L2, shared memory, registers. | **Tiles can live in DRAM** (tilized); **L1 CBs** hold working tiles; results written back to DRAM. |
| **Typical kernel structure** | Single kernel with load→shared, compute, store; block‑local cooperation for tiling. | **Program** with **three kernels per core** (reader / compute / writer) + persistent outer loop over assigned tiles. |
| **Performance knobs** | `num_warps`, `num_stages`, shared memory size, tile shapes, vectorization, occupancy. | **CB count/depth**, tile chunk sizes (e.g., K/V chunks), **per‑core work partition** (contiguous/strided/rectangles), multicast/reuse topology. |
| **“Grid‑style” portability** | Natural, since hardware schedules blocks dynamically. | Supported by **codegen**: keep `bx,by` in the body; TT backend wraps it with a generated **static scheduler loop** per core. |

---

## Practical Notes

- **Persistent kernels on TT** mean you’ll decide **ahead of time** which set of tiles each core owns and then implement a **top‑level loop** inside the compute kernel to walk those tiles. This replaces the GPU’s hardware block scheduler.
- **Circular Buffers (CBs)** in **L1** are the central mechanism for **software pipelining**: the reader fills, compute consumes, writer drains, with depths chosen for double or multi‑buffering.
- **Multicast** lets you feed identical tiles (e.g., shared activations/weights) to a **rectangle of cores**—useful for reuse and bandwidth efficiency.
- To port GPU‑style grid code, keep the kernel body expressed against `bx, by`; then generate a **per‑core outer loop** that maps a linear tile range `(start_id..start_id+count)` back to `(by, bx)`.

---

### Reference
- Tenstorrent Metalium Guide: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
