# TileScale: Tile-based AI Compute at All Scales

TileScale is a distributed extension of TileLang. It expands TileLang's tile-level programming to multi-GPU, multi-node, and even distributed chip architecture scopes, with some new feature designs like tile-level communication and hierarchical programming introduced.

TileScale is a distributed-native domain-specific language (DSL) and compiler stack designed for deep learning on next-generation distributed architectures.
As AI model entering the "scaling-law" era, modern AI infrastructure is also scaling the computation across both intra-chip and inter-chip scopes. On one side, current large AI models are already executing on multiple GPUs or even multiple nodes connected by the high-performance links like NVLink or InfiniBand. On the other side, a bunch of next-gen AI accelerators are embracing new chip architectures—such as 3D IC, near/in-memory computing, wafer-scale accelerators, etc., which are all in distributed form inner the chip for better scalability. Together, these trends are shaping modern AI compute systems into a hybrid, multi-level of "distributed architecture".

TileScale is the first programming and compiler stack to unify these intra-chip and inter-chip compute resources into a unified, hierarchical, distributed architecture, which virtualizes the whole distributed system as a unified "mega-device" to users. To facilitate programming, TileScale provides a set of consistent tile-level primitives across all hardware layers for compute, memory, and communication. Thus, users can just write tile-level computing logic or flow at certain layers of interest, then TileScale automatically compiles and optimizes the scheduling of computation, communication, memory access, and their overlap. The goal of TileScale is to define an open, streamlined programming model for future distributed architectures and systems, addressing the emerging needs of modern AI computation, such as fine-grained computation and communication overlap, flexible parallel mechanisms, dataflow computation, NUMA programming, etc.

## The full technical white-paper is coming soon.

## Hierarchical Distributed Architecture (HDA)
Unlike traditional GPU SIMT programming, which assumes thread-level computation on a single device, TileScale is designed to manage compute, memory, and communication across all hierarchical scales, from threads and PEs to dies, chips, and nodes. It introduces a unified virtual device architecture, called Hierarchical Distributed Architecture (HDA), to abstract these distributed systems.
![](./images/arch.png "Hierarchical Distributed Architecture(HDA)")

HDA is built upon three fundamental resources: *compute units, memory, and network*.
Those resources can be logically organized into hierarchical groups, which provide different scales of computation capability. For example, on a GPU, the smallest granularity is a thread-scale. Threads can be grouped into a warp (e.g., 32 threads), which executes warp-scale operations. These warp-scale compute units (e.g., tensor cores) and thread-scale units (e.g., CUDA cores) are further organized into an SM-scale unit, capable of executing thread block tasks. The number of scale levels and naming of each scale are hardware-defined and can vary across architectures.

HDA contains multiple memory layers. Each layer can be either shared or distributed to individual compute unit. For example, the L1 cache or shared memory is accessible to all threads within a thread-block, while register memory can only be accessed by individual threads. Note that a compute unit at a certain scale can access different layer of memory.

<!-- For instance, on GPUs, the L1 cache or shared memory is accessible to all threads within a block (SM-level scope), whereas DSMEM (distributed shared memory) is accessible across SMs within a cluster-level scope. A compute unit at a given scope has control over all memory layers within that scope. For example, an SM-level task can access both thread registers and shared memory. -->

Parallel units at the same level scale can be interconnected via a network. For example, in NVIDIA Hopper GPUs, SMs within a CTA cluster are interconnected via a NoC (Network-on-Chip), enabling peer SM memory access. Similarly, multiple GPUs within a node can be connected using NVLink to support inter-GPU communication.

This hierarchical structure of compute, memory, and network forms the backbone of HDA, enabling scalable and programmable execution across complex, distributed AI systems.

<!-- In HDA, each top-level compute unit is associated with an L0 memory (e.g., a CUDA core with few registers). A group of such units then forms a higher-level compute unit with L1 memory—for example, a Streaming Multiprocessor (SM) containing multiple CUDA cores with shared memory. This hierarchical composition can be extended to form SM clusters, GPUs, nodes, super-nodes, and beyond, each with its corresponding level of memory.

At each layer, the associated memory may be shared among all units or distributed to individual units. Compute units or groups of units at the same level can be interconnected via a dedicated network. For instance, in Hopper GPUs, SM clusters are connected via a Network-on-Chip (NoC), enabling DSMem programming capabilities. Similarly, GPUs within a single node are interconnected using an NVLink switch. -->

## Tile-based Programming Interface
Following the hierarchical hardware architecture, TileScale exposes a hierarchical programming interface. The fundamental unit of computation in TileScale is at the *tile* granularity. TileScale provides consistent tile-level compute, memory, and communication operators corresponding to each hardware scales.
<div align="center">    <img src="./images/interface.png" alt="TileScale Programming Interface" width=80% />
</div>

- *Compute*: A compute primitive takes input tensor tiles at certain memory layer and produces output tensor tiles. The same compute primitive can be used at different scale level, which will be translated to different implementations. A primitive at a high-level scale can be implemented by the lower-level-scale primitives. For example, a block-scale operator can be implemented by a group of warp-scale or thread-scale primitives.

- *Memory*: The memory primitives are used to copy data tiles at certain memory layer, as well as to copy data tile between different memory layers.

- *Communicate*: The communication primitives are used to transfer data tiles between compute units over the network, as well as to manage the synchronization. TileScale provides both basic peer-to-peer communication primitives as well as the collective communication primitives like AllReduce, All2All, etc., at a specific scale level.

A primitive for a certain scale level may have multiple implementations. For example, a copy primitive could be implemented using TMA or LSU, while a remote copy across GPUs might be implemented using copy engines, TMA, or LSU. TileScale provides default implementations for each primitive, along with a compilation process to tune the best implementation. Users can also specify particular implementations through arguments in the tile primitives.
With this hierarchical interface, user can easily customize the computation at certain scale level. For example, we can leverage the DSMEM feature to implement a general cluster-scale GEMM primitive.

## System Overview and Design
<div align="center">    <img src="./images/overview.png" alt="TileScale system overview" width=50% />
</div>
The frontend of TileScale provides all the tile primitives, Python bindings, and related programming syntax. The middle layer consists of three modules: compiler, tile kernels, and cost model. The compiler module lowers the frontend program into an intermediate representation (IR), applies optimization passes, and lowers tile primitives to lower-level primitives. For example, a block-scale GEMM primitive can be either directly mapped to a pre-implemented kernel or lowered to low-level code.
The tile-kernel module is a library that contains all the implementations of tile primitives.
The cost model builds a performance database and provides lightweight performance feedback for specific optimization plans. This feedback is used by the compiler module to optimize the program.
Finally, the backend module defines a configurable hardware architecture following the HDA abstraction. Unlike existing compilers that target few specific hardware, TileScale can compile a program to any user-defined architecture.

### Memory management

A tensor tile can be allocated at a specified memory layer for a certain scale compute. For example, the above example allocates a block-scale tile that physically resides in L0 (i.e., register) memory. This means the tile will be partitioned into each individual thread's registers, similar to the concept of a fragment in CUDA.
To use the tile at different levels of scale, we can use the T.view primitive to reference the specific partition of the tile within the corresponding scale. In the above example, it could be viewed as a warpgroup-, warp-, or thread-scale tile.
The layout and partition dimensions are either automatically inferred through a layout inference pass or specified by the user.
<div align="center">    <img src="./images/view.png" alt="T.alloc and T.view" width=50% />
</div>

### Parallel task scheduling
TileScale introduces a *T.Scale* primitive to control which hardware scale the current computations are conducted on.
It follows the SPMD (Single Program Multiple Data) programming model that scale the specified computation to all parallel units at this level.
For example, the following *T.gemm* represents a warp GEMM, which executes on all warps in parallel.
```python
# warp-level compute example
with T.Scale("warp"):
    T.gemm(A, B, C)
```
#### UTCMMA
For NVIDIA latest CTA cluster level GEMM, e.g., UTCMMA, it is fundamentally a distributed GEMM running on two SMs. This can be easily expressed in TileScale. For example,
```python
# cluster-level GEMM example
with T.Kernel(
    cta_cluster=(2),
    block=(block_M, block_N),
    threads=256
):
    with T.Scale("cta_cluster"):
        T.gemm(A, B, C)
```
#### Task(warp) specialization
Additionally, the T.Scale primitive can also return the rank and the total number of ranks of the current scale level. This allows you to easily leverage the rank index for task specialization, such as warp specialization or any other scale-level specialization.

```python
# warp specialize example
with T.Scale("warpgroup") as wg_id, wg_num:
    if wg_id == 0:
        # do something
    else:
        # do other thing
```
#### MPI-style programming
Combined with the communication primitives, you can also implement MPI-like programs if a communication channel exists across those ranks. For those compute units without hardware links, TileScale can also implement software channels by passing data through lower-level memory.
```python
# communication example: send data to neighbor GPU
with T.Scale("device") as dev_id, dev_num:
    T.copy(remote_B, local_A, dst=(dev_id + 1) % dev_num)
    T.barrier()
```

## Example:
```python
# Example of GEMM
# 4-GPU Tensor Parallelism, using L2 to communicate
# 2-warpgroups split along K dimension and reduce, using L1 to communicate
def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
):
    with T.Kernel( # launch config
        device=(4),
        block=(T.ceildiv(N, block_N), T.ceildiv(M, block_M)),
        threads=256
    ):
        with T.Scale("device"):
            A_global = T.view(A, layout=T.FullCol)
            B_global = T.view(B, layout=T.FullRow)
            C_global = T.view(C, layout=T.Replica)

        with T.Scale("block"):
            A_local = T.alloc((block_M, block_K), dtype, level="l0")
            B_local = T.alloc((block_K, block_N), dtype, level="l0")
            C_local = T.alloc((block_M, block_N), accum_dtype, level="l0")
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(A_global.shape[1], block_K), num_stages=3):
                with T.Scale("warpgroup") as wg_id, wg_num:
                    A_local_wg = T.view(A_local, layout=T.FullCol)
                    B_local_wg = T.view(B_local, layout=T.FullRow)
                    C_local_wg = T.view(C_local, layout=T.Replica)
                    T.copy(A_local_wg, A_global[by * block_M, k * block_K])
                    T.copy(B_local_wg, B_global[k * block_K, bx * block_N])
                    T.gemm(A_local_wg, B_local_wg, C_local_wg)

                    # Allreduce C_local_wg through software-defined channel on L1
                    T.allreduce(C_local_wg)
            T.copy(C_global[by * block_M, bx * block_N], C_local)

        with T.Scale("device") as dev_id, dev_num:
            # Allreduce C on L2
            T.allreduce(C_global)

```
```python
# Example of FlashMLA
# 4-GPU Context Parallelism, using L2 to communicate
# 2-warpgroups split acc_S and all-gather, using L1 to communicate
def flash_mla(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
):
    with T.Kernel(
        device=(4),
        block=(batch, heads // min(block_H, kv_group_num),
        threads=256)
    ):
        with T.Scale("device"):
            Q_global = T.view(Q, layout=T.Replica)
            Q_pe_global = T.view(Q_pe, layout=T.Replica)
            KV_global = T.view(KV, layout=lambda i, j, k, l: j // (seqlen_kv // 4))
            K_pe_global = T.view(K_pe, layout=lambda i, j, k, l: j // (seqlen_kv // 4))
            output_global = T.view(Output, layout=T.Replica)
            logsum_global = T.alloc([batch, heads], accum_dtype, level="l2")

        with T.Scale("block"):
            Q_shared = T.alloc([block_H, dim], dtype, level="l1")
            Q_pe_shared = T.alloc([block_H, pe_dim], dtype, level="l1")
            KV_shared = T.alloc([block_N, dim], dtype, level="l1")
            K_pe_shared = T.alloc([block_N, pe_dim], dtype, level="l1")

            acc_s = T.alloc([block_H, block_N], accum_dtype, level="l0")
            acc_s_cast = T.alloc([block_H, block_N], dtype, level="l0")
            acc_o = T.alloc([block_H, dim], accum_dtype, level="l0")
            scores_max = T.alloc([block_H], accum_dtype, level="l0")
            scores_max_prev = T.alloc([block_H], accum_dtype, level="l0")
            scores_scale = T.alloc([block_H], accum_dtype, level="l0")
            scores_sum = T.alloc([block_H], accum_dtype, level="l0")
            logsum = T.alloc([block_H], accum_dtype, level="l0")

            cur_kv_head = by // (kv_group_num // block_H)

            T.copy(Q_shared, Q_global[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])
            T.copy(Q_pe_shared, Q_pe_global[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(KV_global.shape[1], block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(KV_shared, KV_global[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :])
                T.copy(K_pe_shared, K_pe_global[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :])
                T.clear(acc_s)

                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                T.copy(scores_max_prev, scores_max)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)

                with T.Scale("warpgroup") as wg_id, wg_num:
                    acc_s_local = T.view(acc_s, layout=T.FullCol)
                    acc_s_cast_local = T.view(acc_s_cast, layout=T.Replica)
                    T.copy(acc_s_cast_local[:, 0:block_N // 2], acc_s_local)
                    # transfer on l0 using l1
                    T.copy(acc_s_cast_local[:, block_N // 2:block_N], acc_s_local, dst=(wg_id + 1) % wg_num)
                    # Or, you can use high level cooperative primitive
                    # T.allgather(acc_s_local), and Cast ...

                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(output_global[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], acc_o)
            T.copy(logsum_global[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H], logsum)

        with T.Scale("device"):
            # AllReduce on L2
            T.allreduce(output_global, logsum_global, fn=...)
            # Or, you can write copy output_global to peers and reduce by hand
```

## Installation
For detailed instructions, please refer to the [Installation Guide](docs/get_started/Installation.md).

## Call for Contribution!!
TileScale is in its early experimental stage and driven by the open-source community. We're looking for passionate contributors to help shape the future of distributed programming together! If you're excited about designing and developing the next-generation programming paradigm, please contact us: tile-ai@outlook.com. For more information, please check out our [roadmap](https://github.com/tile-ai/tilescale/issues/4).
