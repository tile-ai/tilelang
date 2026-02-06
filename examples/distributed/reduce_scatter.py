import dataclasses
from typing import List, Optional

import torch
import importlib.metadata
from packaging import version

cuda_python_version = importlib.metadata.version("cuda-python")
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
else:
    from cuda import cuda
import tilelang
from tilelang.distributed.utils import CUDA_CHECK, has_fullmesh_nvlink
from tilelang.utils.target import target_is_hopper, determine_target
import torch.distributed as dist
import tilelang.language as T

tilelang.disable_cache()


@dataclasses.dataclass
class ReduceScatter2DContext:
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    overlap_with_gemm: bool

    # comm buffer
    scatter_bufs: List[torch.Tensor]
    rs_per_node_bufs: List[torch.Tensor]
    p2p_bufs: List[torch.Tensor]

    # barrier bufs
    signal_bufs: List[torch.Tensor]  # need reset: signal_buf =  scatter_signal | rs_per_node_signal

    counter_bufs: List[torch.Tensor]

    # intra-node barrier
    barrier: List[torch.Tensor]

    # stream
    reduction_stream: torch.cuda.Stream

    # sms
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int

    # preprocess to reduce cpu overhead
    # comm barriers
    scatter_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)
    rs_per_node_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    scatter_signal_buf_list_for_each_node: List[torch.Tensor] = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.max_M % self.world_size == 0
        assert len(self.signal_bufs) == self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.scatter_signal_buf_list_for_each_node = []
        for buf in self.signal_bufs:
            assert buf.shape[0] >= 2 * self.world_size

        self.scatter_signal_bufs = [buf[: self.world_size] for buf in self.signal_bufs]
        self.rs_per_node_signal_bufs = [buf[self.world_size : self.world_size * 2] for buf in self.signal_bufs]

        for node_id in range(self.nnodes):
            self.scatter_signal_buf_list_for_each_node.append(
                self.scatter_signal_bufs[self.local_rank][node_id * self.local_world_size : (node_id + 1) * self.local_world_size]
            )

    def reset_barriers(self):
        self.signal_bufs[self.local_rank].fill_(0)
        self.counter_bufs[self.local_rank].fill_(0)

    def get_scatter_bufs_and_signal_for_each_node(self, input, node_id):
        M = input.shape[0]
        M_per_rank = M // self.world_size
        M_per_node = M_per_rank * self.local_world_size
        M_start = node_id * M_per_node
        M_end = M_start + M_per_node
        scatter_bufs_intra_node = [self.scatter_bufs[i][M_start:M_end] for i in range(self.local_world_size)]
        return scatter_bufs_intra_node, self.scatter_signal_buf_list_for_each_node[node_id]

    @property
    def rs_per_node_buf(self) -> torch.Tensor:
        return self.rs_per_node_bufs[self.local_rank]

    @property
    def rs_per_node_signal_buf(self) -> torch.Tensor:
        return self.rs_per_node_signal_bufs[self.local_rank]

    @property
    def p2p_buf(self) -> torch.Tensor:
        return self.p2p_bufs[self.local_rank]

    @property
    def num_rs_sms(self) -> int:
        if self.nnodes > 1:
            return self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms
        else:
            # for intra node rs, no need sm.
            return 0

    @property
    def scatter_signal_buf(self) -> torch.Tensor:
        return self.scatter_signal_bufs[self.local_rank]


def create_reduce_scater_2d_ctx(
    max_M, N, rank, world_size, local_world_size, dtype, allocator, overlap_with_gemm=True, num_reduction_sms=15
) -> ReduceScatter2DContext:
    """
    for num_reduction_sms: tunable param, 16 are enough for H800
        For H800, we overlap local reduce and inter-node p2p with intra-node scatter.
        The reduction kernel bandwidth is not a bottleneck if it exceeds 450GB, so only a few SMs are needed.
        For machines with higher intra_node bandwidth(e.g. H100), we may need to increase the number of SMs or redesign overlapping.
    """
    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = tilelang.tensor((max_M, N), dtype, allocator=allocator, return_peers=True)
    rs_per_node_bufs = tilelang.tensor((max_M // local_world_size, N), dtype, allocator=allocator, return_peers=True)
    p2p_bufs = tilelang.tensor((max_M // local_world_size, N), dtype, allocator=allocator, return_peers=True)

    # signal_buf: scatter_signal | rs_per_node_signal
    num_signal_bufs = 2
    signal_bufs = tilelang.tensor((world_size * num_signal_bufs), dtype=torch.uint32, allocator=allocator, return_peers=True)
    symm_barriers = tilelang.tensor((local_world_size,), torch.int32, allocator=allocator, return_peers=True)
    symm_barriers[rank] = 0

    counter_signal_buf = tilelang.tensor((local_world_size), dtype=torch.uint32, allocator=allocator, return_peers=True)

    dist.barrier()

    reduction_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

    num_sync_sms = 0
    num_p2p_sms = 1
    ctx = ReduceScatter2DContext(
        max_M=max_M,
        N=N,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        overlap_with_gemm=overlap_with_gemm,
        scatter_bufs=scatter_bufs,
        rs_per_node_bufs=rs_per_node_bufs,
        p2p_bufs=p2p_bufs,
        signal_bufs=signal_bufs,
        counter_bufs=counter_signal_buf,
        barrier=symm_barriers,
        reduction_stream=reduction_stream,
        num_sync_sms=num_sync_sms,
        num_p2p_sms=num_p2p_sms,
        num_reduction_sms=num_reduction_sms,
    )
    return ctx


@tilelang.jit
def kernel_ring_reduce_tma(
    M_per_rank, N, block_M, block_N, begin_idx, num_splits, threads, persistent=False, dtype="float16", accum_dtype="float"
):
    @T.prim_func
    def _kernel_ring_reduce_tma(
        C: T.Tensor((M_per_rank * num_splits, N), dtype),
        output: T.Tensor((M_per_rank, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M_per_rank, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_N), dtype)
            init_shared = T.alloc_shared((block_M, block_N), dtype)
            data_local = T.alloc_fragment((block_M, block_N), dtype)
            accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            cur_rank = (begin_idx + 1) % num_splits
            T.copy(C[cur_rank * M_per_rank + bx * block_M, by * block_N], init_shared)
            T.copy(init_shared, accum)
            for i in T.Pipelined(num_splits - 1, num_stages=0):
                cur_rank = (i + 1 + begin_idx + 1) % num_splits
                T.copy(C[cur_rank * M_per_rank + bx * block_M, by * block_N], data_shared)
                T.copy(data_shared, data_local)
                for i, j in T.Parallel(block_M, block_N):
                    accum[i, j] += data_local[i, j]
            T.copy(accum, output[bx * block_M, by * block_N])

    return _kernel_ring_reduce_tma


def _wait_eq_cuda(signal_tensor: torch.Tensor, signal: int, stream: Optional[torch.cuda.Stream] = None, require_i64=False):
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype in (torch.int32, torch.uint32):
        (err,) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    elif signal_tensor.dtype in (torch.int64, torch.uint64):
        (err,) = cuda.cuStreamWaitValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def intra_node_scatter(
    input_intra_node,
    scatter_bufs_intra_node: List[torch.Tensor],
    scatter_signal_buf_intra_node: torch.Tensor,
    local_rank,
    overlap_with_gemm=True,
):
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size

    stream = torch.cuda.current_stream()

    for i in range(local_world_size):
        remote_local_rank = (local_rank + i + 1) % local_world_size

        # print(f"scatter_signal_buf_intra_node[remote_local_rank]: {scatter_signal_buf_intra_node[remote_local_rank]}")
        if overlap_with_gemm:
            _wait_eq_cuda(scatter_signal_buf_intra_node[remote_local_rank], 1, stream)
        src = input_intra_node[remote_local_rank * M_per_rank : (remote_local_rank + 1) * M_per_rank, :]
        dst = scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank : (local_rank + 1) * M_per_rank, :]
        with torch.cuda.stream(stream):
            dst.copy_(src)


def ring_reduce_tma(
    input: torch.Tensor,  # [M_per_node, N]
    output: torch.Tensor,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    num_sms=-1,
):
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0, f"{output.shape}, {total_M}, {num_splits}"

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    if num_sms == -1:
        ring_reduce_tma_func = kernel_ring_reduce_tma(
            M_per_split, N, block_M=64, block_N=64, begin_idx=begin_idx, num_splits=num_splits, threads=128
        )
        # if begin_idx == 0:
        #     print(ring_reduce_tma_func.get_kernel_source())
        ring_reduce_tma_func(input, output, stream=torch.cuda.current_stream().cuda_stream)
    else:
        raise NotImplementedError("Currently only support num_sms = -1 for TMA ring reduce.")
        # grid = lambda META: (min(
        #     triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
        # kernel_ring_reduce_tma[grid](
        #     input,
        #     output,
        #     M_per_split,
        #     N,
        #     begin_idx,
        #     num_splits,
        #     BLOCK_SIZE_M=256,
        #     BLOCK_SIZE_N=128,
        #     num_warps=8,
        # )

    return output


target = determine_target(return_object=True)


def ring_reduce(
    input,  # [M_per_node, N]
    output,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    num_sms=-1,
):
    if target_is_hopper(target):
        return ring_reduce_tma(input, output, begin_idx, num_splits, num_sms)
    else:
        raise NotImplementedError("Only Hopper ring reduce is implemented now.")


def reduce_scatter_for_each_node(input: torch.Tensor, ctx: ReduceScatter2DContext, output: Optional[torch.Tensor] = None):
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    M, N = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.rs_per_node_buf
    p2p_buf = ctx.p2p_buf

    stream = torch.cuda.current_stream()
    for n in range(0, nnodes):
        cur_node_id = (node_id + n + 1) % nnodes
        input_intra_node = input[cur_node_id * M_per_node : (cur_node_id + 1) * M_per_node]
        scatter_bufs_intra_node, scatter_signal_buf_intra_node = ctx.get_scatter_bufs_and_signal_for_each_node(input, cur_node_id)
        intra_node_scatter(
            input_intra_node, scatter_bufs_intra_node, scatter_signal_buf_intra_node, local_rank, overlap_with_gemm=ctx.overlap_with_gemm
        )

        # ring reduce intra node
        rs_buf_cur_node = rs_per_node_buf[M_per_rank * cur_node_id : (cur_node_id + 1) * M_per_rank]
        # nvshmem_barrier_all_on_stream(stream)
        reduction_stream.wait_stream(stream)
        with torch.cuda.stream(reduction_stream):
            reduce_out_buf = output if nnodes == 1 else rs_buf_cur_node
            ring_reduce(
                scatter_bufs_intra_node[local_rank],
                reduce_out_buf,
                local_rank,
                local_world_size,
                num_sms=-1 if n == nnodes - 1 else num_reduction_sms,
            )

            # inter node p2p
            if nnodes > 1:
                raise NotImplementedError("Inter-node p2p is not implemented yet.")
                # if n == nnodes - 1:
                #     p2p_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)].copy_(
                #         rs_per_node_buf[M_per_rank * node_id:M_per_rank * (node_id + 1)])
                # else:
                #     grid = lambda META: (ctx.num_p2p_sms,)
                #     kernel_inter_node_p2p_for_same_local_rank[grid](
                #         n,
                #         local_world_size,
                #         M_per_rank,
                #         N,
                #         rs_per_node_buf,
                #         p2p_buf,
                #         num_warps=16,
                #     )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return output
    return p2p_buf[: M_per_rank * nnodes]


def reduce_scatter_multi_node(input: torch.Tensor, ctx: ReduceScatter2DContext, output: Optional[torch.Tensor] = None):
    """
    A hierarchical reduce-scatter implementation that overlaps the intra-node scatter
    with the local reduce and the inter-node p2p(after reduce). It also provides a rank-wise
    signal and supports overlap with gemm.
    """
    M, N = input.shape
    M_per_rank = M // ctx.world_size

    current_stream = torch.cuda.current_stream()
    ctx.reduction_stream.wait_stream(current_stream)

    # directly reduce_scatter to output if nnodes == 1
    out_each_node = output if ctx.nnodes == 1 else None
    if not has_fullmesh_nvlink():
        raise Exception("Only support fullmesh nvlink topology for now.")
    else:
        print("Using fullmesh nvlink reduce_scatter.")
        rs_result_per_node = reduce_scatter_for_each_node(input, ctx, out_each_node)

    if ctx.nnodes == 1:
        return rs_result_per_node

    # nvshmem_barrier_all_on_stream(current_stream)
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)
    ring_reduce(rs_result_per_node, output, ctx.node_id, ctx.nnodes)
    return output


def reduce_scatter_2d_op(input: torch.Tensor, ctx: ReduceScatter2DContext, output: Optional[torch.Tensor] = None):
    M, N = input.shape
    assert input.dtype == ctx.dtype
    assert ctx.max_M >= M and ctx.N == N
    assert M % ctx.world_size == 0

    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    output = reduce_scatter_multi_node(input, ctx, output)
    ctx.reset_barriers()
    return output
