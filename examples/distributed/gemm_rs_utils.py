import dataclasses
from typing import List

import torch

import pynvshmem

SIGNAL_DTYPE = torch.uint64


class BarrierAllContext:
    """
    You may use this to barrier all ranks in global, or just in intra-node team.

    NOTE: nvshmem_barrier_all is slower for intra-node only.
    """

    def __init__(self, is_intra_node):
        self.is_intra_node = is_intra_node
        # TODO: implement these for intra-node
        # if self.is_intra_node:
        #     self.rank = pynvshmem.nvshmem_my_pe()
        #     self.local_rank = pynvshmem.nvshmem_team_my_pe(pynvshmem.Team.NODE)
        #     self.num_local_ranks = pynvshmem.nvshmem_team_n_pes(pynvshmem.Team.NODE)
        #     self.symm_barrier = pynvshmem.nvshmem_create_tensor((1, ), torch.int32)
        #     self.symm_barrier.fill_(0)
        #     pynvshmem.nvshmem_barrier_all()


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

    # intra-node barrier
    barrier: BarrierAllContext

    # stream
    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

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

    def reset_barriers(self) -> int:
        # self.scatter_signal_bufs[self.local_rank].fill_(0)
        # self.rs_per_node_signal_bufs[self.local_rank].fill_(0)
        self.signal_bufs[self.local_rank].fill_(0)

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
    max_M, N, rank, world_size, local_world_size, dtype, overlap_with_gemm=True, num_reduction_sms=15
) -> ReduceScatter2DContext:
    """
    for num_reduction_sms: tunable param, 16 are enough for H800
        For H800, we overlap local reduce and inter-node p2p with intra-node scatter.
        The reduction kernel bandwidth is not a bottleneck if it exceeds 450GB, so only a few SMs are needed.
        For machines with higher intra_node bandwidth(e.g. H100), we may need to increase the number of SMs or redesign overlapping.
    """
    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], dtype)

    rs_per_node_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M // local_world_size, N], dtype)

    p2p_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M // local_world_size, N], dtype)

    # signal_buf: scatter_signal | rs_per_node_signal
    num_signal_bufs = 2
    signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node(
        [
            world_size * num_signal_bufs,
        ],
        SIGNAL_DTYPE,
    )

    # TODO: implement barrier_all_on_stream
    # barrier_all_on_stream(None, torch.cuda.current_stream())

    p2p_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
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
        barrier=BarrierAllContext(True),
        reduction_stream=reduction_stream,
        p2p_stream=p2p_stream,
        num_sync_sms=num_sync_sms,
        num_p2p_sms=num_p2p_sms,
        num_reduction_sms=num_reduction_sms,
    )
    return ctx


################### context ###################
@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rs_ctx: ReduceScatter2DContext
    output_dtype: torch.dtype

    # gemm bufs (symm address)
    gemm_out_bufs: List[torch.Tensor]

    # stream
    rs_stream: torch.cuda.Stream

    # gemm kernel config
    num_gemm_sms: int
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    stages: int = 3

    def update(self, rs_stream, output_dtype=None, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8, stages=3):
        self.rs_stream = rs_stream
        self.output_dtype = output_dtype
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.stages = stages

    def get_gemm_out_buf(self, input):
        M, _ = input.shape
        local_rank = self.rs_ctx.local_rank
        return self.gemm_out_bufs[local_rank][:M]


def create_gemm_rs_context(
    max_M, N, rank, world_size, local_world_size, output_dtype, rs_stream, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8, stages=3
) -> GEMMReduceScatterTensorParallelContext:
    rs_ctx = create_reduce_scater_2d_ctx(max_M, N, rank, world_size, local_world_size, output_dtype, overlap_with_gemm=True)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms
    gemm_out_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], output_dtype)
    ctx = GEMMReduceScatterTensorParallelContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out_bufs=gemm_out_bufs,
        rs_stream=rs_stream,
        num_gemm_sms=num_gemm_sms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        stages=stages,
    )
    return ctx
