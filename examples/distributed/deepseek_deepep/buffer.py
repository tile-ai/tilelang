"""The interface for DeepEP."""

import torch
import torch.distributed as dist
from typing import Tuple, Optional

import tilelang
from deepep_utils import Config
from tilelang.distributed.utils import create_mapped_tensor
from intranode.get_dispatch_layout import get_dispatch_layout
from intranode.dispatch import intranode_dispatch
from intranode.combine import intranode_combine


class EPBuffer:
    """
    TileScale communication buffers for DeepEP

    Attributes:
        num_sms: the number of SMs used in high-throughput kernels
        group: the communication process group
        rank: the local rank
        num_ranks: the total number of ranks
        num_nvl_bytes: the buffer size for intranode NVLink communication.
    """

    num_sms: int = 20
    symm_heap_size: int = 2**30  # size of the symm heap for allocators

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int,
        num_topk: int,
        num_experts: int,
        hidden: int,
        dispatch_cfg: Optional[Config] = None,
        combine_cfg: Optional[Config] = None,
    ):
        """
        Initialize the communication buffer.

        Args:
            group: the communication group
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_topk: the number of topk experts to select.
            num_experts: the number of experts.
            hidden: the hidden dimension.
            dispatch_cfg: the performance tuning config for dispatch.
            combine_cfg: the performance tuning config for combine.
        """
        self.group = group
        self.rank = group.rank()
        self.num_ranks = group.size()

        self.num_nvl_bytes = num_nvl_bytes
        assert self.num_ranks <= 8, "currently only support intranode"  # todo: rm this
        self.num_topk = num_topk
        self.num_experts = num_experts
        assert num_experts % self.num_ranks == 0, "num_experts must be divisible by num_ranks"
        self.num_local_experts = num_experts // self.num_ranks
        self.hidden = hidden

        self.dispatch_cfg = dispatch_cfg if dispatch_cfg is not None else self.default_dispatch_config
        self.combine_cfg = combine_cfg if combine_cfg is not None else self.default_combine_config

        self.comm_stream = torch.cuda.Stream()

        self._allocator = tilelang.get_allocator(
            size=EPBuffer.symm_heap_size,
            device="cuda",
            is_distributed=True,
            local_rank=self.rank,
            num_local_ranks=self.num_ranks,
            group=group,
        )

        self._pre_alloc_symm_buffers()
        self._prepare_counters()

        torch.cuda.synchronize()
        self.group.barrier()

    def _pre_alloc_symm_buffers(self):
        """Pre-allocate the symmetric buffers via the allocator for later communication."""
        if self.num_ranks <= 8:
            self._pre_alloc_symm_buffers_intranode()  # todo: rm this
        else:
            self._pre_alloc_symm_buffers_internode()

    def _pre_alloc_symm_buffers_intranode(self):
        # barrier signal is always zeroed after each usage, so we can pre-init here
        barrier_signal = tilelang.tensor((self.num_ranks), dtype=torch.int32, device="cuda", allocator=self._allocator).zero_()

        per_rank_buffer = tilelang.tensor((self.num_ranks, self.num_ranks), dtype=torch.int32, device="cuda", allocator=self._allocator)
        per_expert_buffer = tilelang.tensor(
            (self.num_ranks, self.num_local_experts), dtype=torch.int32, device="cuda", allocator=self._allocator
        )

        channel_start_offset = tilelang.tensor(
            [self.num_channels, self.num_ranks], dtype=torch.int32, device="cuda", allocator=self._allocator
        )
        channel_end_offset = tilelang.tensor(
            [self.num_channels, self.num_ranks], dtype=torch.int32, device="cuda", allocator=self._allocator
        )
        channel_head_idx = tilelang.tensor([self.num_channels, self.num_ranks], dtype=torch.int32, device="cuda", allocator=self._allocator)
        channel_tail_idx = tilelang.tensor([self.num_channels, self.num_ranks], dtype=torch.int32, device="cuda", allocator=self._allocator)
        # NOTE: for each #ranks, dispatch and combine cfg have the same num_max_nvl_chunked_recv_tokens, so we can use the same buffer here
        channel_x_buffers = tilelang.tensor(
            [self.num_channels, self.num_ranks, self.dispatch_cfg.num_max_nvl_chunked_recv_tokens, self.hidden],
            dtype=torch.bfloat16,
            device="cuda",
            allocator=self._allocator,
        )
        channel_src_idx_buffers = tilelang.tensor(
            [self.num_channels, self.num_ranks, self.dispatch_cfg.num_max_nvl_chunked_recv_tokens],
            dtype=torch.int32,
            device="cuda",
            allocator=self._allocator,
        )
        channel_topk_idx_buffers = tilelang.tensor(
            [self.num_channels, self.num_ranks, self.dispatch_cfg.num_max_nvl_chunked_recv_tokens, self.num_topk],
            dtype=torch.int64,
            device="cuda",
            allocator=self._allocator,
        )
        channel_topk_weights_buffers = tilelang.tensor(
            [self.num_channels, self.num_ranks, self.dispatch_cfg.num_max_nvl_chunked_recv_tokens, self.num_topk],
            dtype=torch.float32,
            device="cuda",
            allocator=self._allocator,
        )

        self._symm_buffers = (
            barrier_signal,
            per_rank_buffer,
            per_expert_buffer,
            channel_start_offset,
            channel_end_offset,
            channel_head_idx,
            channel_tail_idx,
            channel_x_buffers,
            channel_src_idx_buffers,
            channel_topk_idx_buffers,
            channel_topk_weights_buffers,
        )

    def _pre_alloc_symm_buffers_internode(self):
        raise NotImplementedError("internode is not supported yet")

    def _prepare_counters(self):
        self._moe_recv_counter, self._moe_recv_counter_mapped = create_mapped_tensor([1], torch.int32)
        self._moe_recv_expert_counter, self._moe_recv_expert_counter_mapped = create_mapped_tensor([self.num_local_experts], torch.int32)

        if self.num_ranks > 8:  # internode
            self._moe_recv_rdma_counter, self._moe_recv_rdma_counter_mapped = create_mapped_tensor([1], torch.int32)

    @staticmethod
    def set_num_sms(num_sms: int):
        """Set the number of SMs used in high-throughput kernels

        Args:
            num_sms: the number of SMs used in high-throughput kernels
        """
        assert num_sms % 2 == 0, "num_sms must be even"
        EPBuffer.num_sms = num_sms

    @property
    def num_channels(self):
        """Get the number of communication channels

        Returns:
            the number of communication channels
        """
        return self.num_sms // 2
        # 1 sm for send, 1 sm for recv in each channel

    @property
    def default_dispatch_config(self):
        return Config.get_dispatch_config(self.num_ranks)

    @property
    def default_combine_config(self):
        return Config.get_combine_config(self.num_ranks)

    def get_dispatch_layout(self, topk_idx: torch.Tensor):
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
        """
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = get_dispatch_layout(topk_idx, self.num_experts, self.num_ranks)
        return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank

    def dispatch(
        self,
        x: torch.Tensor,
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
    ):
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
        """
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            recv_x = intranode_dispatch(
                self.rank,
                self._allocator,
                self._symm_buffers,
                self._moe_recv_counter,
                self._moe_recv_expert_counter,
                self._moe_recv_counter_mapped,
                self._moe_recv_expert_counter_mapped,
                x,
                self.dispatch_cfg,
                handle,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                self.comm_stream,
            )
            return recv_x  # cached-mode, only return recv_x
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle = intranode_dispatch(
                self.rank,
                self._allocator,
                self._symm_buffers,
                self._moe_recv_counter,
                self._moe_recv_expert_counter,
                self._moe_recv_counter_mapped,
                self._moe_recv_expert_counter_mapped,
                x,
                self.dispatch_cfg,
                handle,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                self.comm_stream,
            )
            return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle

    def combine(self, x: torch.Tensor, handle: Tuple, topk_weights: torch.Tensor):
        # todo: support bias
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
        """
        recv_x, recv_topk_weights = intranode_combine(
            self.rank, self._allocator, self._symm_buffers, x, self.combine_cfg, handle, topk_weights, self.comm_stream
        )
        return recv_x, recv_topk_weights
