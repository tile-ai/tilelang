import os
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

import tilelang
import tilelang.language as T
from utils import Config
from intranode import get_dispatch_layout


class TSBuffer:
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

    def __init__(self, group: dist.ProcessGroup, num_nvl_bytes: int):
        """
        Initialize the communication buffer.

        Args:
            group: the communication group
            num_nvl_bytes: the buffer size for intranode NVLink communication.
        """
        self.group = group
        self.rank = group.rank()
        self.num_ranks = group.size()
        self.num_nvl_bytes = num_nvl_bytes
        assert self.num_ranks <= 8, "currently only support intranode"

        self._allocator= tilelang.get_allocator(
            size=2**30,
            device="cuda",
            is_distributed=True,
            local_rank=self.rank,
            num_local_ranks=self.num_ranks,
            group=group)

    @staticmethod
    def set_num_sms(num_sms: int):
        """Set the number of SMs used in high-throughput kernels
        
        Args:
            num_sms: the number of SMs used in high-throughput kernels
        """
        assert num_sms % 2 == 0, "num_sms must be even"
        TSBuffer.num_sms = num_sms

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

    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int):
        return get_dispatch_layout(topk_idx, num_experts, self.num_ranks)

    def dispatch(
        self,
        x: torch.Tensor,
        num_tokens_per_rank: torch.Tensor,
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_alignment: int = 1,
    ):
        per_rank_buffer = tilelang.tensor((self.num_ranks, self.num_ranks), dtype=torch.int32, device='cuda', allocator=self._allocator).zero_()
        per_expert_buffer = tilelang.tensor((self.num_ranks, num_tokens_per_expert.shape[0]), dtype=torch.int32, device='cuda', allocator=self._allocator).zero_()
        barrier_signal = tilelang.tensor((self.num_ranks), dtype=torch.int32, device='cuda', allocator=self._allocator).zero_()