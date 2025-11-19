from typing import Union, Tuple
import torch
import os
from dataclasses import dataclass

# Pre-defined constants in DeepEP
NUM_MAX_NVL_PEERS = 8  # Maximum number of NVLink peers per GPU
NUM_MAX_RDMA_PEERS = 20  # Maximum number of RDMA peers per GPU
NUM_MAX_LOCAL_EXPERTS = 1024  # Maximum number of local experts per GPU
NUM_WORKSPACE_BYTES = 32 * 1024 * 1024  # 32 MiB
NUM_BUFFER_ALIGNMENT_BYTES = 128

num_sms: int = 20


@dataclass
class Config:
    num_max_nvl_chunked_send_tokens : int
    num_max_nvl_chunked_recv_tokens : int
    num_max_rdma_chunked_send_tokens : int
    num_max_rdma_chunked_recv_tokens : int

    num_sms : int = 20  # the SMs used in high-throughput kernels

    def __post_init__(self):
        assert self.num_sms % 2 == 0, "num_sms must be even"
    
    @staticmethod
    def get_dispatch_config(num_ranks: int) -> 'Config':
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(num_sms, 24, 256, 6, 128),
            4: Config(num_sms, 6, 256, 6, 128),
            8: Config(num_sms, 6, 256, 6, 128),
            16: Config(num_sms, 36, 288, 20, 128),
            24: Config(num_sms, 32, 288, 8, 128),
            32: Config(num_sms, 32, 288, 8, 128),
            48: Config(num_sms, 32, 288, 8, 128),
            64: Config(num_sms, 32, 288, 8, 128),
            96: Config(num_sms, 20, 480, 12, 128),
            128: Config(num_sms, 20, 560, 12, 128),
            144: Config(num_sms, 32, 720, 12, 128),
            160: Config(num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]
    
    
    @staticmethod
    def get_combine_config(num_ranks: int) -> 'Config':
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(num_sms, 10, 256, 6, 128),
            4: Config(num_sms, 9, 256, 6, 128),
            8: Config(num_sms, 4, 256, 6, 128),
            16: Config(num_sms, 4, 288, 12, 128),
            24: Config(num_sms, 1, 288, 8, 128),
            32: Config(num_sms, 1, 288, 8, 128),
            48: Config(num_sms, 1, 288, 8, 128),
            64: Config(num_sms, 1, 288, 8, 128),
            96: Config(num_sms, 1, 480, 8, 128),
            128: Config(num_sms, 1, 560, 8, 128),
            144: Config(num_sms, 2, 720, 8, 128),
            160: Config(num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]


# Only necessary in inter-node cases
def set_rdma_env_args(num_qps_per_rank: int = 24, allow_nvlink_for_low_latency_mode: bool = True, allow_mnnvl: bool = False):
    os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'

    # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
    nvshmem_qp_depth = int(os.environ.get('NVSHMEM_QP_DEPTH', '1024'))
    os.environ['NVSHMEM_QP_DEPTH'] = str(nvshmem_qp_depth)

    # Reduce gpu memory usage
    # 6 default teams + 1 extra team
    os.environ['NVSHMEM_MAX_TEAMS'] = '7'
    # Disable NVLink SHArP
    os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
    # NOTES: NVSHMEM initialization requires at least 256 MiB
    os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

    if not allow_mnnvl:
        # Disable multi-node NVLink detection
        os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'


def unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    bias_0, bias_1 = None, None
    if isinstance(bias, torch.Tensor):
        bias_0 = bias
    elif isinstance(bias, tuple):
        assert len(bias) == 2
        bias_0, bias_1 = bias
    return bias_0, bias_1


# Check: DeepEP/tests/test_intranode.py:test_main
def gen_inputs(num_tokens: int, hidden: int, num_topk: int, num_experts: int, num_ranks: int):
    """Generate random inputs for testing purpose.
    Args:
        num_tokens: the number of tokens.
        hidden: the hidden dimension.
        num_topk: the number of top-k experts to select for each token.
        num_experts: the number of experts.
        num_ranks: the number of total ranks.
        
    Returns:
        x: `[num_tokens, hidden]` with `torch.bfloat16`, the input to MoE layer.
        topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
            `-1` means no selections.
        topk_weights: `[num_tokens, num_topk]` with `torch.float32`, the weights corresponding to 
            each selected expert for each token.
        rank_idx: `[num_tokens, num_topk]` with `torch.int64`, the rank indices corresponding to 
            each selected expert, `-1` means no selections.
    """
    assert num_topk <= num_experts, "num_topk must be less than or equal to num_experts"
    assert num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    return x, topk_idx, topk_weights, rank_idx


def inplace_unique(x: torch.Tensor, num_slots: int):
    """
    Keep at most `num_slots` different values in each row of `x`, 
    and fill `x` with -1 in other positions.
    """
    assert x.dim() == 2 and num_slots <= x.size(-1)
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


# Check: csrc/deep_ep.cpp:Buffer::Buffer
def create_moe_recv_counters(num_ranks: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rdma_ranks = max(1, num_ranks // NUM_MAX_NVL_PEERS)  # noqa: F841
    num_nvl_ranks = min(num_ranks, NUM_MAX_NVL_PEERS)  # noqa: F841

    moe_recv_counter = torch.tensor(
        -1, dtype=torch.int64, device='cuda', pin_memory=True)  # MoE counter
    moe_recv_expert_counter = torch.tensor(
        [-1] * NUM_MAX_LOCAL_EXPERTS, dtype=torch.int32, device='cuda',
        pin_memory=True)  # MoE expert-level counter
    moe_recv_rdma_counter = torch.tensor(
        -1, dtype=torch.int, device='cuda', pin_memory=True)  # MoE RDMA-level counter

    return moe_recv_counter, moe_recv_expert_counter, moe_recv_rdma_counter