import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from example_fusedmoe_torch import *


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def moe_forward_tilelang_shared(
    d_hidden,
    d_expert,
    n_shared_experts,
    dtype,
    num_tokens,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
):
    scale = 1.44269504  # log2(e)

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert * n_shared_experts

    # Tensors: Note that input shape is reshape to (num_tokens, dhidden)
    input_shape = (num_tokens, dhidden)
    shared_W_gate_shape = (dexpert, dhidden)
    shared_W_up_shape = (dexpert, dhidden)
    shared_W_down_shape = (dhidden, dexpert)

    accum_type = T.float32

    @T.prim_func
    def kernel_shared(
        input: T.Tensor(input_shape, dtype),  # type: ignore
        shared_W_gate: T.Tensor(shared_W_gate_shape, dtype),  # type: ignore
        shared_W_up: T.Tensor(shared_W_up_shape, dtype),  # type: ignore
        shared_W_down: T.Tensor(shared_W_down_shape, dtype),  # type: ignore
        up_logits: T.Tensor((num_tokens, dexpert), dtype),  # type: ignore
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(T.ceildiv(num_tokens, block_token), T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            # Split the block to shared experts and routed experts
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            W_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            W_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            # Shared experts: no need to check expert_indices

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)

            T.use_swizzle(10)
            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            # Parallel for gate and up matmul
            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(input[bx * block_token, k * block_dhidden], input_shared)
                T.copy(shared_W_gate[by * block_dexpert, k * block_dhidden], W_gate_shared)
                T.copy(shared_W_up[by * block_dexpert, k * block_dhidden], W_up_shared)
                T.gemm(input_shared, W_gate_shared, gate_logits_local, transpose_B=True)
                T.gemm(input_shared, W_up_shared, up_logits_local, transpose_B=True)

            # Fuse with SiLU and element-wise product
            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            T.copy(up_logits_local, up_logits[bx * block_token, by * block_dexpert])

        # Step 2: Compute down logits
        with T.Kernel(T.ceildiv(num_tokens, block_token), T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            W_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_type)

            T.use_swizzle(10)
            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(up_logits[bx * block_token, k * block_dexpert], up_logits_shared)
                T.copy(shared_W_down[by * block_dhidden, k * block_dexpert], W_down_shared)
                T.gemm(up_logits_shared, W_down_shared, output_local, transpose_B=True)

            T.copy(output_local, output[bx * block_token, by * block_dhidden])

    return kernel_shared


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def moe_forward_tilelang_routed(
    d_hidden,
    d_expert,
    n_routed_experts,
    dtype,
    group_sum,
    group_count,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
    k_pack=1,
    coalesced_width=None,
):
    scale = 1.44269504  # log2(e)

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert
    n_routed_experts = n_routed_experts

    # Group info
    # group_sum = sum(group_sizes_list)
    # group_count = len(group_sizes_list)
    # M = sum([(group_size + block_token - 1) // block_token for group_size in group_sizes_list])
    M = math.ceil(group_sum / block_token) + group_count
    accum_dtype = T.float32

    # Tensors: Note that input shape is reshape to (bs * seq_len * n_experts_per_token, dhidden) for grouped gemm
    input_shape = (group_sum, dhidden)
    intermediate_shape = (group_sum, dexpert)
    routed_expert_gate_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_up_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_down_shape = (n_routed_experts, dhidden, dexpert)
    routed_expert_weights_shape = group_sum
    group_sizes_shape = n_routed_experts

    @T.prim_func
    def kernel(
        input: T.Tensor(input_shape, dtype),  # type: ignore
        routed_expert_gate: T.Tensor(routed_expert_gate_shape, dtype),  # type: ignore
        routed_expert_up: T.Tensor(routed_expert_up_shape, dtype),  # type: ignore
        routed_expert_down: T.Tensor(routed_expert_down_shape, dtype),  # type: ignore
        routed_expert_weights: T.Tensor(routed_expert_weights_shape, dtype),  # type: ignore
        group_sizes: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_padded_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_idx_for_bx: T.Tensor((M,), T.int32),  # type: ignore
        up_logits: T.Tensor(intermediate_shape, dtype),  # type: ignore
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            routed_expert_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            routed_expert_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)

            T.use_swizzle(10, enable=True)

            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(
                    input[m_start : m_start + block_token, k * block_dhidden : (k + 1) * block_dhidden],
                    input_shared,
                    coalesced_width=coalesced_width,
                )
                T.copy(
                    routed_expert_gate[
                        cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
                    ],
                    routed_expert_gate_shared,
                    coalesced_width=coalesced_width,
                )
                T.gemm(input_shared, routed_expert_gate_shared, gate_logits_local, k_pack=k_pack, transpose_B=True)
                T.copy(
                    routed_expert_up[
                        cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
                    ],
                    routed_expert_up_shared,
                    coalesced_width=coalesced_width,
                )
                T.gemm(input_shared, routed_expert_up_shared, up_logits_local, k_pack=k_pack, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]

        # Step 2: Compute down logits
        with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(10, enable=True)

            m_start_padded = bx * block_token

            cur_group_idx = group_idx_for_bx[bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(
                    up_logits[m_start : m_start + block_token, k * block_dexpert : (k + 1) * block_dexpert],
                    up_logits_shared,
                    coalesced_width=coalesced_width,
                )
                T.copy(
                    routed_expert_down[
                        cur_group_idx, by * block_dhidden : (by + 1) * block_dhidden, k * block_dexpert : (k + 1) * block_dexpert
                    ],
                    routed_expert_down_shared,
                    coalesced_width=coalesced_width,
                )
                T.gemm(up_logits_shared, routed_expert_down_shared, output_local, k_pack=k_pack, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]

    return kernel


class Expert(nn.Module):
    def __init__(self, config: Dict, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert
        self.device = torch.device("cuda")

        self.W_gate_weight = gate.t().contiguous().to(self.device)
        self.W_up_weight = up.t().contiguous().to(self.device)
        self.W_down_weight = down.t().contiguous().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(x @ self.W_gate_weight)
        out = (gate * (x @ self.W_up_weight)) @ self.W_down_weight
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict, weights: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g_weight = weights["router.weight"].t()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.W_g_weight
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(
        self, config: Dict, shared_kernel: tilelang.JITKernel, routed_kernel: tilelang.JITKernel, weights: Dict, padding_M: int = 128
    ):
        super().__init__()
        self.config = config
        self.shared_kernel = shared_kernel
        self.routed_kernel = routed_kernel
        self.padding_M = padding_M
        self.experts = nn.ModuleList(
            [
                Expert(
                    config,
                    gate=weights[f"experts.{i}.0.weight"],
                    up=weights[f"experts.{i}.1.weight"],
                    down=weights[f"experts.{i}.2.weight"],
                )
                for i in range(config["n_routed_experts"])
            ]
        )
        self.device = torch.device("cuda")
        self.gating_network = MoEGate(config, weights).to(self.device)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(
            config=config,
            gate=weights["shared_experts.0.weight"],
            up=weights["shared_experts.1.weight"],
            down=weights["shared_experts.2.weight"],
            d_expert=shared_expert_dim,
        ).to(self.device)
        self.expert_cache = torch.zeros(
            (config["batch_size"] * config["seq_len"], config["d_hidden"]), dtype=torch.float16, device=self.device
        )
        self.stacked_expert_w_gate = torch.stack([expert.W_gate_weight for expert in self.experts], dim=0)
        self.stacked_expert_w_up = torch.stack([expert.W_up_weight for expert in self.experts], dim=0)
        self.stacked_expert_w_down = torch.stack([expert.W_down_weight for expert in self.experts], dim=0)
        self.stacked_expert_tokens = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_hidden"]),
            dtype=torch.float16,
            device=self.device,
        )
        self.stacked_expert_weights = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"]), dtype=torch.float16, device=self.device
        )
        self.stacked_expert_tokens_idxs = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"]), dtype=torch.int64, device=self.device
        )

        self.up_logits_shared = torch.empty(
            (config["batch_size"] * config["seq_len"], self.config["d_expert"]), dtype=torch.float16, device=self.device
        )
        self.expert_output_shared = torch.empty(
            (config["batch_size"] * config["seq_len"], self.config["d_hidden"]), dtype=torch.float16, device=self.device
        )
        self.up_logits_routed = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_expert"]),
            dtype=torch.float16,
            device=self.device,
        )
        self.expert_output_routed = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_hidden"]),
            dtype=torch.float16,
            device=self.device,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        expert_indices, expert_scores = self.gating_network(x)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1)
        x_flat = x.view(-1, hidden_dim)

        # Prepare for grouped GEMM
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        # counts = flat_expert_indices.bincount()
        tokens_per_expert = counts.cumsum()
        # tokens_per_expert = torch.cumsum(counts, dim=0)
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok

        # Get stacked expert tokens and expert weights

        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x_flat[exp_token_idxs]

            self.stacked_expert_tokens[start_idx:end_idx] = expert_tokens
            self.stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
            self.stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]

        group_sizes = torch.tensor(counts, dtype=torch.int32, device=self.device)
        group_offset = torch.tensor(tokens_per_expert - counts, dtype=torch.int32, device=self.device)

        group_padded_offsets = [0 for _ in range(len(group_sizes))]
        for i in range(1, len(group_sizes)):
            group_padded_offsets[i] = group_padded_offsets[i - 1] + math.ceil((counts[i - 1] + 1) / self.padding_M) * self.padding_M

        block_token = 128
        M = (
            math.ceil(self.config["batch_size"] * self.config["seq_len"] * self.config["n_experts_per_token"] / block_token)
            + self.config["n_routed_experts"]
        )
        group_idx_for_bx = [0 for _ in range(M)]

        for bx in range(M):
            m_start_padded = bx * block_token
            for i in range(self.config["n_routed_experts"]):
                if m_start_padded >= group_padded_offsets[i]:
                    group_idx_for_bx[bx] = i

        group_padded_offsets = torch.tensor(group_padded_offsets, dtype=torch.int32, device=self.device)
        group_idx_for_bx = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=self.device)

        # Multi-stream execution
        shared_stream = torch.cuda.Stream()
        routed_stream = torch.cuda.default_stream()
        torch.cuda.synchronize()

        with torch.cuda.stream(routed_stream):
            # Tilelang version: Grouped GEMM
            self.routed_kernel(
                self.stacked_expert_tokens,
                self.stacked_expert_w_gate,
                self.stacked_expert_w_up,
                self.stacked_expert_w_down,
                self.stacked_expert_weights,
                group_sizes,
                group_offset,
                group_padded_offsets,
                group_idx_for_bx,
                self.up_logits_routed,
                self.expert_output_routed,
            )

            # Scatter reduce
            self.expert_cache = torch.scatter_reduce(
                self.expert_cache,
                0,
                self.stacked_expert_tokens_idxs.view(-1, 1).repeat(1, x_flat.shape[-1]),
                self.expert_output_routed,
                reduce="sum",
            )
            routed_output = self.expert_cache.view(*orig_shape)

        with torch.cuda.stream(shared_stream):
            self.shared_kernel(
                x_flat,
                self.shared_expert.W_gate_weight,
                self.shared_expert.W_up_weight,
                self.shared_expert.W_down_weight,
                self.up_logits_shared,
                self.expert_output_shared,
            )
            shared_output = self.expert_output_shared.view(*orig_shape)

        torch.cuda.synchronize()

        return shared_output + routed_output


def custom_kernel(data: Tuple[torch.Tensor, Dict, Dict]) -> torch.Tensor:
    """
    DeepSeek-style Mixture of Experts using Tilelang.

    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters

    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
    """
    input_tensor, weights, config = data

    dtype_str = T.float16

    shared_kernel = moe_forward_tilelang_shared(
        config["d_hidden"],
        config["d_expert"],
        config["n_shared_experts"],
        dtype=dtype_str,
        num_tokens=config["batch_size"] * config["seq_len"],
    )
    routed_kernel = moe_forward_tilelang_routed(
        config["d_hidden"],
        config["d_expert"],
        config["n_routed_experts"],
        dtype=dtype_str,
        group_sum=config["batch_size"] * config["seq_len"] * config["n_experts_per_token"],
        group_count=config["n_routed_experts"],
        block_token=128,
        block_dhidden=128,
        block_dexpert=128,
        threads=256,
        num_stages=1,
        k_pack=1,
        coalesced_width=2,
    )

    moe = MoE(config, shared_kernel, routed_kernel, weights, padding_M=128)

    output = moe(input_tensor)

    return output


def main(d_hidden=7168, d_expert=2048, n_routed_experts=8, n_shared_experts=1, n_experts_per_token=4, batch_size=1, seq_len=8192):
    config = {
        "dhidden": d_hidden,
        "dexpert": d_expert,
        "nroutedexperts": n_routed_experts,
        "nsharedexperts": n_shared_experts,
        "nexpertspertoken": n_experts_per_token,
        "bs": batch_size,
        "seqlen": seq_len,
        "seed": 81394,
    }

    data = generate_input(**config)

    torch.cuda.synchronize()
    ref_output = ref_kernel(clone_data(data)).to(torch.float32)
    torch.cuda.synchronize()
    tilelang_output = custom_kernel(clone_data(data)).to(torch.float32)
    torch.cuda.synchronize()

    torch.testing.assert_close(ref_output, tilelang_output, atol=1e-2, rtol=1e-2)
    print("✅ Tilelang and Torch match")


def run_regression_perf(
    d_hidden=7168, d_expert=2048, n_routed_experts=8, n_shared_experts=1, n_experts_per_token=4, batch_size=1, seq_len=8192
):
    config = {
        "dhidden": d_hidden,
        "dexpert": d_expert,
        "nroutedexperts": n_routed_experts,
        "nsharedexperts": n_shared_experts,
        "nexpertspertoken": n_experts_per_token,
        "bs": batch_size,
        "seqlen": seq_len,
        "seed": 81394,
    }
    from tilelang.profiler import do_bench

    data = generate_input(**config)

    x, weights, config = data

    dtype_str = "float16"

    shared_kernel = moe_forward_tilelang_shared(
        config["d_hidden"],
        config["d_expert"],
        config["n_shared_experts"],
        dtype=dtype_str,
        num_tokens=config["batch_size"] * config["seq_len"],
    )
    routed_kernel = moe_forward_tilelang_routed(
        config["d_hidden"],
        config["d_expert"],
        config["n_routed_experts"],
        dtype=dtype_str,
        group_sum=config["batch_size"] * config["seq_len"] * config["n_experts_per_token"],
        group_count=config["n_routed_experts"],
        block_token=128,
        block_dhidden=128,
        block_dexpert=128,
        threads=256,
        num_stages=1,
        k_pack=1,
        coalesced_width=2,
    )

    moe = MoE(config, shared_kernel, routed_kernel, weights, padding_M=128)
    batch_size, seq_len, hidden_dim = x.shape
    expert_indices, expert_scores = moe.gating_network(x)
    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1)
    x_flat = x.view(-1, hidden_dim)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = moe.config["n_experts_per_token"]
    token_idxs = idxs // num_per_tok
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x_flat[exp_token_idxs]
        moe.stacked_expert_tokens[start_idx:end_idx] = expert_tokens
        moe.stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
        moe.stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]
    group_sizes = torch.tensor(counts, dtype=torch.int32, device=moe.device)
    group_offset = torch.tensor(tokens_per_expert - counts, dtype=torch.int32, device=moe.device)
    group_padded_offsets = [0 for _ in range(len(group_sizes))]
    for i in range(1, len(group_sizes)):
        group_padded_offsets[i] = group_padded_offsets[i - 1] + math.ceil((counts[i - 1] + 1) / moe.padding_M) * moe.padding_M
    block_token = 128
    M = (
        math.ceil(moe.config["batch_size"] * moe.config["seq_len"] * moe.config["n_experts_per_token"] / block_token)
        + moe.config["n_routed_experts"]
    )
    group_idx_for_bx = [0 for _ in range(M)]
    for bx in range(M):
        m_start_padded = bx * block_token
        for i in range(moe.config["n_routed_experts"]):
            if m_start_padded >= group_padded_offsets[i]:
                group_idx_for_bx[bx] = i
    group_padded_offsets = torch.tensor(group_padded_offsets, dtype=torch.int32, device=moe.device)
    group_idx_for_bx = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=moe.device)

    def run_shared_kernel_only():
        moe.routed_kernel(
            moe.stacked_expert_tokens,
            moe.stacked_expert_w_gate,
            moe.stacked_expert_w_up,
            moe.stacked_expert_w_down,
            moe.stacked_expert_weights,
            group_sizes,
            group_offset,
            group_padded_offsets,
            group_idx_for_bx,
            moe.up_logits_routed,
            moe.expert_output_routed,
        )

    def run_routed_kernel_only():
        moe.routed_kernel(
            moe.stacked_expert_tokens,
            moe.stacked_expert_w_gate,
            moe.stacked_expert_w_up,
            moe.stacked_expert_w_down,
            moe.stacked_expert_weights,
            group_sizes,
            group_offset,
            group_padded_offsets,
            group_idx_for_bx,
            moe.up_logits_routed,
            moe.expert_output_routed,
        )

    return do_bench(run_routed_kernel_only, warmup=100, rep=1000, backend="cupti")


if __name__ == "__main__":
    main()
