# DeepEP

To install and compare with the original DeepEP implementation, please refer to https://github.com/deepseek-ai/DeepEP.

## TODO
- [x] Intranode Normal Mode
- [] Internode Normal Mode
- [] Low-latency Mode

# Benchmark Results

The table below shows a latency and bandwidth comparison for DeepEP and TileScale on the same NVLink hardware (as reported by the example):

*Measured on: 8xH100 on NVL, 10 channels, 8 ranks, 32 experts, 7168 hidden, 4096 tokens.*

## Normal Mode

| Method      | Dispatch Time (ms) | Dispatch Bandwidth (GB/s) | Combine Time (ms) | Combine Bandwidth (GB/s) |
|-------------|--------------------|---------------------------|-------------------|--------------------------|
| DeepEP      | 1.0045             | 328.97                    | 1.1552            | 287.14                   |
| TileScale   | 1.0720             | 308.25                    | 1.0809            | 306.86                   |

# Intra-node Introduction

This example implements DeepEP’s intra‑node (NVLink) dispatch/combine using TileScale kernels.
z
The intra‑node path lives under `intranode/` and provides a minimal public API that mirrors DeepEP’s behavior for NVLink‑connected ranks.

## Overview

- Scope: intra‑node (NVLink) only; all ranks must be within one node and NVLink‑visible.
- Topology: experts are evenly partitioned across ranks (`num_experts % num_ranks == 0`).
- Datatypes: inputs are `torch.bfloat16`; routing `topk_idx` is `torch.int64`; `topk_weights` is `torch.float32`.
- Channels: each channel uses 2 SMs (send/recv). With default `num_sms=20`, there are `num_channels=10`.

## Public API (intranode)

- `intranode.get_dispatch_layout(topk_idx, num_experts, num_ranks)`
  - Computes the routing layout entirely on device.
  - Returns `(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)`:
    - `num_tokens_per_rank`: `[num_ranks]`, `torch.int32` — tokens destined for each rank.
    - `num_tokens_per_expert`: `[num_experts]`, `torch.int32` — tokens per expert.
    - `is_token_in_rank`: `[num_tokens, num_ranks]`, `torch.bool` — whether a token should be sent to a rank.

- `intranode.intranode_dispatch(...)`
  - Sends selected tokens to destination ranks over NVLink and prepares a reusable communication handle.
  - Non‑cached mode (no handle input):
    - Inputs: `rank`, `allocator`, `symm_buffers`, MoE counters, `x`, `config`, `num_tokens_per_rank`, `is_token_in_rank`, `num_tokens_per_expert`, `topk_idx`, `topk_weights`, `expert_alignment`.
    - Returns: `(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle)`.
  - Cached mode (pass handle):
    - Reuses precomputed matrices/buffers and returns only `recv_x`.

- `intranode.intranode_combine(rank, allocator, symm_buffers, x, config, handle, topk_weights)`
  - Reduces contributions back to origin ranks (sum, no weighting) and returns reduced weights for external use.
  - Returns `(recv_x, recv_topk_weights)`.

Convenience wrapper used by examples/tests:

- `EPBuffer` in `buffer.py`
  - Exposes the interface for the functions above via methods: `get_dispatch_layout`, `dispatch`, `combine`.
  - Manages TileScale allocator, symmetric buffers, and recommended kernel configs.

## Core Data Structures and Handle

- `rank_prefix_matrix` (num_ranks × num_ranks): cumulative per‑rank token counts; used to compute global offsets for receiver writes.
- `channel_prefix_matrix` (num_ranks × num_channels): per‑channel cumulative counts for each destination rank; senders split work across channels.
- `recv_channel_prefix_matrix` (num_ranks × num_channels): receiver‑side channel offsets populated during dispatch; consumed by combine.
- `send_head` (num_recv_tokens × num_ranks): per received token, expected per‑rank head index in the receiver’s ring buffer. Negative values encode “not yet present” via `-head-1` convention.
- `recv_src_idx` (num_recv_tokens): original source token index; forwarded during dispatch and used by combine senders to tag return traffic.
- `is_token_in_rank` (num_tokens × num_ranks): boolean mask whether a token contributes to a destination rank; reused in cached dispatch.
- `moe_recv_counter(_mapped)`: pinned host + device mapping, total tokens the current rank will receive.
- `moe_recv_expert_counter(_mapped)`: per‑local‑expert received counts (rounded up to `expert_alignment`).
- Symmetric ring buffers per channel/rank:
  - Metadata: `channel_start_offset`, `channel_end_offset`, `channel_head_idx`, `channel_tail_idx`.
  - Payload: `channel_x_buffers`, `channel_src_idx_buffers`, `channel_topk_idx_buffers`, `channel_topk_weights_buffers`.

Dispatch returns the handle:
`(rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)`
which can be reused for cached re‑dispatch and is required by the combine stage.

## Kernel Responsibilities (high level)

- Layout
  - `get_dispatch_layout_kernel`: counts per‑rank/per‑expert and builds `is_token_in_rank` in one device pass.

- Notify + Dispatch (A2A send)
  - `notify_dispatch_kernel`: computes per‑rank and per‑channel prefixes, writes MoE counters, and zeros the 4 symmetric metadata buffers.
  - `dispatch_kernel`: senders push `x`, `src_idx`, and remapped `topk_idx`/`topk_weights` to remote buffers; receivers drain via head/tail indices and assemble `recv_x`, `recv_topk_idx`, `recv_topk_weights`, plus `recv_channel_prefix_matrix`. Also fills `send_head` used by combine.
  - Cached variants (`cached_notify_dispatch_kernel`, `cached_dispatch_kernel`) reuse matrices/handle and only clear or advance necessary state.

- Notify + Combine (reduce back)
  - `cached_notify_combine_kernel`: recalculates `send_head` expectations and zeros `channel_head_idx`/`channel_tail_idx` for the combine round.
  - `combine_kernel`: senders return expert outputs; receivers reduce by sum per token. `recv_topk_weights` is the sum of returned weights per token. Requires `hidden % 8 == 0` for vectorized access on the receiver side.

## Configuration and Tuning

- `utils.Config` provides recommended values for `num_max_nvl_chunked_send_tokens` and `num_max_nvl_chunked_recv_tokens` per `num_ranks`. These control per‑round trunk sizes and receiver buffer depth per channel.
- `EPBuffer.num_sms` controls total SMs assigned to high‑throughput kernels. Channels = `num_sms // 2` (one send SM + one recv SM per channel).
- `expert_alignment` pads per‑local‑expert MoE receive counters up to the specified multiple, which can be used to size per‑expert workspace.

## Execution Flow (non‑cached)

1) Prepare group and buffers
- Initialize the distributed process group.
- Construct `EPBuffer(group, num_nvl_bytes, num_topk, num_experts, hidden)`; it creates a TileScale distributed allocator, pre‑allocates symmetric buffers and counters, and selects recommended configs based on `num_ranks`.

2) Routing layout
- Call `EPBuffer.get_dispatch_layout(topk_idx)` to obtain `(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)`.
  - Inputs must satisfy: `topk_idx.dtype == torch.int64`, 2D contiguous; `num_experts > 0` and divisible by `num_ranks`.

3) Dispatch
- Call `EPBuffer.dispatch(x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment)`.
- Internals:
  - `notify_dispatch` computes prefix matrices, zeros channel metadata, and populates MoE counters (including per‑expert counts aligned to `expert_alignment`).
  - `dispatch_kernel` executes A2A via channels. For each token and destination rank:
    - `topk_idx` is remapped into local‑expert indices for the destination rank; non‑local selections become `-1` with weight `0`.
    - Sender writes `x`, `src_idx` (token id), and the remapped `topk_idx`/`topk_weights` into receiver buffers; receiver drains and assembles `recv_x`, `recv_topk_idx`, `recv_topk_weights`.
    - `send_head` is produced to orchestrate the subsequent combine.
- Returns `(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle)`.

4) Expert compute
- Run local experts on `recv_x` to produce expert outputs (shape `[num_recv_tokens, hidden]`).

5) Combine (reduce back to origin)
- Call `EPBuffer.combine(expert_out, handle, recv_topk_weights)`.
- Internals:
  - `cached_notify_combine` recomputes `send_head` expectations per token/rank and zeros receiver heads/tails.
  - `combine_kernel` sends expert outputs back and receiver reduces by sum. It also returns `recv_topk_weights` as the sum of returned weights per token, enabling external weighted aggregation if desired.
- Returns `(reduced_x, reduced_topk_weights)`.

6) Cached re‑dispatch (optional)
- For repeated communication with the same layout, pass `handle` back into `EPBuffer.dispatch(x, handle, ...)` to skip layout/notify work and return only `recv_x`.

## Usage

Quick start (intra‑node test):

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TILELANG_USE_DISTRIBUTED=1 python intranode/example_intranode.py \
  --num_ranks 8 --num_tokens 4096 --hidden 7168 --num_topk 8 --num_experts 32 [--cached]
```

Minimal pattern via EPBuffer:

```python
from buffer import EPBuffer
from tilelang.distributed.utils import init_dist
from utils import gen_inputs

rank, world_size, group = init_dist(local_rank, num_local_ranks)
buf = EPBuffer(group, num_nvl_bytes=1<<30, num_topk=8, num_experts=32, hidden=7168)

# Prepare inputs
x, topk_idx, topk_weights, _ = gen_inputs(num_tokens, 7168, 8, 32, world_size)

# 1) Layout
num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = buf.get_dispatch_layout(topk_idx)

# 2) Dispatch (non-cached)
recv_x, recv_topk_idx, recv_topk_weights, per_expert_counts, handle = buf.dispatch(
    x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment=1)

# 3) Expert compute on recv_x -> expert_out

# 4) Combine back
reduced_x, reduced_weights = buf.combine(expert_out, handle, recv_topk_weights)
```

## Notes and Limits

- Intra‑node only: ranks must be NVLink‑visible; current code asserts `num_ranks <= 8` and `num_experts % num_ranks == 0`.
- Combine requires `hidden % 8 == 0` for vectorized receiver loads/stores.
- `dispatch` currently targets BF16 paths. FP8 is not wired end‑to‑end.
- Combine reduces data by sum (no weighting). Reduced weights are returned to enable external weighting logic.
- Ensure `topk_idx` is contiguous, 2D, and `torch.int64`.
- Set `TILELANG_USE_DISTRIBUTED=1` to enable TileScale’s distributed runtime.

## Files

- `intranode/__init__.py` — re‑exports `get_dispatch_layout`, `intranode_dispatch`, `intranode_combine`.
- `intranode/get_dispatch_layout.py` — layout computation function and kernel.
- `intranode/dispatch.py` — notify and main dispatch kernels; host orchestration and cached variants.
- `intranode/combine.py` — notify for combine and main combine kernel; host orchestration.
- `buffer.py` — EPBuffer wrapper: allocator and symmetric buffers, public methods.
- `utils.py` — recommended configs and MoE counter helpers.

## Implementation Notes

- Negative offset encoding: senders write channel start/end offsets as `-value-1` so that a zero token count is distinguishable from an uninitialized `0`.
- Queue semantics: senders update `channel_tail_idx` with release semantics; receivers poll heads/tails with acquire/volatile loads to ensure visibility across PEs.
- `send_head` orchestration: combine waits until each contributing rank’s head meets the expected position for a token, ensuring all contributions are present before reduction.
