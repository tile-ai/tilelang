from __future__ import annotations

import tilelang
import argparse
from itertools import accumulate
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist
from tilelang.distributed import perf_fn
from sp_ag_attention_intra_node import (
    create_sp_ag_attention_context_intra_node,
    fused_sp_ag_attn_intra_node,
)

tilelang.disable_cache()


class FusedSequenceParallelAttn(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        batch_size: int,
        q_head: int,
        kv_head: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        head_dim: int,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        is_causal=True,
        enable_zig_zag=True,
        enable_specialized=False,
        allocator=None,
    ):
        super(FusedSequenceParallelAttn, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()

        self.batch_size = batch_size
        self.q_head = q_head
        self.kv_head = kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.head_dim = head_dim

        assert max_seqlen_q % self.world_size == 0 and max_seqlen_q % self.world_size == 0, (
            f"sequence length should be multiple of world_size({self.world_size})"
        )
        self.max_q_shard_len = self.max_seqlen_q // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device
        self.is_causal = is_causal
        self.enable_zig_zag = enable_zig_zag
        self.enable_specialized = enable_specialized
        self.allocator = allocator

        self.ctx = create_sp_ag_attention_context_intra_node(
            self.batch_size,
            self.q_head,
            self.kv_head,
            self.max_seqlen_k,
            self.max_q_shard_len,
            self.head_dim,
            self.input_dtype,
            self.output_dtype,
            self.rank,
            self.world_size,
            self.device,
            self.allocator,
        )

    def forward(self, q_shard, k_shards, v_shards, cu_seqlens_q, cu_seqlens_k, print_source=False):
        total_q_shard = cu_seqlens_q[-1]
        output_buffer = self.ctx.attn_output_buffer[:total_q_shard]

        fused_sp_ag_attn_intra_node(
            self.ctx,
            q_shard,
            k_shards,
            v_shards,
            output_buffer,
            cu_seqlens_q,
            cu_seqlens_k,
            self.max_q_shard_len,
            self.max_seqlen_k,
            self.rank,
            self.world_size,
            self.is_causal,
            self.enable_zig_zag,
            self.enable_specialized,
            print_source,
        )

        return output_buffer


class TorchSequenceParallelAttn(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        batch_size: int,
        q_head: int,
        kv_head: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        head_dim: int,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        is_causal=True,
        enable_zig_zag=True,
    ):
        super(TorchSequenceParallelAttn, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()

        self.batch_size = batch_size
        self.q_head = q_head
        self.kv_head = kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.head_dim = head_dim

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.device = device
        self.is_causal = is_causal
        self.enable_zig_zag = enable_zig_zag
        assert self.input_dtype == self.output_dtype

        self.max_q_shard_len = max_seqlen_q // self.world_size
        self.max_kv_shard_ken = max_seqlen_q // self.world_size
        assert max_seqlen_q % self.world_size == 0 and max_seqlen_q % self.world_size == 0, (
            f"sequence length should be multiple of world_size({self.world_size})"
        )

        self.ag_k_buffer: torch.Tensor = torch.empty(
            self.batch_size * self.max_seqlen_k,
            self.kv_head,
            self.head_dim,
            dtype=self.input_dtype,
            device=self.device,
        )
        self.ag_v_buffer: torch.Tensor = torch.empty(
            self.batch_size * self.max_seqlen_k,
            self.kv_head,
            self.head_dim,
            dtype=self.input_dtype,
            device=self.device,
        )

    def forward(self, q_shard, k_shard, v_shard, cu_seqlens_q, cu_seqlens_k):
        # construct casual mask with offset
        def _gen_mask(offset, q_shard_len, kv_len):
            if self.is_causal:
                mask = torch.zeros((q_shard_len, kv_len), dtype=torch.bool, device=self.device)
                mask[:, : offset + q_shard_len] = True
                if offset < kv_len:
                    mask[:, offset : offset + q_shard_len].tril_()
                return mask
            return None

        batch_size = cu_seqlens_q.shape[0] - 1
        total_q_shard, q_head, head_dim = q_shard.shape
        total_kv_shard, kv_head, head_dim = k_shard.shape
        out_list = []
        for i in range(batch_size):
            cu_seqlens_q_start = cu_seqlens_q[i].item()
            cu_seqlens_q_end = cu_seqlens_q[i + 1].item()
            q_shard_len = cu_seqlens_q_end - cu_seqlens_q_start
            q_len = q_shard_len * self.world_size

            cu_seqlens_k_start = cu_seqlens_k[i].item() // self.world_size
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item() // self.world_size
            kv_shard_len = cu_seqlens_k_end - cu_seqlens_k_start
            kv_len = kv_shard_len * self.world_size

            if self.enable_zig_zag:
                half_q_shard_len = q_shard_len // 2
                half_kv_shard_len = kv_shard_len // 2

                q0_shard = q_shard[cu_seqlens_q_start : cu_seqlens_q_start + half_q_shard_len, :, :].clone()
                q1_shard = q_shard[cu_seqlens_q_start + half_q_shard_len : cu_seqlens_q_end, :, :].clone()

                q0_shard_permute = torch.permute(q0_shard.reshape(1, half_q_shard_len, q_head, head_dim), (0, 2, 1, 3)).contiguous()
                q1_shard_permute = torch.permute(q1_shard.reshape(1, half_q_shard_len, q_head, head_dim), (0, 2, 1, 3)).contiguous()

                k0_shard = k_shard[cu_seqlens_k_start : cu_seqlens_k_start + half_kv_shard_len, :, :].clone()
                k1_shard = k_shard[cu_seqlens_k_start + half_kv_shard_len : cu_seqlens_k_end, :, :].clone()
                v0_shard = v_shard[cu_seqlens_k_start : cu_seqlens_k_start + half_kv_shard_len, :, :].clone()
                v1_shard = v_shard[cu_seqlens_k_start + half_kv_shard_len : cu_seqlens_k_end, :, :].clone()

                buffer_size = half_kv_shard_len * kv_head * head_dim * self.world_size

                ag_k0 = self.ag_k_buffer.reshape(-1)[:buffer_size].reshape(half_kv_shard_len * self.world_size, kv_head, head_dim)
                ag_k1 = self.ag_k_buffer.reshape(-1)[buffer_size : 2 * buffer_size].reshape(
                    half_kv_shard_len * self.world_size, kv_head, head_dim
                )
                ag_v0 = self.ag_v_buffer.reshape(-1)[:buffer_size].reshape(half_kv_shard_len * self.world_size, kv_head, head_dim)
                ag_v1 = self.ag_v_buffer.reshape(-1)[buffer_size : 2 * buffer_size].reshape(
                    half_kv_shard_len * self.world_size, kv_head, head_dim
                )
                torch.distributed.all_gather_into_tensor(
                    ag_k0,
                    k0_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_k1,
                    k1_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_v0,
                    v0_shard,
                    group=self.pg,
                )
                torch.distributed.all_gather_into_tensor(
                    ag_v1,
                    v1_shard,
                    group=self.pg,
                )
                ag_k1 = ag_k1.reshape(self.world_size, half_kv_shard_len, kv_head, head_dim)
                ag_k1 = torch.flip(ag_k1, [0]).reshape(self.world_size * half_kv_shard_len, kv_head, head_dim)
                ag_k = torch.cat((ag_k0, ag_k1), dim=0)
                ag_k = torch.permute(ag_k.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_k = ag_k.repeat_interleave(q_head // kv_head, -3)

                ag_v1 = ag_v1.reshape(self.world_size, half_kv_shard_len, kv_head, head_dim)
                ag_v1 = torch.flip(ag_v1, [0]).reshape(self.world_size * half_kv_shard_len, kv_head, head_dim)
                ag_v = torch.cat((ag_v0, ag_v1), dim=0)
                ag_v = torch.permute(ag_v.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_v = ag_v.repeat_interleave(q_head // kv_head, -3)

                offset_q0 = half_q_shard_len * self.rank
                offset_q1 = q_len - half_q_shard_len * (self.rank + 1)
                prefix = kv_len - q_len
                mask0 = _gen_mask(prefix + offset_q0, half_q_shard_len, kv_len)
                mask1 = _gen_mask(prefix + offset_q1, half_q_shard_len, kv_len)
                out0 = torch.nn.functional.scaled_dot_product_attention(q0_shard_permute, ag_k, ag_v, attn_mask=mask0)
                out1 = torch.nn.functional.scaled_dot_product_attention(q1_shard_permute, ag_k, ag_v, attn_mask=mask1)
                out = torch.cat((out0, out1), dim=2)  # [1, q_head, q_shard_len, head_dim]
            else:
                cu_q_shard = q_shard[cu_seqlens_q_start:cu_seqlens_q_end, :, :].clone()
                cu_q_shard_permute = torch.permute(cu_q_shard.reshape(1, q_shard_len, q_head, head_dim), (0, 2, 1, 3)).contiguous()

                total_size = kv_len * kv_head * head_dim
                ag_k = self.ag_k_buffer.reshape(-1)[:total_size].reshape(kv_len, kv_head, head_dim)
                cu_k_shard = k_shard[cu_seqlens_k_start:cu_seqlens_k_end, :, :].clone()
                torch.distributed.all_gather_into_tensor(
                    ag_k,
                    cu_k_shard,
                    group=self.pg,
                )
                ag_v = self.ag_v_buffer.reshape(-1)[:total_size].reshape(kv_len, kv_head, head_dim)
                cu_v_shard = v_shard[cu_seqlens_k_start:cu_seqlens_k_end, :, :].clone()
                torch.distributed.all_gather_into_tensor(
                    ag_v,
                    cu_v_shard,
                    group=self.pg,
                )
                ag_k = torch.permute(ag_k.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_k = ag_k.repeat_interleave(q_head // kv_head, -3)
                ag_v = torch.permute(ag_v.reshape(1, kv_len, kv_head, head_dim), (0, 2, 1, 3)).contiguous()
                ag_v = ag_v.repeat_interleave(q_head // kv_head, -3)

                offset = self.rank * q_shard_len
                prefix = kv_len - q_len
                mask = _gen_mask(prefix + offset, q_shard_len, kv_len)
                out = torch.nn.functional.scaled_dot_product_attention(
                    cu_q_shard_permute, ag_k, ag_v, attn_mask=mask
                )  # [1, q_head, q_shard_len, head_dim]

            out = torch.permute(out.reshape(q_head, q_shard_len, head_dim), (1, 0, 2)).contiguous()
            out_list.append(out)

        output = torch.cat(out_list)

        return output


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    device = "cuda"

    batch_size = args.batch_size
    q_head = args.q_head
    kv_head = args.kv_head
    max_seqlen_q = args.max_seqlen_q
    max_seqlen_k = args.max_seqlen_k
    head_dim = args.head_dim
    is_causal = args.is_causal
    enable_zig_zag = args.zig_zag
    enable_specialized = args.enable_specialized
    seqlens_q = args.seqlens_q
    cu_seqlens_q_list = [0] + list(accumulate(seqlens_q))
    seqlens_k = args.seqlens_k
    cu_seqlens_k_list = [0] + list(accumulate(seqlens_k))

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single node for now"
    allocator = tilelang.get_allocator(
        size=2**30, device=device, is_distributed=True, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )

    cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=device)
    cu_seqlens_q = cu_seqlens_q // num_local_ranks
    cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=device)

    q_shard = tilelang.tensor((cu_seqlens_q[-1], q_head, head_dim), dtype=dtype, allocator=allocator).normal_(mean=0.0, std=0.5)
    k_shards = tilelang.tensor(
        (cu_seqlens_k[-1] // num_local_ranks, kv_head, head_dim), dtype=dtype, allocator=allocator, return_peers=True
    )
    v_shards = tilelang.tensor(
        (cu_seqlens_k[-1] // num_local_ranks, kv_head, head_dim), dtype=dtype, allocator=allocator, return_peers=True
    )
    k_shards[local_rank].normal_(mean=0.0, std=0.5)
    v_shards[local_rank].normal_(mean=0.0, std=0.5)

    dist.barrier()

    tilescale_module = FusedSequenceParallelAttn(
        group,
        batch_size,
        q_head,
        kv_head,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        dtype,
        dtype,
        device,
        is_causal,
        enable_zig_zag,
        enable_specialized,
        allocator=allocator,
    )
    torch_module = TorchSequenceParallelAttn(
        group,
        batch_size,
        q_head,
        kv_head,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        dtype,
        dtype,
        device,
        is_causal,
        enable_zig_zag,
    )

    tilescale_out = tilescale_module(q_shard, k_shards, v_shards, cu_seqlens_q, cu_seqlens_k, print_source=True)
    print(f"tilescale_out: {tilescale_out.shape}")

    torch_out = torch_module(q_shard, k_shards[local_rank], v_shards[local_rank], cu_seqlens_q, cu_seqlens_k)
    print(f"torch_out: {torch_out.shape}")

    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(torch_out, tilescale_out, atol=atol, rtol=rtol):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"torch_out: {torch_out}, tilelang_out: {tilescale_out}")

    _, tl_t = perf_fn(lambda: tilescale_module(q_shard, k_shards, v_shards, cu_seqlens_q, cu_seqlens_k), warmup=5, rep=5)

    print(f"rank {local_rank} tilescale time: {tl_t:.2f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--q_head", type=int, default=32, help="num q heads")
    parser.add_argument("--kv_head", type=int, default=8, help="num kv heads")
    parser.add_argument("--max_seqlen_q", type=int, default=8192, help="max sequence length of q")
    parser.add_argument("--max_seqlen_k", type=int, default=12288, help="max sequence length of k/v")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim")
    parser.add_argument("--seqlens_q", type=int, nargs="+", default=[4096, 8192], help="sequence lengths of q")
    parser.add_argument("--seqlens_k", type=int, nargs="+", default=[6144, 12288], help="sequence lengths of k/v")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument(
        "--zig-zag",
        "--no-zig-zag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable zig zag opt",
    )
    parser.add_argument(
        "--enable-specialized",
        "--disable-specialized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable specialized optimized version",
    )

    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
