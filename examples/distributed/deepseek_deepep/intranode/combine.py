# For intranode only
# This op is distributed

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def cached_notify_combine_kernel(
    num_ranks,
    num_sms,
):
    num_channels = num_sms // 2
    threads = max(128, 32 * num_ranks)

    num_recv_tokens = T.dynamic('num_recv_tokens')

    @T.prim_func
    def cached_notify_combine_main(
        send_head: T.Tensor([num_recv_tokens, num_ranks], "int32"),
        ##### symm buffers #####
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
        barrier_signal: T.Tensor((num_ranks,), 'int32'),
    ):
        with T.Kernel(num_channels + 1, threads=threads) as bx:
            tx = T.get_thread_binding()

            if bx == 0:  # clearing channel_head/tail_idx buffers
                T.sync_blocks(barrier_signal)
                T.clear(channel_head_idx)
                T.clear(channel_tail_idx)
                T.barrier_blocks(barrier_signal)
            else:  # calculate send_head
                channel_id = bx - 1
                rank_id = tx // 32
                lane_id = tx % 32
                if rank_id >= num_ranks:
                    T.thread_return()

                tokens_per_channel = T.ceildiv(num_recv_tokens, num_channels)
                token_start_idx = T.min(tokens_per_channel * channel_id, num_recv_tokens)
                token_end_idx = T.min(token_start_idx + tokens_per_channel, num_recv_tokens)
               
                last_head = T.alloc_var('int32', init=2**25)  # a heuristic large number
                # todo: tilelang doesn't support reverse loop, we simulate this
                for i in T.serial(0, token_end_idx-token_start_idx, 32):
                    token_idx_tail = token_end_idx - i - 1
                    token_idx = token_idx_tail - lane_id
                    current_head = T.alloc_var('int32')
                    if token_idx >= token_start_idx:
                        T.ld(send_head[token_idx, rank_id], current_head, nc=True)
                    else:
                        current_head = -1
                    expected_head = T.alloc_var('int32')
                    expected_head = 0
                    for j in T.serial(T.min(32, token_idx_tail-token_start_idx + 1)):
                        head = T.tvm_warp_shuffle(-1, current_head, j, 32, 32)
                        if head < 0:
                            if lane_id == j:
                                expected_head = -last_head - 1
                        else:
                            last_head = head
                    if current_head < 0 and token_idx >= token_start_idx:
                        send_head[token_idx, rank_id] = expected_head
                
    return cached_notify_combine_main


def cached_notify_combine(
    num_ranks,
    num_sms,
    ##### symm buffers #####
    send_head: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    barrier_signal: torch.Tensor,    
    allocator,
    comm_stream=None
):
    kernel = cached_notify_combine_kernel(num_ranks, num_sms)
    kernel.initialize(allocator=allocator, stream=comm_stream.cuda_stream)

    kernel(send_head, channel_head_idx, channel_tail_idx, barrier_signal, stream=comm_stream.cuda_stream,
      skip_tensor_validation=True)  # reduce runtime overhead


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # use TMA later
        "tl.disable_warp_specialized": True}, 
    # debug_root_path='/home/wt/debug/combine'
)
def combine_kernel(
    num_ranks,
    num_max_send_tokens,  # config.num_max_nvl_chunked_send_tokens
    num_recv_buffer_tokens,  # config.num_max_nvl_chunked_recv_tokens
    hidden, 
    num_topk, 
    num_sms,
    dtype: str = 'bfloat16',
):
    num_tokens = T.dynamic('num_tokens')
    num_recv_tokens = T.dynamic('num_recv_tokens')

    num_channels = num_sms // 2
    threads = 768  # 24 warps
    warps = threads // 32
    warps_per_rank = warps // num_ranks  # 3
    threads_per_rank = threads // num_ranks  # 96
    TMABytesPerWarp = 4096
    smem_size = TMABytesPerWarp * (threads // 32)
    num_stages = 8

    assert hidden % 8 == 0  # manual vectorize on recv-side

    @T.prim_func
    def combine_main(
        rank: T.int32,
        # inputs
        x: T.Tensor([num_tokens, hidden], dtype),
        topk_weights: T.Tensor([num_tokens, num_topk], "float32"),
        src_idx: T.Tensor([num_tokens], "int32"),
        # todo: support bias as inputs
        # outputs
        recv_x: T.Tensor([num_recv_tokens, hidden], dtype),
        recv_topk_weights: T.Tensor([num_recv_tokens, num_topk], "float32"),
        # metadata
        rank_prefix_matrix: T.Tensor([num_ranks, num_ranks], "int32"),
        channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
        send_head: T.Tensor([num_recv_tokens, num_ranks], "int32"),
        # symm buffers
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),  # reuse, already zeroed
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),  # reuse, already zeroed
        channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden], dtype),
        channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
        channel_topk_weights_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id = tx % 32
            warp_id = tx // 32
            responsible_channel = bx // 2


            if bx % 2 == 0:  # sender
                send_rank_id = (responsible_channel + warp_id) % num_ranks
                send_warp_id_in_rank = warp_id // num_ranks

                # get tasks
                rank_offset = T.if_then_else(send_rank_id > 0, rank_prefix_matrix[send_rank_id-1, rank], 0)
                num_rank_tokens = rank_prefix_matrix[send_rank_id, rank] - rank_offset
                channel_offset = channel_prefix_matrix[send_rank_id, responsible_channel]
                num_channel_tokens=  T.if_then_else(
                    responsible_channel == num_channels - 1,
                    num_rank_tokens,
                    channel_prefix_matrix[send_rank_id, responsible_channel + 1]
                ) - channel_offset
                token_start_idx = rank_offset + channel_offset
                token_end_idx = token_start_idx + num_channel_tokens

                # Iterate over all tokens and send by trunk
                current_channel_tail_idx = T.alloc_var('int32')
                current_channel_tail_idx = 0
                token_idx = T.alloc_var('int32')
                token_idx = token_start_idx
                with T.While(token_idx < token_end_idx):
                    # Check destination queue emptiness, or wait a buffer to be released (rare cases)
                    num_round_tokens = T.min(num_max_send_tokens, token_end_idx - token_idx)
                    if T.elect_one_sync():
                        T.wait_ge(channel_head_idx[responsible_channel, rank], current_channel_tail_idx + num_round_tokens - num_recv_buffer_tokens, peer=send_rank_id)
                    T.sync_warp()

                    # Send by trunk
                    for i in T.serial(send_warp_id_in_rank, num_round_tokens, warps_per_rank):
                        # Get an empty slot
                        dst_slot_idx = T.alloc_var('int32')
                        dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens

                        # 1. copy data
                        T.put_warp(T.address_of(x[token_idx + i, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=send_rank_id, unroll_factor=4, enable_aggresive_vectorize=True)
                            
                        # 2. send src idx
                        idx = T.alloc_var('int32')
                        if T.elect_one_sync():
                            T.ld(src_idx[token_idx + i], idx, nc=True)
                            T.st(channel_src_idx_buffers[responsible_channel, rank, dst_slot_idx], idx,
                                dst_pe=send_rank_id)

                        # 3. send topk_weights
                        if num_topk > 0 and lane_id < num_topk:
                            weight = T.alloc_var('float32')
                            T.ld(topk_weights[token_idx + i, lane_id], weight, nc=True)
                            T.st(channel_topk_weights_buffers[responsible_channel, rank, dst_slot_idx, lane_id], weight,
                                dst_pe=send_rank_id)

                    token_idx += num_round_tokens
                    current_channel_tail_idx += num_round_tokens

                    # move tail index
                    T.sync_threads(send_rank_id, threads_per_rank)
                    if send_warp_id_in_rank == 0 and T.elect_one_sync():
                        T.st(channel_tail_idx[responsible_channel, rank], current_channel_tail_idx,
                            scope='sys', sem='release',
                            dst_pe=send_rank_id)
            
            else:  # receiver
                #? Why we must need scope='shared', not 'shared.dynamic' here?
                warp_channel_head_idx = T.alloc_shared([warps, num_ranks], 'int32', scope='shared')
                shared_channel_tail_idx = T.alloc_shared([32], 'int32', scope='shared')  #! workaround for illegal address
                warp_retired = T.alloc_shared([warps], 'bool', scope='shared')
                if tx < warps:
                    warp_retired[tx] = False
                if lane_id < num_ranks:
                    warp_channel_head_idx[warp_id, lane_id] = 0
                if tx < 32:
                    shared_channel_tail_idx[tx] = 0
                T.sync_threads()

                if tx < 32:  # one warp for moving the queue head
                    last_head = T.alloc_var('int32')
                    last_head = 0
                    with T.While(lane_id < num_ranks):
                        # check retired
                        retired = T.alloc_var('bool')
                        retired = True
                        for i in T.serial(1, warps):
                            retired = retired and warp_retired[i]
                        if retired:
                            T.loop_break()
                        
                        # Update queue tail
                        new_tail = T.alloc_var('int32')
                        T.ld(channel_tail_idx[responsible_channel, lane_id], new_tail, sem="acquire", scope="sys")
                        # Use release semantics to ensure receiver warps see the update
                        T.st(shared_channel_tail_idx[lane_id], new_tail, sem="release", scope="cta")  # todo: weaker sem pair

                        # Update minimum head
                        min_head = T.alloc_var('int32')
                        min_head = 2**31 - 1  # int32 max
                        for i in T.serial(1, warps):
                            if not warp_retired[i]:
                                min_head = T.min(min_head, warp_channel_head_idx[i, lane_id])
                        if min_head != 2**31 - 1 and min_head > last_head:
                            last_head = min_head
                            T.st(channel_head_idx[responsible_channel, lane_id], min_head, sem="relaxed", scope="sys")
                else:  # other warps for reduction
                    # All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`

                    # The same tokens as the dispatch process
                    num_tokens_per_channel = T.truncdiv(num_recv_tokens+num_channels-1, num_channels)
                    # todo: this is a workaround, as TVM has a bug when calculating safe ceildiv for tir.Var
                    token_start_idx = T.min(num_tokens_per_channel * responsible_channel, num_recv_tokens)
                    token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_recv_tokens)

                    # Iterate over all tokens and combine
                    for token_idx in T.serial(token_start_idx+warp_id-1, token_end_idx, warps-1):
                        # Read expected head
                        expected_head = T.alloc_var('int32')
                        expected_head = -1
                        if lane_id < num_ranks:
                            T.ld(send_head[token_idx, lane_id], expected_head, nc=True)

                        condvar = T.alloc_var('int32')
                        T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                        with T.While(T.warp_any(condvar <= expected_head and expected_head >= 0)):
                            T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                            T.loop_continue()
                        # can we simplify this ?
                        T.sync_warp()

                        # Broadcast current heads
                        num_topk_ranks = T.alloc_var('int32')
                        num_topk_ranks = 0
                        topk_ranks= T.alloc_local([num_ranks], 'int32')
                        slot_indices = T.alloc_local([num_ranks], 'int32')
                        for i in T.serial(num_ranks):
                            expected_head_i = T.tvm_warp_shuffle(-1, expected_head, i, 32, 32)
                            if expected_head_i >= 0:
                                slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens
                                topk_ranks[num_topk_ranks] = i
                                num_topk_ranks += 1

                        # Reduce data with pipeline
                        # todo: vectorize
                        recv_value = T.alloc_local([num_ranks, 8], dtype)
                        values = T.alloc_local([8], "float32")
        
                        for i in T.serial(lane_id, hidden // 8, 32):
                            T.clear(values)
                            for j in T.serial(num_topk_ranks):
                                for k in T.vectorized(8):
                                    T.ld(channel_x_buffers[responsible_channel, topk_ranks[j], slot_indices[j], i*8+k], recv_value[j, k], nc=True)
                                
                            # todo: support bias

                            # Reduce a2a results
                            for j in T.serial(num_topk_ranks):
                                for k in T.vectorized(8):
                                    values[k] += recv_value[j, k]
                            for j in T.vectorized(8):
                                recv_x[token_idx, i*8+j] = values[j]

                        # Reduce topk_weights
                        if lane_id < num_topk:
                            weight_sum = T.alloc_var('float32')
                            weight_sum = 0
                            for i in T.serial(num_topk_ranks):
                                weight = T.alloc_var('float32')
                                T.ld(channel_topk_weights_buffers[responsible_channel, topk_ranks[i], slot_indices[i], lane_id], weight, nc=True)
                                weight_sum += weight
                            recv_topk_weights[token_idx, lane_id] = weight_sum

                        # Update head
                        if lane_id < num_ranks:
                            warp_channel_head_idx[warp_id, lane_id] = T.if_then_else(
                                expected_head < 0,
                                -expected_head - 1,
                                expected_head + 1)

                    # Retired
                    T.sync_warp()
                    if T.elect_one_sync():
                        warp_retired[warp_id] = True

    return combine_main


@tilelang.engine.register_cuda_postproc
def _(code, _):
    if not 'void combine_main_kernel' in code:
        return code
    return r'''
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#include <tl_templates/cuda/distributed.h>
#include <tl_templates/cuda/sync.h>
#include <tl_templates/cuda/ldst.h>
uint64_t __constant__ meta_data[1024];
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void combine_main_kernel(int* __restrict__ channel_head_idx, int* __restrict__ channel_prefix_matrix, int* __restrict__ channel_src_idx_buffers, int* __restrict__ channel_tail_idx, float* __restrict__ channel_topk_weights_buffers, bfloat16_t* __restrict__ channel_x_buffers, int* __restrict__ rank_prefix_matrix, float* __restrict__ recv_topk_weights, bfloat16_t* __restrict__ recv_x, int* __restrict__ send_head, int* __restrict__ src_idx, float* __restrict__ topk_weights, bfloat16_t* __restrict__ x, int num_recv_tokens, int num_tokens, int rank);
extern "C" __global__ void __launch_bounds__(768, 1) combine_main_kernel(int* __restrict__ channel_head_idx, int* __restrict__ channel_prefix_matrix, int* __restrict__ channel_src_idx_buffers, int* __restrict__ channel_tail_idx, float* __restrict__ channel_topk_weights_buffers, bfloat16_t* __restrict__ channel_x_buffers, int* __restrict__ rank_prefix_matrix, float* __restrict__ recv_topk_weights, bfloat16_t* __restrict__ recv_x, int* __restrict__ send_head, int* __restrict__ src_idx, float* __restrict__ topk_weights, bfloat16_t* __restrict__ x, int num_recv_tokens, int num_tokens, int rank) {
  int current_channel_tail_idx = 0;
  int token_idx = 0;
  int dst_slot_idx = 0;
  __shared__ signed char warp_retired[24];
  __shared__ int warp_channel_head_idx[192];
  __shared__ int shared_channel_tail_idx[32];
  int last_head = 0;
  signed char retired = (signed char)0;
  int new_tail = 0;
  int min_head = 0;
  int idx = 0;
  int condvar = 0;
  int slot_indices[8];
  int topk_ranks[8];
  float values[8];
  bfloat16_t recv_value[64];
  float weight_sum = 0x0p+0f/*0.000000e+00*/;
  float weight = 0x0p+0f/*0.000000e+00*/;
  if ((((int)blockIdx.x) % 2) == 0) {
    int condval;
    if ((0 < (((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7))) {
      condval = rank_prefix_matrix[((((((((int64_t)((int)threadIdx.x)) >> (int64_t)5) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) & (int64_t)7) * (int64_t)8) + ((int64_t)rank)) - (int64_t)8)];
    } else {
      condval = 0;
    }
    int rank_offset = condval;
    int num_rank_tokens = (rank_prefix_matrix[(((((((int64_t)((int)threadIdx.x)) >> (int64_t)5) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) & (int64_t)7) * (int64_t)8) + ((int64_t)rank))] - rank_offset);
    int channel_offset = channel_prefix_matrix[(((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7) * 10) + (((int)blockIdx.x) >> 1))];
    int condval_1;
    if (((((int)blockIdx.x) >> 1) == 9)) {
      condval_1 = num_rank_tokens;
    } else {
      condval_1 = channel_prefix_matrix[((((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7) * 10) + (((int)blockIdx.x) >> 1)) + 1)];
    }
    int num_channel_tokens = (condval_1 - channel_offset);
    current_channel_tail_idx = 0;
    token_idx = (rank_offset + channel_offset);
    while (1) {
      if (!((token_idx < ((rank_offset + channel_offset) + num_channel_tokens)))) { break; }
      int num_round_tokens = min(4, (((rank_offset + channel_offset) + num_channel_tokens) - token_idx));
      if (cute::elect_one_sync()) {
        tl::wait_ge((tl::get_remote_base_ptr((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7)) + (tl::get_uintptr_t((&(channel_head_idx[(((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)8) + ((int64_t)rank))]))) - tl::get_remote_base_ptr(tl::get_rank()))), ((current_channel_tail_idx + num_round_tokens) - 256));
      }
      __syncwarp();
      for (int v = 0; v < ((((num_round_tokens + 2) - (((int)threadIdx.x) >> 8)) / 3) + ((((num_round_tokens + 2) - (((int)threadIdx.x) >> 8)) % 3) >> 31)); ++v) {
        dst_slot_idx = ((((v * 3) + (((int)threadIdx.x) >> 8)) + current_channel_tail_idx) & 255);
        if (0 <= (((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx)) {
          if ((((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx) < num_tokens) {
            if (0 <= rank) {
              if (rank < 8) {
                tl::cp_warp<7168, 4, true>((tl::get_remote_base_ptr((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7)) + (tl::get_uintptr_t((&(channel_x_buffers[((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)14680064) + (((int64_t)rank) * (int64_t)1835008)) + (((int64_t)dst_slot_idx) * (int64_t)7168))]))) - tl::get_remote_base_ptr(tl::get_rank()))), (&(x[(((((int64_t)v) * (int64_t)21504) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)8) * (int64_t)7168)) + (((int64_t)token_idx) * (int64_t)7168))])));
              }
            }
          }
        }
        if (cute::elect_one_sync()) {
          if (0 <= (((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx)) {
            if ((((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx) < num_tokens) {
              tl::ld<Semantic::WEAK, Scope::GPU, true, false>((&(src_idx[(((((int64_t)v) * (int64_t)3) + (((int64_t)((int)threadIdx.x)) >> (int64_t)8)) + ((int64_t)token_idx))])), idx);
            }
          }
          if (0 <= rank) {
            if (rank < 8) {
              tl::st<Semantic::WEAK, Scope::GPU, false>((tl::get_remote_base_ptr((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7)) + (tl::get_uintptr_t((&(channel_src_idx_buffers[((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + (((int64_t)rank) * (int64_t)256)) + ((int64_t)dst_slot_idx))]))) - tl::get_remote_base_ptr(tl::get_rank()))), idx);
            }
          }
        }
        if ((((int)threadIdx.x) & 31) < 8) {
          if (0 <= (((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx)) {
            if ((((v * 3) + (((int)threadIdx.x) >> 8)) + token_idx) < num_tokens) {
              tl::ld<Semantic::WEAK, Scope::GPU, true, false>((&(topk_weights[((((((int64_t)v) * (int64_t)24) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)8) * (int64_t)8)) + (((int64_t)token_idx) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31))])), idx);
            }
          }
          if (0 <= rank) {
            if (rank < 8) {
              tl::st<Semantic::WEAK, Scope::GPU, false>((tl::get_remote_base_ptr((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7)) + (tl::get_uintptr_t((&(channel_topk_weights_buffers[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384) + (((int64_t)rank) * (int64_t)2048)) + (((int64_t)dst_slot_idx) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31))]))) - tl::get_remote_base_ptr(tl::get_rank()))), idx);
            }
          }
        }
      }
      token_idx = (token_idx + num_round_tokens);
      current_channel_tail_idx = (current_channel_tail_idx + num_round_tokens);
      tl::__sync_thread_partial((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7), 96);
      if (((((int)threadIdx.x) >> 8) == 0) && cute::elect_one_sync()) {
        tl::st<Semantic::RELEASE, Scope::SYS, false>((tl::get_remote_base_ptr((((((int)threadIdx.x) >> 5) + (((int)blockIdx.x) >> 1)) & 7)) + (tl::get_uintptr_t((&(channel_tail_idx[(((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)8) + ((int64_t)rank))]))) - tl::get_remote_base_ptr(tl::get_rank()))), current_channel_tail_idx);
      }
    }
  } else {
    if (((int)threadIdx.x) < 24) {
      warp_retired[((int)threadIdx.x)] = (signed char)0;
    }
    if ((((int)threadIdx.x) & 31) < 8) {
      warp_channel_head_idx[(((((int)threadIdx.x) >> 5) * 8) + (((int)threadIdx.x) & 31))] = 0;
    }
    if (((int)threadIdx.x) < 32) {
      shared_channel_tail_idx[((int)threadIdx.x)] = 0;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      last_head = 0;
      while (1) {
        if (!((((int)threadIdx.x) < 8))) { break; }
        retired = (signed char)1;
        for (int i = 1; i < 24; ++i) {
          retired = ((signed char)(((bool)retired) && ((bool)warp_retired[i])));
        }
        if ((bool)retired) {
          break;
        }
        tl::ld<Semantic::ACQUIRE, Scope::SYS, false, false>((&(channel_tail_idx[(((((int)blockIdx.x) >> 1) * 8) + ((int)threadIdx.x))])), new_tail);
        tl::st<Semantic::RELEASE, Scope::CTA, false>((&(shared_channel_tail_idx[((int)threadIdx.x)])), new_tail);
        min_head = 2147483647;
        for (int i_1 = 1; i_1 < 24; ++i_1) {
          if (!((bool)warp_retired[i_1])) {
            min_head = min(min_head, warp_channel_head_idx[((i_1 * 8) + ((int)threadIdx.x))]);
          }
        }
        if ((min_head < 2147483647) && (last_head < min_head)) {
          last_head = min_head;
          tl::st<Semantic::RELAXED, Scope::SYS, false>((&(channel_head_idx[(((((int)blockIdx.x) >> 1) * 8) + ((int)threadIdx.x))])), min_head);
        }
      }
    } else {
      for (int v_1 = 0; v_1 < ((((min(((num_recv_tokens + 9) / 10), max((num_recv_tokens - (((num_recv_tokens + 9) / 10) * (((int)blockIdx.x) >> 1))), 0)) + 214748401) - (((int)threadIdx.x) >> 5)) / 23) - 9336886); ++v_1) {
        idx = -1;
        if ((((int)threadIdx.x) & 31) < 8) {
          tl::ld<Semantic::WEAK, Scope::GPU, true, false>((&(send_head[(((((((int64_t)v_1) * (int64_t)184) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + (min((((((int64_t)num_recv_tokens) + (int64_t)9) / (int64_t)10) * (((int64_t)((int)blockIdx.x)) >> (int64_t)1)), ((int64_t)num_recv_tokens)) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31)) - (int64_t)8)])), idx);
        }
        tl::ld<Semantic::ACQUIRE, Scope::CTA, false, false>((&(shared_channel_tail_idx[(((int)threadIdx.x) & 31)])), condvar);
        while (1) {
          if (!(__any_sync(-1, ((condvar <= idx) && (0 <= idx))))) { break; }
          tl::ld<Semantic::ACQUIRE, Scope::CTA, false, false>((&(shared_channel_tail_idx[(((int)threadIdx.x) & 31)])), condvar);
          continue;
        }
        __syncwarp();
        condvar = 0;
        for (int i_2 = 0; i_2 < 8; ++i_2) {
          int expected_head_i = __shfl_sync(-1, idx, i_2, 32);
          if (0 <= expected_head_i) {
            slot_indices[condvar] = (expected_head_i & 255);
            topk_ranks[condvar] = i_2;
            condvar = (condvar + 1);
          }
        }
        for (int v_2 = 0; v_2 < ((927 - (((int)threadIdx.x) & 31)) >> 5); ++v_2) {
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            *(float4*)(values + (i_3 * 4)) = make_float4(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
          }
          /// change 1 (major)
          for (int j = 0; j < condvar; ++j) {
            auto src = (&(channel_x_buffers[(((((((((int)blockIdx.x) >> 1) * 14680064) + (topk_ranks[j] * 1835008)) + (slot_indices[j] * 7168)) + (v_2 * 256)) + ((((int)threadIdx.x) & 31) * 8)))]));
            auto dst = &(recv_value[((((int64_t)j) * (int64_t)8))]);
            *reinterpret_cast<int4*>(dst) = __ldg(reinterpret_cast<int4*>(src));
          }
          ///
          for (int j_1 = 0; j_1 < condvar; ++j_1) {
            for (int k_1 = 0; k_1 < 2; ++k_1) {
              float4 __1;
                float4 v_ = *(float4*)(values + (k_1 * 4));
                float4 __2;
                uint2 v__1 = *(uint2*)(recv_value + ((((int64_t)j_1) * (int64_t)8) + (((int64_t)k_1) * (int64_t)4)));
                ((float2*)(&__2))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__1)));
                ((float2*)(&__2))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__1))+1));
                __1.x = (v_.x+__2.x);
                __1.y = (v_.y+__2.y);
                __1.z = (v_.z+__2.z);
                __1.w = (v_.w+__2.w);
              *(float4*)(values + (k_1 * 4)) = __1;
            }
          }
          /// change 2 (minor)
          // for (int j_2 = 0; j_2 < 2; ++j_2) {
          //   if ((((v_1 * 23) + min((((num_recv_tokens + 9) / 10) * (((int)blockIdx.x) >> 1)), num_recv_tokens)) + (((int)threadIdx.x) >> 5)) <= num_recv_tokens) {
          //     uint2 __3;
          //     float4 v__2 = *(float4*)(values + (j_2 * 4));
          //     (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(*(float2*)(&(v__2)));
          //     (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(*((float2*)(&(v__2))+1));
          //     *(uint2*)(recv_x + (((((((((int64_t)v_1) * (int64_t)164864) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)7168)) + (min((((((int64_t)num_recv_tokens) + (int64_t)9) / (int64_t)10) * (((int64_t)((int)blockIdx.x)) >> (int64_t)1)), ((int64_t)num_recv_tokens)) * (int64_t)7168)) + (((int64_t)v_2) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (((int64_t)j_2) * (int64_t)4)) - (int64_t)7168)) = __3;
          //   }
          // }
          if ((((v_1 * 23) + min((((num_recv_tokens + 9) / 10) * (((int)blockIdx.x) >> 1)), num_recv_tokens)) + (((int)threadIdx.x) >> 5)) <= num_recv_tokens) {
            int4 __3;
            (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(*(float2*)(values));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(*((float2*)(values)+1));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[2] = __float22bfloat162_rn(*((float2*)(values)+2));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[3] = __float22bfloat162_rn(*((float2*)(values)+3));
            *(int4*)(recv_x + (((((((((int64_t)v_1) * (int64_t)164864) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)7168)) + (min((((((int64_t)num_recv_tokens) + (int64_t)9) / (int64_t)10) * (((int64_t)((int)blockIdx.x)) >> (int64_t)1)), ((int64_t)num_recv_tokens)) * (int64_t)7168)) + (((int64_t)v_2) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (((int64_t)0) * (int64_t)4)) - (int64_t)7168)) = __3;
          }
          ///
        }
        if ((((int)threadIdx.x) & 31) < 8) {
          weight_sum = 0x0p+0f/*0.000000e+00*/;
          for (int i_4 = 0; i_4 < condvar; ++i_4) {
            if (0 <= slot_indices[i_4]) {
              if (slot_indices[i_4] < 256) {
                if (0 <= topk_ranks[i_4]) {
                  if (topk_ranks[i_4] < 8) {
                    tl::ld<Semantic::WEAK, Scope::GPU, true, false>((&(channel_topk_weights_buffers[(((((((int)blockIdx.x) >> 1) * 16384) + (topk_ranks[i_4] * 2048)) + (slot_indices[i_4] * 8)) + (((int)threadIdx.x) & 31))])), weight);
                  }
                }
              }
            }
            weight_sum = (weight_sum + weight);
          }
          recv_topk_weights[(((((((int64_t)v_1) * (int64_t)184) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + (min((((((int64_t)num_recv_tokens) + (int64_t)9) / (int64_t)10) * (((int64_t)((int)blockIdx.x)) >> (int64_t)1)), ((int64_t)num_recv_tokens)) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31)) - (int64_t)8)] = weight_sum;
          int condval_2;
          if ((idx < 0)) {
            condval_2 = ((0 - idx) - 1);
          } else {
            condval_2 = (idx + 1);
          }
          warp_channel_head_idx[(((((int)threadIdx.x) >> 5) * 8) + (((int)threadIdx.x) & 31))] = condval_2;
        }
      }
      __syncwarp();
      if (cute::elect_one_sync()) {
        warp_retired[(((int)threadIdx.x) >> 5)] = (signed char)1;
      }
    }
  }
}
'''


def intranode_combine(
    rank: int, 
    allocator, 
    symm_buffers,
    x,
    config,
    handle,
    topk_weights, 
    comm_stream=None
):
    assert handle is not None
    rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, _, send_head = handle  
    barrier_signal, _, _, _, _, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers, _, channel_topk_weights_buffers = symm_buffers

    # acquire_shapes
    num_tokens, hidden = x.shape
    _, num_topk = topk_weights.shape
    num_ranks, num_channels = channel_prefix_matrix.shape
    num_recv_tokens = send_head.shape[0]
    
    # notify combine
    cached_notify_combine(num_ranks, config.num_sms, send_head, channel_head_idx, channel_tail_idx, barrier_signal, allocator, comm_stream=comm_stream)

    # combine
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device='cuda')
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device='cuda')

    kernel = combine_kernel(
        num_ranks,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        hidden,
        num_topk,
        config.num_sms,
        dtype='bfloat16'
    )
    kernel.initialize(allocator=allocator, stream=comm_stream.cuda_stream)
    kernel(
        rank, 
        x,
        topk_weights,
        recv_src_idx,
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        channel_head_idx,
        channel_tail_idx,
        channel_x_buffers,
        channel_src_idx_buffers,
        channel_topk_weights_buffers,
        stream=comm_stream.cuda_stream,
        skip_tensor_validation=True
    )  # reduce runtime overhead
    compute_stream = torch.cuda.current_stream()
    compute_stream.wait_stream(comm_stream)
    return recv_x, recv_topk_weights
