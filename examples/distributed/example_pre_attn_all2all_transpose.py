import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.distributed import init_distributed, dtype_map
import argparse

tilelang.disable_cache()


def torch_sequence_all_to_all_transpose_reference(data_src, group):
    """
    PyTorch Distributed All-to-All Golden Reference with Transpose

    Input:  [BATCH_SIZE, SEQ_PER_PE, NUM_HEADS, HEAD_DIM] - partial sequence, full heads per PE
    Output: [BATCH_SIZE, HEADS_PER_PE, SEQ_LEN, HEAD_DIM] - partial heads, full sequence per PE

    Args:
        data_src: Input tensor on each PE
        group: Distributed process group

    Returns:
        Output tensor after all-to-all communication
    """
    world_size = dist.get_world_size(group)
    batch_size, seq_per_pe, num_heads, head_dim = data_src.shape
    seq_len = seq_per_pe * world_size
    heads_per_pe = num_heads // world_size

    # Step 1: Prepare input list for all_to_all
    input_list = []
    for pe_idx in range(world_size):
        start_head = pe_idx * heads_per_pe
        end_head = (pe_idx + 1) * heads_per_pe

        # Extract [BATCH_SIZE, SEQ_PER_PE, HEADS_PER_PE, HEAD_DIM] for target PE
        send_data = data_src[:, :, start_head:end_head, :].contiguous()
        input_list.append(send_data)

    # Step 2: Prepare output list for all_to_all
    output_list = []
    for _ in range(world_size):
        recv_data = torch.empty(batch_size, seq_per_pe, heads_per_pe, head_dim, dtype=data_src.dtype, device=data_src.device)
        output_list.append(recv_data)

    # Step 3: Execute all_to_all
    dist.all_to_all(output_list, input_list, group=group)

    # Step 4: Reorganize received data with transpose
    result = torch.empty(batch_size, heads_per_pe, seq_len, head_dim, dtype=data_src.dtype, device=data_src.device)

    for pe_idx in range(world_size):
        seq_start = pe_idx * seq_per_pe
        seq_end = (pe_idx + 1) * seq_per_pe
        # Transpose: [B, S, H, D] -> [B, H, S, D]
        transposed_data = output_list[pe_idx].transpose(1, 2)
        result[:, :, seq_start:seq_end, :] = transposed_data

    return result


def sequence_parallel_all_to_all_transpose(PE_num, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype="float16"):
    """
    Coarse-grained version with proper transpose handling
    Each block handles one (batch, head) combination and processes all sequence positions
    """
    SEQ_PER_PE = SEQ_LEN // PE_num
    HEADS_PER_PE = NUM_HEADS // PE_num

    # Fewer blocks: one per (batch, head) combination
    NUM_BLOCKS_X = BATCH_SIZE * HEADS_PER_PE

    @T.prim_func
    def main(
        data_src: T.Tensor((BATCH_SIZE, SEQ_PER_PE, NUM_HEADS, HEAD_DIM), dtype),
        data_dst: T.Tensor((BATCH_SIZE, HEADS_PER_PE, SEQ_LEN, HEAD_DIM), dtype),
        signal: T.Tensor((PE_num,), "uint64"),
    ):
        with T.Kernel(NUM_BLOCKS_X, PE_num, threads=128) as (bx, target_pe):
            tx = T.thread_binding(128, thread="threadIdx.x")

            mype = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()

            # Each block handles one (batch, head) combination
            batch_idx = bx // HEADS_PER_PE
            head_idx = bx % HEADS_PER_PE

            src_head_idx = target_pe * HEADS_PER_PE + head_idx
            dst_seq_start = mype[0] * SEQ_PER_PE

            if batch_idx < BATCH_SIZE and src_head_idx < NUM_HEADS:
                # Use for loop to handle each sequence position separately for proper transpose
                # This ensures data continuity and correct transpose mapping
                for seq_idx in T.serial(SEQ_PER_PE):
                    dst_seq_idx = dst_seq_start + seq_idx

                    # Transfer HEAD_DIM elements for this specific (batch, seq, head) combination
                    # From: data_src[batch_idx, seq_idx, src_head_idx, :] -> [HEAD_DIM]
                    # To:   data_dst[batch_idx, head_idx, dst_seq_idx, :] on target_pe
                    transfer_size = HEAD_DIM * 2  # float16 = 2 bytes

                    T.putmem_nbi_block(
                        T.address_of(data_dst[batch_idx, head_idx, dst_seq_idx, 0]),
                        T.address_of(data_src[batch_idx, seq_idx, src_head_idx, 0]),
                        transfer_size,
                        target_pe,
                    )

            T.fence()

            # Signaling: each block signals completion of SEQ_PER_PE work items
            if tx == 0:
                T.signal_op(
                    T.address_of(signal[mype[0]]),
                    1,  # Signal the number of sequence positions processed
                    T.Amo.SIGNAL_ADD,
                    target_pe,
                )
                T.fence()
                # Wait for all blocks to complete all sequence positions
                T.signal_wait_until(T.address_of(signal[target_pe]), T.CmpType.EQ, NUM_BLOCKS_X)

    return main


def verify_results(tilelang_output, torch_output, rank, tolerance=1e-3):
    """Verify TileLang output against PyTorch golden reference"""
    if not torch.allclose(tilelang_output, torch_output, atol=tolerance, rtol=tolerance):
        print(f"❌ PE {rank} Verification FAILED!")

        diff = torch.abs(tilelang_output - torch_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)

        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   TileLang shape: {tilelang_output.shape}")
        print(f"   PyTorch shape:  {torch_output.shape}")

        # Find position with maximum difference
        max_pos = torch.unravel_index(torch.argmax(diff), diff.shape)
        print(f"   Max diff position: {max_pos}")
        print(f"   TileLang value: {tilelang_output[max_pos]:.6f}")
        print(f"   PyTorch value:  {torch_output[max_pos]:.6f}")

        return False
    else:
        print(f"✅ PE {rank} Verification PASSED!")
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--dtype", default="float16", help="Data type")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--print_source", action="store_true", help="Print kernel source")
    return parser.parse_args()


@tilelang.testing.requires_distributed
def test_transpose_all_to_all_with_golden_reference():
    args = parse_args()

    # Initialize distributed environment
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    PE_num = WORLD_SIZE

    # Validate divisibility
    assert args.seq_len % PE_num == 0, f"seq_len {args.seq_len} must be divisible by PE_num {PE_num}"
    assert args.num_heads % PE_num == 0, f"num_heads {args.num_heads} must be divisible by PE_num {PE_num}"

    SEQ_PER_PE = args.seq_len // PE_num
    HEADS_PER_PE = args.num_heads // PE_num

    if RANK == 0:
        print("=== Transpose All-to-All with PyTorch Golden Reference ===")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_len}")
        print(f"Number of heads: {args.num_heads}")
        print(f"Head dimension: {args.head_dim}")
        print(f"PE count: {PE_num}")
        print(f"Sequence per PE: {SEQ_PER_PE}")
        print(f"Heads per PE: {HEADS_PER_PE}")

    # Compile TileLang kernel
    func = sequence_parallel_all_to_all_transpose(PE_num, args.batch_size, args.num_heads, args.seq_len, args.head_dim, args.dtype)
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})

    if RANK == 0:
        print("\nTileLang Kernel Source:")
        print(kernel.get_kernel_source())

    # Create test data - following example_all2all_qkv.py style
    dtype_torch = dtype_map[args.dtype]

    # Create input data: [BATCH_SIZE, SEQ_PER_PE, NUM_HEADS, HEAD_DIM] - random like example
    input_data = torch.rand([args.batch_size, SEQ_PER_PE, args.num_heads, args.head_dim], dtype=dtype_torch, device="cuda")

    print(f"PE {RANK} Input shape: {input_data.shape}")
    print(f"PE {RANK} Input sample: {input_data[0, 0, 0, :3]}")

    # === Test 1: PyTorch Distributed Golden Reference ===
    dist.barrier(TP_GROUP)
    torch_output = torch_sequence_all_to_all_transpose_reference(input_data, TP_GROUP)

    print("start compute tilelang output")

    # === Test 2: TileLang NVSHMEM Implementation ===
    def tilelang_all_to_all():
        # Create NVSHMEM tensors
        data_src = pynvshmem.nvshmem_create_tensor([args.batch_size, SEQ_PER_PE, args.num_heads, args.head_dim], dtype_torch)
        data_dst = pynvshmem.nvshmem_create_tensor([args.batch_size, HEADS_PER_PE, args.seq_len, args.head_dim], dtype_torch)
        signal = pynvshmem.nvshmem_create_tensor([PE_num], torch.uint64)

        # Initialize data
        data_src.copy_(input_data)
        data_dst.fill_(0.0)
        signal.fill_(0)

        # Execute kernel
        kernel(data_src, data_dst, signal)

        return data_dst

    dist.barrier(TP_GROUP)
    tilelang_output = tilelang_all_to_all()

    # === Verification ===
    print(f"PE {RANK} Starting verification...")

    print(f"PE {RANK} PyTorch output shape: {torch_output.shape}")
    print(f"PE {RANK} TileLang output shape: {tilelang_output.shape}")
    print(f"PE {RANK} PyTorch output sample: {torch_output[0, 0, 0, :3]}")
    print(f"PE {RANK} TileLang output sample: {tilelang_output[0, 0, 0, :3]}")

    verify_results(tilelang_output, torch_output, RANK)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    test_transpose_all_to_all_with_golden_reference()
