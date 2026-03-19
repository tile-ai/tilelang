import tilelang
import tilelang.language as T
from tilelang.quantize import _tir_u8_to_f4_to_bf16
from tilelang import tvm as tvm
from tvm import DataType
import torch
from dequantize_utils import torch_convert_bit_twiddling, assert_similar
from tilelang.autotuner import set_autotune_inputs
import argparse


def get_configs():
    """
    Generate a list of hyperparameter configuration dictionaries for tuning.
    """
    import itertools

    iter_params = dict(
        block_M=[128],
        block_N=[64, 128, 256],
        block_K=[128],
        num_stages=[0, 1, 2],
        threads=[128, 256, 512],
        split=[1],
    )
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_configs())
@tilelang.jit
def matmul(
    A,
    B,
    Scale,
    Bias,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    topk: int = 4,
    in_dtype: T.dtype = T.bfloat16,
    out_dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
    source_format: T.dtype = T.uint32,
    num_bits: int = 4,
    scale_size: int = 32,
    fast_dequant: bool = True,
    with_bias: bool = False,
    block_M: int = 128,
    block_N: int = 256,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    split: int = 1,
):
    """
    Construct and return a grouped (Mixture-of-Experts) matrix-multiply TIR kernel.
    """

    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    M, K = T.const("M K")
    E, N, _ = T.const("E N _")
    _E, _N, _SK = T.const("_E _N _SK")
    __E, __N = T.const("__E __N")
    _TW = T.const("_TW")
    _PM = T.const("_PM")
    _EI = T.const("_EI")
    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[E, N, _], storage_dtype]
    Scale: T.Tensor[[_E, _N, _SK], storage_dtype]
    Bias: T.Tensor[[__E, __N], out_dtype]
    topk_weights: T.Tensor[[_TW], out_dtype]
    sorted_token_ids: T.Tensor[[_PM], T.int32]
    expert_ids: T.Tensor[[_EI], T.int32]

    Block_QK = block_K // num_elems_per_byte
    padding_M = _PM
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, Block_QK)
    Bias_shared_shape = block_N
    B_dequantize_shared_shape = (block_N, block_K)
    assert K % (block_K * split) == 0

    from tilelang.quantize import get_mxfp_intrin_group

    # fast_dequant_bf16_fp4_twiddling
    mxfp_intrin_info = get_mxfp_intrin_group(
        out_dtype=in_dtype,
        source_format=source_format,
        source_bit=num_bits,
        storage_dtype=storage_dtype,
        use_twiddling=True,
    )
    import_source = mxfp_intrin_info["c_source"]
    func_name = mxfp_intrin_info["func_name"]
    assert import_source is not None, "mxfp_intrin_info is not found"
    assert func_name is not None, "mxfp_intrin_info is not found"
    import_source = import_source

    # the dequant part is the same as in dequant_gemm
    def get_fast_dequant_twiddling_func(in_dtype="fp4", out_dtype=T.bfloat16):
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale_shared, k):
            # import fast_dequantize plugin
            T.import_source(import_source)

            tx = T.get_thread_binding()

            B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
            B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
            Scale_local_thread = T.alloc_local((1,), storage_dtype)
            Scale_local_thread_exponent = T.alloc_local((1,), out_dtype)

            for i in T.serial(0, block_N * block_K // threads // local_size):
                # First, load data from share memory to register.
                # Prepare for dequant.
                index_base = i * threads * local_compress_size + tx * local_compress_size
                for v in T.vectorized(0, local_compress_size):
                    index = index_base + v
                    B_local_thread[v] = B_shared[index // Block_QK, index % Block_QK]
                index_scale = index_base // (scale_size // num_elems_per_byte)
                si = index_scale // (block_K // scale_size)
                sj = index_scale % (block_K // scale_size)
                Scale_local_thread[0] = Scale_shared[si, k * block_K // scale_size + sj]
                Scale_local_thread_exponent[0] = T.shift_left(1, (Scale_local_thread[0]))

                # Then, dequant.
                T.call_extern(
                    func_name,
                    T.access_ptr(B_local_thread, "r"),
                    T.access_ptr(B_dequantize_local_thread, "w"),
                    1,
                    dtype=out_dtype,
                )

                # Finally, store the dequantized data to shared memory.
                for v in T.Parallel(local_size):
                    B_dequantize_local_thread[v] *= Scale_local_thread_exponent[0]

                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequantize_shared[index // block_K, index % block_K] = B_dequantize_local_thread[v]

        return fast_dequant_bf16_fp4_twiddling

    def get_simple_dequant_func(in_dtype="fp4", out_dtype=T.bfloat16):
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared, Scale_shared, k):
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, out_dtype)

            T.copy(B_shared, B_local)
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                    num_bits,
                    B_local[i, j // num_elems_per_byte],
                    j % num_elems_per_byte,
                    Scale_shared[
                        i, k * block_K // scale_size + j // scale_size
                    ],  # Scale is the exponential part, within the representation of uint8
                    dtype=out_dtype,
                ) * T.shift_left(1, (Scale_shared[i, k * block_K // scale_size + j // scale_size]))
            T.copy(B_dequantize_local, B_dequantize_shared)

        return simple_dequant_bf16_fp4

    C = T.empty((M, topk, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(padding_M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
        B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
        Bias_shared = T.alloc_shared(Bias_shared_shape, out_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        topk_weights_shared = T.alloc_shared((block_M), out_dtype)
        sorted_token_ids_shared = T.alloc_shared((block_M), T.int32)
        expert_id = T.alloc_local((1), T.int32)  # the expert id for the current block
        # To use 1D TMA, the last dim of Scale_shared must have stride=1
        # May use much more shared memory than necessary
        Scale_shared = T.alloc_shared((block_N, K // scale_size), storage_dtype)

        T.annotate_layout(
            {
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            }
        )
        T.use_swizzle(10)

        if threads == 512:
            T.disable_warp_group_reg_alloc()

        T.copy(sorted_token_ids[by * block_M : (by + 1) * block_M], sorted_token_ids_shared)
        expert_id[0] = expert_ids[by]

        # Get the topk weights of each token in the current block
        for i in T.Parallel(block_M):
            if sorted_token_ids_shared[i] != -1:
                topk_weights_shared[i] = topk_weights[sorted_token_ids_shared[i]]

        # Get bias and scale based on the expert id
        if with_bias:
            T.copy(Bias[expert_id[0], bx * block_N : (bx + 1) * block_N], Bias_shared)
        else:
            T.clear(Bias_shared)

        T.copy(Scale[expert_id[0], bx * block_N : (bx + 1) * block_N, :], Scale_shared)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = Bias_shared[j]

        tx = T.get_thread_binding()

        for k in T.Pipelined(K // block_K, num_stages=num_stages):
            # Each thread copies 4 bytes, local size is 16
            for copy_i in T.serial(block_M * block_K // threads // 16):
                base = copy_i * threads * 16 + tx * 16
                if sorted_token_ids_shared[base // block_K] != -1:
                    for copy_j in T.vectorized(16):
                        A_shared[base // block_K, base % block_K + copy_j] = A[
                            sorted_token_ids_shared[base // block_K] // topk, k * block_K + base % block_K + copy_j
                        ]

            T.copy(B[expert_id[0], bx * block_N, k * block_K // num_elems_per_byte], B_shared)
            if fast_dequant:
                get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale_shared, k)
            else:
                get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale_shared, k)

            T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = C_local[i, j] * topk_weights_shared[i]

        T.copy(C_local, C_shared)
        for copy_i in T.serial(block_M * block_N // threads // 16):
            base = copy_i * threads * 16 + tx * 16
            if sorted_token_ids_shared[base // block_N] != -1:
                for copy_j in T.vectorized(16):
                    C[
                        sorted_token_ids_shared[base // block_N] // topk,
                        sorted_token_ids_shared[base // block_N] % topk,
                        bx * block_N + base % block_N + copy_j,
                    ] = C_shared[base // block_N, base % block_N + copy_j]

    return C


def ref_moe(A, qB, Scale, Bias, topk_weights, sorted_token_ids, expert_ids, block_M=256):
    dtypeC = T.bfloat16
    M, K = A.shape
    E, N, QK = qB.shape
    topk = topk_weights.shape[0] // M
    scale_size = K // Scale.shape[2]
    assert scale_size == 32  # MXFP4

    # Initialize output tensor
    C = torch.ones((M, topk, N), dtype=getattr(torch, dtypeC), device="cuda")

    # Iterate over sorted_token_ids
    for idx in range(len(sorted_token_ids)):  # padding_M
        token_id = sorted_token_ids[idx]
        if token_id == -1:
            continue
        expert_id = expert_ids[idx // block_M]
        topk_idx = token_id % topk

        # Get the token embedding
        token_embedding = A[token_id // topk]

        # Dequantize the expert weights
        B = torch_convert_bit_twiddling(qB[expert_id])  # shape: (N, K)
        B *= 2 ** (Scale[expert_id][:, (torch.arange(B.shape[1], device=B.device) // scale_size)].to(torch.bfloat16))

        # Compute the output for this token-expert pair
        # token_embedding @ B.T + bias
        output = torch.matmul(token_embedding.to(torch.bfloat16), B.T.to(torch.bfloat16)) + Bias[expert_id]
        output = output.to(torch.__getattribute__(dtypeC))

        # Apply the topk weight
        weight = topk_weights[token_id]
        output = output * weight

        # Store the result
        C[token_id // topk, topk_idx] = output

    return C


def get_data(m, n, k, qk, scale_size, topk, E, block_M):
    A = torch.empty(m, k, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    qB = torch.randint(0, 256, (E, n, qk), dtype=torch.uint8, device="cuda")  #  Quantized weight tensor for E experts.
    Scale = torch.randint(0, 8, (E, n, k // scale_size), dtype=torch.uint8, device="cuda")
    Bias = torch.empty(E, n, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)

    weights = torch.empty(m, E, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    # topk_weights: Router weights for the top-k experts for each token.
    # Shape: (m, topk)
    # tokens_experts: A flattened tensor of expert assignments for each token.
    # For each of m tokens, topk unique experts are chosen. Shape: (m * topk,)
    topk_weights, tokens_experts = torch.topk(weights, topk, dim=-1)
    tokens_experts = tokens_experts.reshape(m * topk)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.reshape(m * topk)

    sorted_expert_vals, sorted_indices = torch.sort(tokens_experts, stable=True)
    sorted_token_ids = sorted_indices
    unique_expert_ids, counts = torch.unique_consecutive(sorted_expert_vals, return_counts=True)
    expert_ids = []
    padded_token_ids = []
    start = 0
    for eid, cnt in zip(unique_expert_ids.tolist(), counts.tolist()):
        end = start + cnt
        group_token_ids = sorted_token_ids[start:end]
        pad_len = ((cnt + block_M - 1) // block_M) * block_M - cnt
        if pad_len > 0:
            # -1 for padding (`M` instead in vLLM moe_align_block_size())
            group_token_ids = torch.cat([group_token_ids, torch.full((pad_len,), -1, dtype=group_token_ids.dtype, device="cuda")])
        padded_token_ids.append(group_token_ids)
        expert_ids.extend([eid] * ((cnt + block_M - 1) // block_M))
        start = end

    # sorted_token_ids: The final flattened and padded tensor of token indices.
    sorted_token_ids = torch.cat(padded_token_ids, dim=0).to(torch.int32)  # (padding_M,)
    # expert_ids: The final tensor of expert IDs corresponding to `sorted_token_ids`.
    expert_ids = torch.tensor(expert_ids, dtype=torch.int32, device="cuda")  # (padding_M,)
    padding_M = sorted_token_ids.shape[0]  # padding_M: token number after padding

    return A, qB, Scale, Bias, topk_weights, sorted_token_ids, expert_ids, padding_M


def main(m=256, n=256, k=256, scale_size=32, topk=4, E=32, fast_dequant=True, with_bias=False, tune=False):
    # Tunable parameters
    block_M, block_N, block_K = 128, 256, 128  # noqa: F841
    num_stages = 1  # noqa: F841
    threads = 512  # noqa: F841
    split = 1  # noqa: F841

    total_flops = 2 * m * n * k * topk
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    qk = k // num_elems_per_byte
    A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t, padding_M = get_data(m, n, k, qk, scale_size, topk, E, block_M)

    if tune:
        with set_autotune_inputs([A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t]):
            # Autotune with inputs manually composed
            kernel = matmul(
                A,
                qB,
                Scale,
                Bias,
                topk_weights_t,
                sorted_token_ids,
                expert_ids_t,
                topk=topk,
                in_dtype=T.bfloat16,
                out_dtype=T.bfloat16,
                accum_dtype=T.float32,
                num_bits=num_bits,
                scale_size=scale_size,
                fast_dequant=fast_dequant,
                with_bias=with_bias,
            )
    else:
        kernel = matmul(
            A,
            qB,
            Scale,
            Bias,
            topk_weights_t,
            sorted_token_ids,
            expert_ids_t,
            topk=topk,
            in_dtype=T.bfloat16,
            out_dtype=T.bfloat16,
            accum_dtype=T.float32,
            num_bits=num_bits,
            scale_size=scale_size,
            fast_dequant=fast_dequant,
            with_bias=with_bias,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            split=split,
        )

    output = kernel(
        A,
        qB,
        Scale,
        Bias,
        topk_weights_t,
        sorted_token_ids,
        expert_ids_t,
    )
    print("Tilelang kernel run finished.")

    ref_output = ref_moe(A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t, block_M=block_M)  # Maybe a little bit slow...

    latency = tilelang.profiler.do_bench(lambda: kernel(A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t), warmup=100)
    print("Tilelang: {:.2f} ms".format(latency))
    print("Tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    diff = (output - ref_output).abs()
    max_val = diff.max()
    max_idx = diff.argmax()
    print(f"max abs diff: {max_val} at index: {max_idx}")
    assert_similar(output, ref_output, name="output", eps=2e-5)  # We care about the similarity rather than abs. difference
    print("All checks pass. ✅")


def run_regression_perf(m=4096, n=4096, k=4096, scale_size=32, topk=4, E=32, fast_dequant=True, with_bias=False, tune=False):
    block_M, block_N, block_K = 128, 256, 128
    num_stages = 1
    threads = 512
    split = 1
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    qk = k // num_elems_per_byte
    A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t, padding_M = get_data(m, n, k, qk, scale_size, topk, E, block_M)

    if tune:
        with set_autotune_inputs([A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t]):
            kernel = matmul(
                A,
                qB,
                Scale,
                Bias,
                topk_weights_t,
                sorted_token_ids,
                expert_ids_t,
                topk=topk,
                in_dtype="bfloat16",
                out_dtype="bfloat16",
                accum_dtype="float32",
                num_bits=num_bits,
                scale_size=scale_size,
                fast_dequant=fast_dequant,
                with_bias=with_bias,
            )
    else:
        kernel = matmul(
            A,
            qB,
            Scale,
            Bias,
            topk_weights_t,
            sorted_token_ids,
            expert_ids_t,
            topk=topk,
            in_dtype="bfloat16",
            out_dtype="bfloat16",
            accum_dtype="float32",
            num_bits=num_bits,
            scale_size=scale_size,
            fast_dequant=fast_dequant,
            with_bias=with_bias,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            split=split,
        )

    return tilelang.profiler.do_bench(lambda: kernel(A, qB, Scale, Bias, topk_weights_t, sorted_token_ids, expert_ids_t), backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256, help="M")  # From gpt-oss-20b MoE's first gemm
    parser.add_argument("--N", type=int, default=256, help="N")
    parser.add_argument("--K", type=int, default=256, help="K")
    parser.add_argument("--scale_size", type=int, default=32, help="scale size")
    parser.add_argument("--topk", type=int, default=4, help="topk")  # experts activated for each token
    parser.add_argument("--E", type=int, default=32, help="E")  # number of experts
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.scale_size, topk=args.topk, E=args.E, fast_dequant=True, with_bias=True, tune=args.tune)
