import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tilelang import register_cuda_postproc  # noqa: F401
from tvm import DataType
import torch

tilelang.disable_cache()

torch.manual_seed(0)

# @register_cuda_postproc
# def tilelang_callback_cuda_postproc(code, _):
#     cuda_code = ""
#     # with open("examples/dequantize_gemm/tilelang_jit_kernel_kernel_func_backup.c", "r") as f:
#     with open("examples/dequantize_gemm/tilelang_jit_kernel_kernel_func_test.c", "r") as f:
#         cuda_code = f.read()
#     return cuda_code


def torch_convert_bit_twiddling(tensor):

    def print_bit(name, val):
        val_cpu = val.cpu().item()
        binary_repr = f'{val_cpu:032b}'
        print(name, binary_repr)

    def _convert(val0, val1, pos) -> torch.bfloat16:
        assert val0.dtype == torch.uint8
        assert val1.dtype == torch.uint8
        val0 = val0.view(torch.uint8)
        val1 = val1.view(torch.uint8)
        val_concat = (val0.item() << 8) | val1.item()
        mask = 0b1000000111000000
        if pos == 0:
            bf16 = val_concat & mask
        elif pos == 1:
            bf16 = (val_concat << 3) & mask
        elif pos == 2:
            bf16 = (val_concat << 6) & mask
        elif pos == 3:
            mask1 = 0b1000000000000000
            mask2 = 0b0000000110000000
            mask3 = 0b0000000001000000
            bf16 = ((val_concat << 1) & mask1) | ((val_concat >> 3) & mask2) | (
                (val_concat >> 7) & mask3)
        bf16_new = torch.tensor([bf16], dtype=torch.uint16, device=val0.device).view(torch.bfloat16)
        # Add bias for change from fp4 to bf16
        bf16_new = bf16_new.item() * (2**126)
        return bf16_new

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 4 * 2], tensor[i][j // 4 * 2 + 1], j % 4)
    return new_tensor


def matmul(M,
           N,
           K,
           in_dtype,
           out_dtype,
           accum_dtype,
           source_format='uint',
           num_bits=4,
           scale_size=32,
           tune=False):

    @tilelang.jit(
        out_idx=[-1],
        # debug_root_path="/home/tzj/tilelang/examples/dequantize_gemm/",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
        },
    )
    def kernel_func(block_M, block_N, block_K, num_stages, threads, split=1):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = "uint8"
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        Scale_shape = (N, K // scale_size)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)
        Scale_shared_shape = (block_N, block_K // scale_size)
        assert K % (block_K * split) == 0

        # Some variables for serial dequant in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(in_dtype).bits
        local_compress_size = local_size // num_elems_per_byte
        # local_compress_size is the same as vectorize_dequant_size
        vectorize_dequant_size = 8

        from tilelang.quantize import get_mxfp_intrin_group
        mxfp_intrin_info = get_mxfp_intrin_group(
            out_dtype=in_dtype,
            source_format=source_format,
            source_bit=num_bits,
            storage_dtype=storage_dtype,
            with_scaling=False,
            with_zeros=False,
            use_twiddling=True,
        )
        import_source = mxfp_intrin_info["c_source"]
        func_name = mxfp_intrin_info["func_name"]
        assert import_source is not None, "mxfp_intrin_info is not found"
        assert func_name is not None, "mxfp_intrin_info is not found"
        import_source = import_source

        @T.prim_func
        def main(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                Scale: T.Tensor(Scale_shape, storage_dtype),
                C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
                B_dequantize_local_thread = T.alloc_local((local_size,), in_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                Scale_shared = T.alloc_shared(Scale_shared_shape, storage_dtype)
                Scale_local_thread = T.alloc_local((1,), storage_dtype)
                Scale_local_thread_exponent = T.alloc_local((1,), "float32")

                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)

                T.annotate_layout({
                    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    C_shared: tilelang.layout.make_swizzled_layout(C_shared),
                    # B_dequantize_shared: tilelang.layout.make_swizzled_layout(B_dequantize_shared),
                })

                T.import_source(import_source)

                tx = T.get_thread_binding()

                T.clear(C_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(Scale[bx * block_N, k * block_K // scale_size], Scale_shared)

                    for i in T.serial(0, block_N * block_K // threads // vectorize_dequant_size):
                        # First, load data from share memory to register.
                        # Prepare for dequant.
                        index_base = i * threads * (
                            vectorize_dequant_size // num_elems_per_byte) + tx * (
                                vectorize_dequant_size // num_elems_per_byte)
                        for v in T.vectorized(0, vectorize_dequant_size // num_elems_per_byte):
                            index = index_base + v
                            vi = index // (block_K // num_elems_per_byte)
                            vj = index % (block_K // num_elems_per_byte)
                            B_local_thread[v] = B_shared[vi, vj]
                        index_scale = index_base // (scale_size // num_elems_per_byte)
                        si = index_scale // (block_K // scale_size)
                        sj = index_scale % (block_K // scale_size)
                        Scale_local_thread[0] = Scale_shared[si, sj]
                        Scale_local_thread_exponent[0] = T.exp2(
                            T.cast(Scale_local_thread[0] - 127, "float"))

                        # Then, dequant.
                        T.call_extern(
                            func_name,
                            T.address_of(B_local_thread[0]),
                            T.address_of(B_dequantize_local_thread[0]),
                            1,
                            dtype=in_dtype,
                        )

                        # Finally, store the dequantized data to shared memory.
                        for v in T.Parallel(vectorize_dequant_size):
                            B_dequantize_local_thread[v] *= Scale_local_thread_exponent[0]
                        for v in T.vectorized(0, vectorize_dequant_size):
                            index = i * threads * vectorize_dequant_size + tx * vectorize_dequant_size + v
                            vi = index // block_K
                            vj = index % block_K
                            # 127 = 2^7 - 1 is the exponent bias for bfloat16
                            B_dequantize_shared[vi, vj] = B_dequantize_local_thread[v]

                    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M:(by + 1) * block_M,
                                   bx * block_N:(bx + 1) * block_N])

        return main

    def kernel(block_M, block_N, block_K, num_stages, threads, split=1):
        return kernel_func(block_M, block_N, block_K, num_stages, threads, split)

    return kernel


def ref_program(A, qB, Scale):
    dtypeC = "bfloat16"
    B = torch_convert_bit_twiddling(qB)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = B[i][j] * (2**(Scale[i][j // 32] - 127))
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, scale_size=32, tune=False):
    total_flops = 2 * m * n * k

    kernel = matmul(
        m, n, k, "bfloat16", "bfloat16", "float32", num_bits=4, scale_size=scale_size, tune=tune)(
            block_M=256, block_N=128, block_K=128, num_stages=2, threads=256, split=1)
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")
    # latency = profiler.do_bench(ref_program, warmup=500)
    # print("Ref: {:.2f} ms".format(latency))
    # print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    # M, N, K = 16384, 8192, 8192
    scale_size = 32
    main(M, N, K, scale_size)
