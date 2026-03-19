import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tvm import tir
import torch
from dequantize_utils import torch_convert_bit_twiddling, torch_convert


def get_configs():
    """
    Return a list of tuning configuration dictionaries for the autotuned matmul kernel.

    Each dictionary is a single combination (Cartesian product) of the following parameters:
    - block_M: tile size for M dimension (one of 64, 128, 256)
    - block_N: tile size for N dimension (one of 64, 128, 256)
    - block_K: tile size for K dimension
    - num_stages: pipeline stages for K-loop (0 or 2)
    - threads: number of threads to launch (128, 256, or 512)
    - split: K-splitting factor (1 or 2)

    Returns:
        list[dict]: List of configuration dicts usable by the autotuner, where each dict maps
        the parameter name to its chosen value.
    """
    import itertools

    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[128],
        num_stages=[0, 2],
        threads=[128, 256, 512],
        split=[1, 2],
    )
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def matmul(
    A,
    B,
    in_dtype: T.dtype = T.bfloat16,
    out_dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
    source_format: T.dtype = T.uint32,
    num_bits: int = 4,
    fast_dequant: bool = True,
    block_M: int = 256,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    split: int = 1,
):
    """
    Builds a parameterized TileLang/TIR matrix-multiplication kernel that dequantizes 4-bit FP inputs to BF16 on-the-fly and computes C = A @ B^T.
    """
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    M, K = T.const("M K")
    N, _ = T.const("N _")
    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, _], storage_dtype]

    Block_QK = block_K // num_elems_per_byte
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, Block_QK)
    B_dequantize_shared_shape = (block_N, block_K)
    assert K % (block_K * split) == 0

    from tilelang.quantize import get_mxfp_intrin_group

    # fast_dequant_bf16_fp4_twiddling
    # It requires that the 2 consecutive uint8 elements (16bits) contains 4 fp4 elements in a bit-twiddling way.
    # The bit-twiddling way is shown here: The pair (x,y) shows that the bit in this position is the y-th bit of the x-th fp4 element.
    # (0,0)(3,0)(3,3)(1,0)(3,1)(3,2)(2,0)(0,1)(0,2)(0,3)(1,1)(1,2)(1,3)(2,1)(2,2)(2,3)
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

    def get_fast_dequant_twiddling_func(in_dtype="fp4", out_dtype=T.bfloat16):
        """
        Create a TileLang macro that performs fast, twiddling-based dequantization from packed FP4 to BF16 using an external runtime plugin.
        """
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared):
            # import fast_dequantize plugin
            T.import_source(import_source)

            tx = T.get_thread_binding()

            B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
            B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
            for i in T.serial(0, block_N * block_K // threads // local_size):
                # First, load data from share memory to register.
                # Prepare for dequant.
                for v in T.vectorized(0, local_compress_size):
                    index = i * threads * local_compress_size + tx * local_compress_size + v
                    B_local_thread[v] = B_shared[index // Block_QK, index % Block_QK]

                # Then, dequant.
                T.call_extern(
                    func_name,
                    T.access_ptr(B_local_thread, "r"),
                    T.access_ptr(B_dequantize_local_thread, "w"),
                    1,
                    dtype=out_dtype,
                )

                # Finally, store the dequantized data to shared memory.
                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequantize_shared[index // block_K, index % block_K] = B_dequantize_local_thread[v]

        return fast_dequant_bf16_fp4_twiddling

    def get_simple_dequant_func(in_dtype="fp4", out_dtype=T.bfloat16):
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr, dtype: str):
            assert nbit == 4
            assert dtype == T.bfloat16
            assert val.dtype == T.uint8
            mask = tir.const((1 << nbit) - 1, T.uint16)
            f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
            s = f4 >> tir.const(3, T.uint16)
            e_f4 = (f4 & tir.const(6, T.uint16)) >> tir.const(1, T.uint16)
            e_bf16 = e_f4 + tir.const(126, T.uint16)
            e_bf16 = T.min(e_bf16 + scale, tir.const((1 << 8) - 1, T.uint16))
            m_f4 = f4 & tir.const(1, T.uint16)
            val_bf16 = tir.reinterpret(
                T.bfloat16,
                ((((s << tir.const(8, T.uint16)) | e_bf16) << tir.const(7, T.uint16)) | (m_f4 << tir.const(6, T.uint16))).astype(T.uint16),
            )
            return val_bf16

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared):
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, out_dtype)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                    num_bits,
                    B_shared[i, j // num_elems_per_byte],
                    j % num_elems_per_byte,
                    0,  # No scale for test
                    dtype=out_dtype,
                )
            T.copy(B_dequantize_local, B_dequantize_shared)

        return simple_dequant_bf16_fp4

    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
        B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)

        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.clear(C_local)
        for k in T.Pipelined(K // block_K, num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

            if fast_dequant:
                get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared)
            else:
                get_simple_dequant_func()(B_shared, B_dequantize_shared)

            T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return C


def ref_program_twiddling(A, qB):
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple(A, qB):
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, fast_dequant=True, tune=False):
    total_flops = 2 * m * n * k
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    A = torch.empty(m, k, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    qB = torch.randint(0, 256, (n, k // num_elems_per_byte), dtype=getattr(torch, storage_dtype), device="cuda")

    if tune:
        kernel = matmul(
            A, qB, in_dtype=T.bfloat16, out_dtype=T.bfloat16, accum_dtype=T.float32, num_bits=num_bits, fast_dequant=fast_dequant
        )
    else:
        kernel = matmul(
            A,
            qB,
            in_dtype=T.bfloat16,
            out_dtype=T.bfloat16,
            accum_dtype=T.float32,
            num_bits=num_bits,
            fast_dequant=fast_dequant,
            block_M=256,
            block_N=128,
            block_K=128,
            num_stages=2,
            threads=256,
            split=1,
        )
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
    if fast_dequant:
        profiler.assert_allclose(ref_program_twiddling, rtol=0.01, atol=0.01)
    else:
        profiler.assert_allclose(ref_program_simple, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def run_regression_perf(m=4096, n=4096, k=4096, fast_dequant=True):
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    A = torch.empty(m, k, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    qB = torch.randint(0, 256, (n, k // num_elems_per_byte), dtype=getattr(torch, storage_dtype), device="cuda")

    kernel = matmul(
        A,
        qB,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
        accum_dtype="float32",
        num_bits=num_bits,
        fast_dequant=fast_dequant,
        block_M=256,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
        split=1,
    )
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main(256, 256, 256, True)
    main(256, 256, 256, False)
