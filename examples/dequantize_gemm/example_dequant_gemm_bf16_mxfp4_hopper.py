import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tvm import tir
import torch
from dequantize_utils import torch_convert_bit_twiddling, torch_convert


def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr, dtype: str):
    """
    Convert a 4-bit field packed in a uint8 into a bfloat16 value, applying an exponent scale.
    """
    assert nbit == 4
    assert dtype == T.bfloat16
    assert val.dtype == T.uint8
    mask = tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)
    e_f4 = (f4 & tir.const(6, T.uint16)) >> tir.const(1, T.uint16)
    # Exponential bias between f4 and bf16 is 2^(8-1) - 2^(2-1) = 126
    e_bf16 = e_f4 + tir.const(126, T.uint16)
    # Scale is the exponential part, within the representation of uint8
    # To handle the overflow, we may use the min function to limit the exponential part to 8 bits
    # e_bf16 = T.min(e_bf16 + scale, tir.const((1 << 8) - 1, "uint16"))
    m_f4 = f4 & tir.const(1, T.uint16)
    val_bf16 = tir.reinterpret(
        T.bfloat16,
        ((((s << tir.const(8, T.uint16)) | e_bf16) << tir.const(7, T.uint16)) | (m_f4 << tir.const(6, T.uint16))).astype(T.uint16),
    )
    return val_bf16


def get_configs():
    """
    Generate a list of hyperparameter configuration dictionaries for tuning.
    """
    import itertools

    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[64, 128, 256],
        num_stages=[0, 2],
        threads=[128, 256, 512],
        split=[1, 2],
    )
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit
def matmul(
    A,
    B,
    Scale,
    Bias,
    in_dtype: T.dtype = T.bfloat16,
    out_dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
    source_format: T.dtype = T.uint32,
    num_bits: int = 4,
    scale_size: int = 32,
    fast_dequant: bool = True,
    with_bias: bool = False,
    block_M: int = 256,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    split: int = 1,
):
    """
    Construct and return a tiled matrix-multiply TIR kernel that multiplies A (shape MxK) by a quantized B (shape Nx(QK)) and writes an MxN output in out_dtype.
    """
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    M, K = T.const("M K")
    N, _ = T.const("N _")
    _, _SK = T.const("_ _SK")
    _BM, _BN = T.const("_BM _BN")
    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, _], storage_dtype]
    Scale: T.Tensor[[N, _SK], storage_dtype]
    Bias: T.Tensor[[_BM, _BN], out_dtype]

    Block_QK = block_K // num_elems_per_byte
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, Block_QK)
    Bias_shared_shape = (block_M, block_N)
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

    def get_fast_dequant_twiddling_func(in_dtype="fp4", out_dtype=T.bfloat16):
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale, k):
            # import fast_dequantize plugin
            T.import_source(import_source)

            tx = T.get_thread_binding()
            bx = T.get_block_binding(0)

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
                Scale_local_thread[0] = Scale[bx * block_N + si, k * block_K // scale_size + sj]
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
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared, Scale, k):
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, out_dtype)

            bx = T.get_block_binding(0)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                    num_bits,
                    B_local[i, j // num_elems_per_byte],
                    j % num_elems_per_byte,
                    Scale[
                        bx * block_N + i, k * block_K // scale_size + j // scale_size
                    ],  # Scale is the exponential part, within the representation of uint8
                    dtype=out_dtype,
                ) * T.shift_left(1, (Scale[bx * block_N + i, k * block_K // scale_size + j // scale_size]))
            T.copy(B_dequantize_local, B_dequantize_shared)

        return simple_dequant_bf16_fp4

    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
        B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
        Bias_shared = T.alloc_shared(Bias_shared_shape, out_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.annotate_layout(
            {
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            }
        )

        if with_bias:
            T.annotate_layout(
                {
                    Bias_shared: tilelang.layout.make_swizzled_layout(Bias_shared),
                }
            )

        if threads == 512:
            T.disable_warp_group_reg_alloc()

        if with_bias:
            T.copy(Bias[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N], Bias_shared)
            T.copy(Bias_shared, C_local)
        else:
            T.clear(C_local)

        for k in T.Pipelined(K // block_K, num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
            if fast_dequant:
                get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale, k)
            else:
                get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale, k)
            T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return C


def ref_program_twiddling(A, qB, Scale, Bias=None):
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_twiddling_with_bias(A, qB, Scale, Bias):
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float)) + Bias
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple(A, qB, Scale, Bias=None):
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple_with_bias(A, qB, Scale, Bias):
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float)) + Bias
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, scale_size=32, fast_dequant=True, with_bias=False, tune=False):
    total_flops = 2 * m * n * k
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    A = torch.empty(m, k, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    qB = torch.randint(0, 256, (n, k // num_elems_per_byte), dtype=getattr(torch, storage_dtype), device="cuda")
    Scale_tensor = torch.randint(0, 8, (n, k // scale_size), dtype=getattr(torch, storage_dtype), device="cuda")
    Bias_tensor = torch.empty(m, n, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)

    if tune:
        kernel = matmul(
            A,
            qB,
            Scale_tensor,
            Bias_tensor,
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
            Scale_tensor,
            Bias_tensor,
            in_dtype=T.bfloat16,
            out_dtype=T.bfloat16,
            accum_dtype=T.float32,
            num_bits=num_bits,
            scale_size=scale_size,
            block_M=256,
            block_N=128,
            block_K=128,
            num_stages=2,
            threads=256,
            split=1,
            fast_dequant=fast_dequant,
            with_bias=with_bias,
        )

    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)

    if fast_dequant:
        if with_bias:
            profiler.assert_allclose(ref_program_twiddling_with_bias, rtol=0.01, atol=0.01)
        else:
            profiler.assert_allclose(ref_program_twiddling, rtol=0.01, atol=0.01)
    else:
        if with_bias:
            profiler.assert_allclose(ref_program_simple_with_bias, rtol=0.01, atol=0.01)
        else:
            profiler.assert_allclose(ref_program_simple, rtol=0.01, atol=0.01)
    print("All checks pass.")
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def run_regression_perf(m=4096, n=4096, k=4096, scale_size=32, fast_dequant=True, with_bias=False):
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    A = torch.empty(m, k, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    qB = torch.randint(0, 256, (n, k // num_elems_per_byte), dtype=getattr(torch, storage_dtype), device="cuda")
    Scale_tensor = torch.randint(0, 8, (n, k // scale_size), dtype=getattr(torch, storage_dtype), device="cuda")
    Bias_tensor = torch.empty(m, n, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)

    kernel = matmul(
        A,
        qB,
        Scale_tensor,
        Bias_tensor,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
        accum_dtype="float32",
        num_bits=num_bits,
        scale_size=scale_size,
        block_M=256,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
        split=1,
        fast_dequant=fast_dequant,
        with_bias=with_bias,
    )
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    scale_size = 32
    main(M, N, K, scale_size, fast_dequant=True, with_bias=True)
    main(M, N, K, scale_size, fast_dequant=False, with_bias=True)
    main(M, N, K, scale_size, fast_dequant=True, with_bias=False)
    main(M, N, K, scale_size, fast_dequant=False, with_bias=False)
