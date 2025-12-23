import tilelang
from tilelang import language as T
from typing import Optional, Callable, Any
import torch
from tilelang import DataType
from tilelang.quantize import (
    _tir_packed_int_to_int_convert,
)


@tilelang.jit
def dequantize_gemv(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_bits: int = 4,
    storage_dtype: T.dtype = T.int8,
    source_format: str = "uint",
    n_partition: int = 4,
    reduce_thread: int = 32,
    fast_decoding: bool = False,
    trans_A: bool = False,
    trans_B: bool = True,
    group_size: int = -1,
    with_scaling: bool = False,
) -> Callable[..., Any]:
    assert n_partition is not None, "n_partition must be provided"
    assert reduce_thread is not None, (
        "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMVsch_outer_reduction_with_config is not implemented"
    )

    assert trans_A is False, "Dequantize only implement for trans_A=False currently"
    assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
    storage_type = "".join(c for c in storage_dtype if not c.isdigit())
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    num_elems_per_byte = storage_nbit // num_bits

    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    block_K = reduce_thread * micro_size_k

    if group_size == -1:
        group_size = K

    A_shape = (M, K)
    B_shape = (N, K // storage_nbit * num_bits)
    C_shape = (M, N)

    dp4a_size = 4
    use_dp4a = in_dtype == T.int8 and accum_dtype == T.int32

    import_source: Optional[str] = None
    func_name: str = ""
    if fast_decoding is True:
        # Lazy import to decrease the startup time
        # as intrin registry may take a while to load
        from tilelang.quantize import get_lop3_intrin_group

        lop3_intrin_info = get_lop3_intrin_group(
            out_dtype=in_dtype,
            source_format=source_format,
            source_bit=num_bits,
            storage_dtype=storage_dtype,
            with_scaling=with_scaling,
            with_zeros=False,
        )
        import_source = lop3_intrin_info["c_source"]
        func_name = lop3_intrin_info["func_name"]
        assert import_source is not None, "lop3_intrin_info is not found"
        assert func_name is not None, "lop3_intrin_info is not found"
        import_source = import_source

    @T.prim_func
    def main(
        A: T.Tensor[A_shape, in_dtype],
        B: T.Tensor[B_shape, storage_dtype],
        C: T.Tensor[C_shape, out_dtype],
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            M,
            threads=(reduce_thread, n_partition),
        ) as (
            bx,
            by,
        ):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(import_source)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed) + kr * micro_size_k_compressed + v,
                    ]

                if fast_decoding:
                    T.call_extern(
                        func_name,
                        T.address_of(B_quant_local[0]),
                        T.address_of(B_dequantize_local[0]),
                        dtype=in_dtype,
                    )
                else:
                    for ki in T.serial(micro_size_k):
                        B_dequantize_local[ki] = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                            num_bits, B_quant_local[ki // num_elems_per_byte], ki % num_elems_per_byte, in_dtype
                        )

                if use_dp4a:
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_dequantize_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki] * B_dequantize_local[ki]

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]

    return main


def main() -> None:
    M = 1
    N = 1024
    K = 1024
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float16
    num_bits = 4
    storage_dtype = T.int8
    source_format = "uint"
    n_partition = 4
    reduce_thread = 32
    fast_decoding = True
    trans_A = False
    trans_B = True
    group_size = -1
    with_scaling = False

    kernel = dequantize_gemv(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_bits,
        storage_dtype,
        source_format,
        n_partition,
        reduce_thread,
        fast_decoding,
        trans_A,
        trans_B,
        group_size,
        with_scaling,
    )

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    num_elems_per_byte = storage_nbit // num_bits
    A = torch.rand(M, K, dtype=getattr(torch, in_dtype)).cuda()
    qB = torch.randint(0, 127, (N, K // num_elems_per_byte), dtype=getattr(torch, storage_dtype)).cuda()
    C = torch.zeros(M, N, dtype=getattr(torch, accum_dtype)).cuda()

    if fast_decoding:
        from tilelang.quantize.utils import interleave_weight

        qB = interleave_weight(qB, num_bits, in_dtype)
    kernel(A, qB, C)

    # int4 reference
    B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4, dtype=torch.half).to(torch.half).to(A.device)
    for j in range(B.shape[1]):
        B[:, j] = ((qB[:, j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    print("C: ", C)
    print("Ref C: ", ref_c)
    # doesn't apply scaling, the absolute error is large
    torch.testing.assert_close(C, ref_c, atol=1e3, rtol=1e-1)


def run_regression_perf():
    M = 1
    N = 1024
    K = 1024
    in_dtype = "float16"
    out_dtype = "float16"
    accum_dtype = "float16"
    num_bits = 4
    storage_dtype = "int8"
    source_format = "uint"
    n_partition = 4
    reduce_thread = 32
    fast_decoding = True
    trans_A = False
    trans_B = True
    group_size = -1
    with_scaling = False

    kernel = dequantize_gemv(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_bits,
        storage_dtype,
        source_format,
        n_partition,
        reduce_thread,
        fast_decoding,
        trans_A,
        trans_B,
        group_size,
        with_scaling,
    )

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    num_elems_per_byte = storage_nbit // num_bits
    A = torch.rand(M, K, dtype=getattr(torch, in_dtype)).cuda()
    qB = torch.randint(0, 127, (N, K // num_elems_per_byte), dtype=getattr(torch, storage_dtype)).cuda()
    C = torch.zeros(M, N, dtype=getattr(torch, accum_dtype)).cuda()

    if fast_decoding:
        from tilelang.quantize.utils import interleave_weight

        qB = interleave_weight(qB, num_bits, in_dtype)
    kernel(A, qB, C)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(A, qB, C)

    return do_bench(run_kernel_only, warmup=100, rep=1000, backend="cupti")


if __name__ == "__main__":
    main()
