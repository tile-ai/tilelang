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
    out_idx=[-1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    source_format=T.uint32,
    num_bits=4,
    fast_dequant=True,
    block_M=256,
    block_N=128,
    block_K=128,
    num_stages=2,
    threads=256,
    split=1,
):
    """
    Builds a parameterized TileLang/TIR matrix-multiplication kernel that dequantizes 4-bit FP inputs to BF16 on-the-fly and computes C = A @ B^T.

    This function returns a tiled, autotunable prim_func implementing a block-wise GEMM with shared-memory buffering and a pipelined K-loop. The kernel accepts:
    - A: dense input of shape (M, K) with dtype `in_dtype`.
    - B: packed quantized input of shape (N, QK) where QK = K / (8 / num_bits) stored as `uint8`.
    - C: output of shape (M, N) with dtype `out_dtype`.

    The generated kernel supports two dequantization paths:
    - fast_dequant (fast_dequant=True): calls an external mxfp dequantization intrinsic (twiddling-based) loaded from a C source returned by get_mxfp_intrin_group.
    - simple dequant (fast_dequant=False): performs a pure-TIR FP4 -> BF16 conversion per element.

    Important behavior and requirements:
    - num_bits (default 4) is the bit-width of the quantized elements; storage_dtype is uint8 and num_elems_per_byte = 8 // num_bits.
    - QK = K // num_elems_per_byte and Block_QK = block_K // num_elems_per_byte determine B and shared-buffer shapes.
    - Asserts that K % (block_K * split) == 0; K must be divisible by block_K * split for the tiling to be valid.
    - When fast_dequant is True, a valid mxfp intrinsic group (C source and function name) must be available via tilelang.quantize.get_mxfp_intrin_group.
    - The kernel launches a 2D grid over ceildiv(N, block_N) and ceildiv(M, block_M) and uses `threads` threads per block with `num_stages` pipeline stages.

    Parameters that alter kernel layout/behavior (brief):
    - block_M, block_N, block_K: tile sizes for M, N, and K dimensions.
    - num_stages: number of software pipeline stages for the K-loop.
    - threads: number of threads used per kernel block.
    - split: extra K-splitting factor; K must be divisible by block_K * split.
    - source_format, num_bits: describe the quantized data layout passed to the mxfp intrinsics.

    Returns:
        A TileLang/TIR prim_func (the compiled `main`) implementing the described dequantize-then-GEMM kernel.
    """
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8

    QK = K // num_elems_per_byte
    Block_QK = block_K // num_elems_per_byte
    A_shape = (M, K)
    B_shape = (N, QK)
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

        This function validates the requested input/output datatypes and returns a TileLang `@T.macro` named `fast_dequant_bf16_fp4_twiddling` which:
        - Loads compressed FP4 bytes from a shared buffer into per-thread local registers (vectorized loads).
        - Invokes an external dequantization routine (via `T.call_extern`) to expand the packed FP4 values into BF16 in registers.
        - Writes the dequantized BF16 values back to a shared dequantized buffer for use by the kernel.

        Notes and preconditions:
        - Asserts that `in_dtype == "fp4"` and `out_dtype == T.bfloat16`.
        - The generated macro depends on several surrounding-scope symbols (e.g., `import_source`, `func_name`, `block_K`, `Block_QK`, `threads`, `num_elems_per_byte`, `storage_dtype`, and `out_dtype`) and expects them to be defined consistently in the enclosing kernel.
        - The macro is optimized for block-wise, per-thread transactions sized to the target storage width (uses a MAX_TRANSACTION_SIZE_BITS constant) and uses local/register buffers sized accordingly.
        - The macro uses `T.import_source` to bring the external plugin into the module and `T.call_extern` to perform the high-throughput dequantization; callers must ensure the external function matches the expected calling convention and memory layout.
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
            """
            Fast dequantization kernel routine that converts packed FP4 values in shared memory to BF16 and writes the results back into a shared dequantized buffer.

            This function is intended to run inside a tiled GPU kernel: each thread loads a small packed segment from the quantized shared buffer `B_shared` into a per-thread local register buffer, calls an external dequantization routine (provided by the runtime plugin imported from `import_source` and identified by `func_name`) to expand the packed values to BF16 in a per-thread local output buffer, and stores the expanded values into `B_dequantize_shared`. It performs vectorized per-thread loads and stores and is sized according to the surrounding kernel's tiling and threading parameters.

            Parameters:
                B_shared: Shared-memory buffer containing packed quantized values (packed FP4 layout).
                B_dequantize_shared: Shared-memory buffer to receive dequantized BF16 values (written in-place by this routine).

            Side effects:
                - Imports the external dequantization plugin via `import_source` and invokes `func_name`.
                - Writes dequantized BF16 results into `B_dequantize_shared`.

            Notes:
                - This routine expects the surrounding kernel to define and provide the tiling/threading constants (e.g., thread count, local buffer sizes, block dimensions) and the runtime plugin identifiers (`import_source`, `func_name`).
                - No value is returned; results are produced by mutation of `B_dequantize_shared`.
            """
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
                    T.address_of(B_local_thread[0]),
                    T.address_of(B_dequantize_local_thread[0]),
                    1,
                    dtype=out_dtype,
                )

                # Finally, store the dequantized data to shared memory.
                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequantize_shared[index // block_K, index % block_K] = B_dequantize_local_thread[v]

        return fast_dequant_bf16_fp4_twiddling

    def get_simple_dequant_func(in_dtype="fp4", out_dtype=T.bfloat16):
        """
        Create a simple TIR dequantization macro that converts packed 4-bit FP (FP4) stored in uint8 into bfloat16.

        The returned macro (named `simple_dequant_bf16_fp4`) expects B_shared and B_dequantize_shared buffers (shapes and a few loop/constant names like
        `B_shared_shape`, `B_dequantize_shared_shape`, `storage_dtype`, `out_dtype`, `num_bits`, `num_elems_per_byte`, `block_N`, and `block_K`) to be available in the surrounding TIR scope. It:
        - Unpacks 4-bit FP values from the packed uint8 representation in B_shared.
        - Converts each 4-bit value to a bfloat16 element using an internal helper `_tir_u8_to_f4_to_bf16`.
        - Writes the dequantized bfloat16 block into B_dequantize_shared.

        Constraints:
        - Supports only in_dtype="fp4" and out_dtype=T.bfloat16.
        - The helper assumes nbit == 4 and produces bfloat16 values.
        - The macro uses a fixed test-scale of 0 (no per-element scaling) as written.

        Returns:
            A TIR macro function performing the described in-place block dequantization from packed uint8 FP4 to bfloat16.
        """
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr, dtype: str):
            """
            Convert a 4-bit FP4 value packed in a uint8 byte into a bfloat16 value.

            This helper extracts the 4-bit field located at the bit position `pos` within the
            byte `val`, interprets it as an FP4 (sign, exponent, mantissa) value, applies an
            exponent `scale` offset to align it with bfloat16 exponent bias, clamps the
            resulting exponent to 8 bits, and returns the assembled bfloat16 bit pattern.

            Parameters:
                nbit (int): Number of bits in the packed element; must be 4.
                val (tir.PrimExpr): A uint8 value containing packed FP4 elements.
                pos (tir.PrimExpr): Index (0-based) of which FP4 nibble inside `val` to extract.
                scale (tir.PrimExpr): Exponent offset applied when converting FP4 exponent to bfloat16.
                dtype (str): Target dtype string; must be T.bfloat16.

            Returns:
                tir.PrimExpr: A bfloat16-typed PrimExpr containing the converted value.

            Notes:
                - The function asserts `nbit == 4`, `dtype == T.bfloat16`, and that `val.dtype` is T.uint8.
                - The conversion uses a fixed mapping from FP4 exponent/mantissa layout into bfloat16
                bit fields and clamps the computed exponent to fit into 8 bits.
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
            # To handle the overflow, we use the max function to limit the exponential part to 8 bits
            e_bf16 = T.min(e_bf16 + scale, tir.const((1 << 8) - 1, T.uint16))
            m_f4 = f4 & tir.const(1, T.uint16)
            val_bf16 = tir.reinterpret(
                T.bfloat16,
                ((((s << tir.const(8, T.uint16)) | e_bf16) << tir.const(7, T.uint16)) | (m_f4 << tir.const(6, T.uint16))).astype(T.uint16),
            )
            return val_bf16

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared):
            """
            Dequantize a packed FP4 uint8 shared buffer into BF16 and store the result into a shared dequantized buffer.

            This helper:
            - Loads B_shared into a local fragment, converts each packed FP4 element to BF16 using `_tir_u8_to_f4_to_bf16`, and writes the dequantized values into B_dequantize_shared.
            - Iterates in parallel over the logical block columns (block_N) and block_K, unpacking elements from bytes using `num_elems_per_byte`.
            - Uses a fixed scale of 0 in the conversion (placeholder for testing); `num_bits` and `num_elems_per_byte` are expected to be available from the enclosing scope.

            Parameters:
                B_shared: shared-memory buffer containing packed FP4 data (uint8-packed).
                B_dequantize_shared: shared-memory buffer to receive BF16 dequantized values.

            Side effects:
                Writes dequantized BF16 values into B_dequantize_shared. No return value.
            """
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

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        """
        Kernel entry for the tiled, pipelined matmul used by the generated prim_func.

        This function implements a block-wise GEMM over a 2D grid (grid dims: ceildiv(N, block_N) x ceildiv(M, block_M)) with a thread block of `threads`. For each output block it:
        - Allocates shared buffers for A, the packed/quantized B, and a dequantized B tile.
        - Allocates a fragment accumulator (C_local) and a shared output tile (C_shared) with a swizzled layout.
        - Pipelines over K in chunks of `block_K` for `num_stages` stages:
          - Loads A and packed B tiles into shared memory.
          - Dequantizes B into B_dequantize_shared using either the fast (twiddling/external) or the simple (pure-TIR) dequantization routine.
          - Performs a GEMM accumulating into C_local with B transposed.
        - Stores the accumulated block from C_local back to the global output C via C_shared.

        Parameters:
        - A: input tile of shape (M, K) with dtype `in_dtype`.
        - B: packed/quantized input of shape (N, QK) with storage dtype `storage_dtype` (quantized FP4 packing).
        - C: output tensor of shape (M, N) with dtype `out_dtype`.

        Side effects:
        - Writes the computed output block into the global tensor `C`.
        - Uses and updates shared memory buffers and per-thread accumulators.

        No value is returned.
        """
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

    return main


def ref_program_twiddling(A, qB):
    """
    Compute reference BF16 matrix multiply using bit-twiddled FP4 quantized B.

    Converts qB (a bit-twiddled, packed FP4 representation of matrix B) back to floating,
    performs C = A @ B^T in full precision, and returns the result converted to bfloat16.

    Parameters:
        A (torch.Tensor): Left operand with shape (M, K). Treated as floating-point (converted to torch.float for compute).
        qB (torch.Tensor): Bit-twiddled, packed FP4 representation of B (quantized). Shape corresponds to B's packed layout.

    Returns:
        torch.Tensor: Result matrix C with shape (M, N) in bfloat16.
    """
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple(A, qB):
    """
    Compute a reference BF16 matrix multiply using a simple (non-twiddled) dequantization of qB.

    Converts the quantized tensor `qB` to full-precision values via `torch_convert`, computes C = A @ B^T in float32, and casts the result to bfloat16 before returning.

    Parameters:
        A (torch.Tensor): Left input matrix with shape (M, K).
        qB (torch.Tensor): Quantized representation of the right matrix; expected to be compatible with `torch_convert` and represent a matrix whose transpose will be multiplied by A.

    Returns:
        torch.Tensor: Resulting matrix C in bfloat16 with shape (M, N).
    """
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, fast_dequant=True, tune=False):
    """
    Run and benchmark the tiled, optionally autotuned FP4->BF16 GEMM kernel and validate results against a PyTorch reference.

    This function builds a matmul kernel (either with autotuning or fixed tiling), obtains a profiler, validates numerical correctness against the appropriate reference implementation (bit-twiddled fast dequantization or simple dequantization), and runs a benchmark that prints measured latency (ms) and effective TFLOPs.

    Parameters:
        m (int): Number of rows of A and output C (default 256).
        n (int): Number of columns of B and output C (default 256).
        k (int): Inner dimension (columns of A, rows of B) (default 256).
        fast_dequant (bool): If True use the fast twiddling dequantization path and validate against the twiddling reference; otherwise use the simple dequant path (default True).
        tune (bool): If True build the kernel with autotuning configurations; if False use a fixed tiling and threading configuration for reproducible benchmarking (default False).

    Side effects:
        - Prints latency and TFLOPs to stdout.
        - Raises an assertion via the profiler if the kernel's outputs do not match the chosen reference within the tolerances (rtol=0.01, atol=0.01).
    """
    total_flops = 2 * m * n * k
    if tune:
        kernel = matmul(m, n, k, T.bfloat16, T.bfloat16, T.float32, num_bits=4, fast_dequant=fast_dequant)
    else:
        kernel = matmul(
            m,
            n,
            k,
            T.bfloat16,
            T.bfloat16,
            T.float32,
            num_bits=4,
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


def run_regression_perf(m=256, n=256, k=256, fast_dequant=True):
    kernel = matmul(
        m,
        n,
        k,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=4,
        fast_dequant=fast_dequant,
        block_M=256,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
        split=1,
    )
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(warmup=500, backend="cupti")


if __name__ == "__main__":
    main(256, 256, 256, True)
    main(256, 256, 256, False)
