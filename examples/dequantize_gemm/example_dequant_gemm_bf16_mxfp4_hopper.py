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

    This helper extracts a 4-bit nibble from `val` at byte-nibble position `pos`, interprets its
    bits as a sign/exponent/mantissa in the 4-bit custom FP4 layout, adjusts the exponent by
    `scale` (clamped to an 8-bit range), and assembles the corresponding bfloat16 representation.

    Parameters:
        nbit (int): Number of bits in the packed field (must be 4).
        val (tir.PrimExpr): Packed input value of dtype `uint8` containing one or more 4-bit fields.
        pos (tir.PrimExpr): Index of the nibble within `val` (used to shift/extract the 4-bit field).
        scale (tir.PrimExpr): Per-element exponent adjustment added to the extracted exponent (uint-like).
        dtype (str): Destination dtype string (must be T.bfloat16).

    Returns:
        tir.PrimExpr: The resulting value reinterpreted as `bfloat16`.

    Notes:
    - Preconditions are enforced via assertions: nbit == 4, dtype == T.bfloat16, and val.dtype == T.uint8.
    - The function clamps the adjusted exponent to the 8-bit range before assembling the bfloat16 bit pattern.
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

    Each configuration is a dict with keys: 'block_M', 'block_N', 'block_K',
    'num_stages', 'threads', and 'split'. The function returns the Cartesian
    product of the parameter value lists:
    - block_M, block_N, block_K: tiling sizes (64, 128, 256)
    - num_stages: pipeline stages (0, 2)
    - threads: thread counts (128, 256, 512)
    - split: K-splitting factor (1, 2)

    Returns:
        List[dict]: A list of configuration dictionaries covering all combinations.
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
@tilelang.jit(
    out_idx=[-1],
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
    scale_size=32,
    fast_dequant=True,
    with_bias=False,
    block_M=256,
    block_N=128,
    block_K=128,
    num_stages=2,
    threads=256,
    split=1,
):
    """
    Construct and return a tiled matrix-multiply TIR kernel that multiplies A (shape MxK) by a quantized B (shape Nx(QK)) and writes an MxN output in out_dtype.

    The generated kernel accepts:
    - A: dense matrix with element type `in_dtype`.
    - B: packed quantized matrix stored as uint8 with `num_bits` bits per element (QK = K / (8/num_bits)).
    - Scale: per-block scale/exponent information used to dequantize B.
    The kernel dequantizes B to a working floating format (out_dtype/accum_dtype) using one of two paths:
    - fast_dequant (True): uses an external, hardware/implementation-specific intrinsic group (twiddling) for batch dequantization.
    - fast_dequant (False): uses a simple elementwise dequantization helper.

    Parameters:
    M, N, K (int): matrix dimensions (A is MxK, result is MxN). K must be divisible by (block_K * split).
    in_dtype (str): element type of A (e.g., "fp4" in this file).
    out_dtype (str): output tensor element type (e.g., T.bfloat16).
    accum_dtype (str): accumulation type used for the inner GEMM.
    source_format (str, optional): format string passed to intrinsic selector (default "uint").
    num_bits (int, optional): number of bits per quantized element in B (default 4).
    scale_size (int, optional): number of elements grouped per scale entry (default 32).
    fast_dequant (bool, optional): choose the fast intrinsic dequantization path when available (default True).
    block_M, block_N, block_K (int, optional): tile sizes for M, N, and K dimensions (defaults 256, 128, 128).
    num_stages (int, optional): pipelining stages for K loop (default 2).
    threads (int, optional): threads per block used by the kernel (default 256).
    split (int, optional): split factor along K used by the scheduler (default 1).
    with_bias (bool, optional): whether to add Bias to the output (default False).

    Returns:
    A T.prim_func implementing the tiled, pipelined GEMM that:
    - loads tiled blocks of A and packed B to shared memory,
    - dequantizes B via the chosen path into a shared dequantized tile,
    - performs a tiled GEMM accumulating into local fragments,
    - writes the final MxN block to the global output tensor.

    Notes:
    - The function queries an intrinsic group to obtain a fast dequantization implementation when fast_dequant is enabled; that intrinsic must supply a valid C source and function name.
    - The kernel layout uses swizzled shared-memory layouts for A, B, and the shared C tile.
    - An assertion enforces that K % (block_K * split) == 0.
    """
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8
    QK = K // num_elems_per_byte
    Block_QK = block_K // num_elems_per_byte
    A_shape = (M, K)
    B_shape = (N, QK)
    Bias_shape = (M, N)
    Scale_shape = (N, K // scale_size)
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
        """
        Return a TileLang macro that performs fast dequantization of twiddled FP4-packed data into BF16.

        The returned macro has signature (B_shared, B_dequantize_shared, Scale, k) and:
        - Loads packed FP4 elements from B_shared into per-thread local registers.
        - Calls an external fast dequantization intrinsic (provided via `import_source` / `func_name` in the outer scope) to expand packed FP4 -> BF16 values.
        - Applies a per-block scale factor derived from the Scale tensor (using exponentiation by powers of two).
        - Writes the scaled BF16 results into B_dequantize_shared.

        Notes:
        - This factory only supports in_dtype="fp4" and out_dtype=T.bfloat16.
        - The macro depends on several names from the enclosing scope (e.g., import_source, func_name, DataType, num_elems_per_byte, storage_dtype, block_N, block_K, threads, scale_size); those must be defined and consistent with the kernel that will use the macro.
        - The macro issues a T.import_source and T.call_extern to invoke the external intrinsic; ensure the external implementation matching `func_name` is available at compilation/runtime.
        """
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale, k):
            # import fast_dequantize plugin
            """
            Fast dequantization kernel: convert packed 4-bit quantized values in B_shared to bfloat16
            in B_dequantize_shared using an external intrinsic optimized for twiddled (bit-packed) FP4,
            applying per-block scale factors from Scale.

            This routine is a tiled, thread-parallel helper that:
            - Imports and calls an external dequantization function (via `import_source`/`func_name`)
              to expand compressed uint8-packed FP4 values into BF16 fragments in-thread.
            - Loads the corresponding per-block scale entry, interprets it as an exponent bias
              (applies 2^(Scale - 127)), and multiplies the dequantized BF16 fragment by that factor.
            - Writes the scaled BF16 results back into the shared B_dequantize_shared buffer in-place.

            Parameters:
            - B_shared: read-only shared buffer containing compressed FP4 data (packed uint8 layout).
            - B_dequantize_shared: shared output buffer that is overwritten with BF16 dequantized values.
            - Scale: per-block scale tensor; entries are interpreted such that the multiplicative scale
              = 2^(Scale - 127).
            - k: block index along the K dimension used to select the appropriate Scale entries.

            Side effects:
            - Mutates B_dequantize_shared in shared memory.
            - Calls an external intrinsic function (must be provided by the environment via `import_source`
              and `func_name`) to perform the low-level unpacking/dequantization.
            """
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
                    T.address_of(B_local_thread[0]),
                    T.address_of(B_dequantize_local_thread[0]),
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
        """
        Create a simple (scalar) dequantization macro that converts 4-bit packed inputs to bfloat16.

        Returns a T.macro that, given shared-storage buffers B_shared, B_dequantize_shared, a Scale tensor, and block index k, unpacks 4-bit values from B_shared, converts each nibble to a bfloat16 value using _tir_u8_to_f4_to_bf16, applies the per-element exponential Scale, and writes the dequantized BF16 block into B_dequantize_shared.

        Notes:
        - Only supports in_dtype="fp4" and out_dtype=T.bfloat16.
        - The macro expects B_shared and B_dequantize_shared to have the shapes established in the enclosing scope (B_shared_shape, B_dequantize_shared_shape) and performs block-local copying into allocated fragments before elementwise conversion.
        - Scale holds the exponent-like scaling values indexed per output element as used by the conversion helper.
        """
        assert in_dtype in ["fp4"]
        assert out_dtype in [T.bfloat16]

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared, Scale, k):
            """
            Dequantizes a packed 4-bit (FP4) block from B_shared into BF16 values in B_dequantize_shared using per-element scale exponents.

            Per-element behavior:
            - Reads packed 4-bit entries from B_shared (uint8 storage, multiple nibbles per byte).
            - Uses Scale to obtain an exponent term (stored as uint8) and reconstructs BF16 values via _tir_u8_to_f4_to_bf16.
            - Writes the dequantized BF16 block into B_dequantize_shared.

            Parameters:
            - B_shared: shared-memory buffer holding packed 4-bit values (uint8-packed layout).
            - B_dequantize_shared: shared-memory buffer to receive dequantized BF16 results.
            - Scale: per-element exponent buffer; used to compute the scale factor for each dequantized element.
            - k: current block index along the K dimension (used to select the appropriate slice of Scale).

            Side effects:
            - Mutates B_dequantize_shared by storing the dequantized BF16 fragment.
            """
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

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        Scale: T.Tensor(Scale_shape, storage_dtype),
        Bias: T.Tensor(Bias_shape, out_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        """
        Tiled, pipelined kernel entry that multiplies A with a quantized B (with per-block Scale) producing C.

        This prim-level kernel implements a blocked, multi-threaded matmul: it loads tiles of A and the packed/quantized B into shared memory, dequantizes B (either via the fast intrinsic twiddling path or the simple per-element path), performs a block GEMM (with B transposed), and writes the accumulated block results into the output tensor C. The kernel allocates shared buffers for A, B, and the dequantized B, and a local fragment for accumulation; it runs over K in pipelined stages and expects the provided shapes and dtypes to match the tiling parameters used to build the function.

        Parameters are self-descriptive in the signature; notable behaviors:
        - B is stored in a compact uint8-packed layout (num_bits per element) and is dequantized using Scale before GEMM.
        - The selected dequantization path is controlled by the outer-scope flag `fast_dequant`.
        - The GEMM uses transpose_B=True (i.e., multiplies A · B^T after dequantization).
        - The function writes results in-place into C.
        """
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

    return main


def ref_program_twiddling(A, qB, Scale, Bias=None):
    """
    Compute A @ B^T where B is reconstructed from bit-twiddled 4-bit quantized data and per-block scales, returning bfloat16 results.

    Converts the quantized matrix `qB` to floating-point via `torch_convert_bit_twiddling`, applies a per-element scale factor of 2^(Scale - 127) (where Scale indexes are grouped by 32 columns of B), computes the matrix product A · B^T in float, and casts the result to bfloat16.

    Parameters:
        A (torch.Tensor): Left operand with shape (M, K), used in floating precision.
        qB (torch.Tensor): Quantized representation of B (packed 4-bit values) compatible with torch_convert_bit_twiddling.
        Scale (torch.Tensor): Per-column-group scale values; Scale indices correspond to groups of 32 columns in B.

    Returns:
        torch.Tensor: Resulting matrix C with shape (M, N) in bfloat16.
    """
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_twiddling_with_bias(A, qB, Scale, Bias):
    """
    Compute A @ B^T where B is reconstructed from bit-twiddled 4-bit quantized data and per-block scales, returning bfloat16 results.

    Converts the quantized matrix `qB` to floating-point via `torch_convert_bit_twiddling`, applies a per-element scale factor of 2^(Scale - 127) (where Scale indexes are grouped by 32 columns of B), computes the matrix product A · B^T in float, and casts the result to bfloat16.

    Parameters:
        A (torch.Tensor): Left operand with shape (M, K), used in floating precision.
        qB (torch.Tensor): Quantized representation of B (packed 4-bit values) compatible with torch_convert_bit_twiddling.
        Scale (torch.Tensor): Per-column-group scale values; Scale indices correspond to groups of 32 columns in B.
        Bias (torch.Tensor): Bias tensor with shape (M, N).

    Returns:
        torch.Tensor: Resulting matrix C with shape (M, N) in bfloat16.
    """
    dtypeC = T.bfloat16
    B = torch_convert_bit_twiddling(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float)) + Bias
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple(A, qB, Scale, Bias=None):
    """
    Compute a BF16 matrix product A · B^T from a quantized B with simple (non-twiddling) dequantization.

    Converts the quantized tensor `qB` to floating B via `torch_convert`, applies a per-element scale factor computed as 2^(Scale[i][j//32] - 127) (Scale supplies exponent offsets in 32-column groups), then computes C = A · B^T and returns the result converted to bfloat16.

    Parameters:
    - A: 2D tensor representing the left operand (will be cast to float32 for the matmul).
    - qB: Quantized representation of B accepted by `torch_convert`.
    - Scale: 2D tensor of exponent offsets; Scale[i][g] is applied to columns j where g == j // 32.

    Returns:
    - 2D bfloat16 tensor C containing the matrix product A · B^T.

    No in-place modification is performed on inputs (a local floating copy of B is scaled).
    """
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple_with_bias(A, qB, Scale, Bias):
    """
    Compute a BF16 matrix product A · B^T from a quantized B with simple (non-twiddling) dequantization.

    Converts the quantized tensor `qB` to floating B via `torch_convert`, applies a per-element scale factor computed as 2^(Scale[i][j//32] - 127) (Scale supplies exponent offsets in 32-column groups), then computes C = A · B^T and returns the result converted to bfloat16.

    Parameters:

    Returns:
    - A: 2D tensor representing the left operand (will be cast to float32 for the matmul).
    - qB: Quantized representation of B accepted by `torch_convert`.
    - Scale: 2D tensor of exponent offsets; Scale[i][g] is applied to columns j where g == j // 32.
    - Bias: 2D tensor representing the Bias (will be cast to float32 for the matmul).


    Returns:
    - 2D bfloat16 tensor C containing the matrix product A · B^T.

    No in-place modification is performed on inputs (a local floating copy of B is scaled).
    """
    dtypeC = T.bfloat16
    B = torch_convert(qB)
    B *= 2 ** (Scale[:, (torch.arange(B.shape[1], device=B.device) // 32)])
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float)) + Bias
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, scale_size=32, fast_dequant=True, with_bias=False, tune=False):
    """
    Run and validate the tiled quantized matmul kernel, then benchmark its latency and report TFLOPS.

    Builds a matmul kernel for the given matrix sizes and quantization scale size. If `tune` is True the kernel is obtained via the autotuning path; otherwise a fixed-parameter kernel is used. Validates numerical correctness against the appropriate reference implementation (bit-twiddling reference when `fast_dequant` is True, plain reference otherwise) with rtol/atol=0.01, prints a confirmation, then runs a benchmark (500 warmup iterations) and prints the measured latency (ms) and achieved TFLOPS.

    Parameters:
        m (int): Number of rows of A / output rows. Default 256.
        n (int): Number of columns of B / output columns. Default 256.
        k (int): Reduction dimension. Default 256.
        scale_size (int): Size of the per-block scale vector used for dequantization. Default 32.
        fast_dequant (bool): If True validate against the twiddling (fast dequant) reference and exercise the fast dequant path; otherwise use the simple dequant reference. Default True.
        tune (bool): If True obtain a tuned/autotuned kernel; otherwise use a fixed-parameter kernel. Default False.

    Returns:
        None
    """
    total_flops = 2 * m * n * k

    if tune:
        kernel = matmul(
            m, n, k, T.bfloat16, T.bfloat16, T.float32, num_bits=4, scale_size=scale_size, fast_dequant=fast_dequant, with_bias=with_bias
        )
    else:
        kernel = matmul(
            m,
            n,
            k,
            T.bfloat16,
            T.bfloat16,
            T.float32,
            num_bits=4,
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


def run_regression_perf(m=256, n=256, k=256, scale_size=32, fast_dequant=True, with_bias=False):
    kernel = matmul(
        m,
        n,
        k,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=4,
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
    return profiler.do_bench(warmup=500, backend="cupti")


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    scale_size = 32
    main(M, N, K, scale_size, fast_dequant=True, with_bias=True)
    main(M, N, K, scale_size, fast_dequant=False, with_bias=True)
    main(M, N, K, scale_size, fast_dequant=True, with_bias=False)
    main(M, N, K, scale_size, fast_dequant=False, with_bias=False)
