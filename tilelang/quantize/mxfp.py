from typing import Literal
from tilelang import language as T

# Implementation asm for fp4 to bf16, using twiddling
# Reference: https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py#L11-L18
decode_f4_to_bf16_twiddling = """
// N should be the number of elements processed by one thread
template<typename T1, typename T2>
__device__ void decode_fp4_to_bf16_twiddling(T1 *B_local, T2 *B_local_decode, const int N = 8) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    uint B_dequantize_local_vec[4];
    uint tmp, bias, d0, d1, d2, d3, d4, d5, d6;
    asm volatile(
      // To handle the endianness issue
      "prmt.b32 %13, %4, 0, 0x0123;"
      "mov.b32 %12, 0x7e807e80;"
      "and.b32 %0, %13, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %0, %0, %12;"
      "shl.b32 %1, %13, 3;"
      "and.b32 %1, %1, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %1, %1, %12;"
      "shl.b32 %2, %13, 6;"
      "and.b32 %2, %2, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %2, %2, %12;"
      "shl.b32 %5, %13, 1;"
      "and.b32 %6, %5, 0b10000000000000001000000000000000;"
      "shr.b32 %7, %13, 3;"
      "and.b32 %8, %7, 0b00000001100000000000000110000000;"
      "or.b32 %9, %6, %8;"
      "shr.b32 %10, %13, 7;"
      "and.b32 %11, %10, 0b00000000010000000000000001000000;"
      "or.b32 %3, %9, %11;"
      "mul.bf16x2 %3, %3, %12;"
      :"=r"(B_dequantize_local_vec[0])
      ,"=r"(B_dequantize_local_vec[1])
      ,"=r"(B_dequantize_local_vec[2])
      ,"=r"(B_dequantize_local_vec[3])
      :"r"(*(uint*)&B_local[i << 2]), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(bias), "r"(tmp)
    );
    for (int j = 0; j < 4; ++j) {
      // Pay attention to the big-endianness issue
      B_local_decode[(i << 3) + j] = reinterpret_cast<T2*>(&B_dequantize_local_vec[j])[1];
      B_local_decode[(i << 3) + j + 4] = reinterpret_cast<T2*>(&B_dequantize_local_vec[j])[0];
    }
  }
  // Check if the synchronization is needed
}
"""


def get_mxfp_intrin_group(
    out_dtype: Literal[T.float16, T.bfloat16] = T.bfloat16,
    source_format: Literal[T.int, T.uint] = T.uint,
    source_bit: int = 4,
    storage_dtype: Literal[T.int32, T.int8, T.uint8] = T.uint8,
    use_twiddling: bool = False,
) -> dict[str, str]:
    """
    Return metadata for an MXFP decoding intrinsic: function name and C source string.

    Validates the requested output dtype, source format, and storage dtype, then constructs
    a lookup key of the form `fp{source_bit}_to_{f16|bf16}` (appending `_twiddling` when
    use_twiddling is True) to select the corresponding C source snippet and a matching
    function name `decode_fp{source_bit}_to_{f16|bf16}` (also optionally suffixed with
    `_twiddling`).

    Parameters:
        out_dtype: Target floating-point type for decoded values; either T.float16 or T.bfloat16.
        source_format: Integer source representation; "int" or "uint".
        source_bit: Bit width of the packed source format (e.g., 4).
        storage_dtype: Underlying storage integer dtype (one of T.int32, T.int8, T.uint8).
        use_twiddling: When True, select the twiddling variant of the decoding intrinsic.

    Returns:
        A dict with:
          - "func_name": the generated C function name string for the requested decode intrinsic.
          - "c_source": the C source string for that intrinsic.

    Raises:
        AssertionError: if out_dtype, source_format, or storage_dtype are not supported.
        KeyError: if the constructed key does not match any available C source implementation.
    """
    out_dtype, source_format, storage_dtype = T.dtype(out_dtype), T.dtype(source_format), T.dtype(storage_dtype)
    assert out_dtype in [T.float16, T.bfloat16], f"Invalid out_dtype: {out_dtype}. Expected 'float16' or 'bfloat16'."
    assert source_format in [T.int, T.uint], f"Invalid source_format: {source_format}. Expected 'int' or 'uint'."
    assert storage_dtype in [T.int32, T.int8, T.uint8], f"Invalid storage_dtype: {storage_dtype}. Expected 'int32' or 'int8' or 'uint8'."

    dtype_map = {T.float16: "f16", T.bfloat16: "bf16"}
    key = f"fp{source_bit}_to_{dtype_map[out_dtype]}"
    if use_twiddling:
        key += "_twiddling"

    import_c_map = {
        "fp4_to_bf16_twiddling": decode_f4_to_bf16_twiddling,
    }

    func_name = f"decode_fp{source_bit}_to_{dtype_map[out_dtype]}"
    if use_twiddling:
        func_name += "_twiddling"

    return {
        "func_name": func_name,
        "c_source": import_c_map[key],
    }
