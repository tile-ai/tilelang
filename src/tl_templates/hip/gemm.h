#pragma once

#include "common.h"
#include <type_traits>

namespace tl {

// Trait to determine the MFMA instruction to use based on data type
template <typename T> struct MfmaTraits;

// Specialization for int8
template <> struct MfmaTraits<int8_t> {
  template <typename AccType>
  static TL_DEVICE void mfma_op(const int8_t *b, const int8_t *a, AccType *c) {
    int64_t *b_packed = reinterpret_cast<int64_t *>(const_cast<int8_t *>(b));
    int64_t *a_packed = reinterpret_cast<int64_t *>(const_cast<int8_t *>(a));

    *c = __builtin_amdgcn_mfma_i32_16x16x32_i8(*b_packed, *a_packed, *c, 0, 0,
                                               0);
  }
};

// Specialization for half/float16
template <> struct MfmaTraits<half> {
  template <typename AccType>
  static TL_DEVICE void mfma_op(const half *b, const half *a, AccType *c) {
    *c = __builtin_amdgcn_mfma_f32_16x16x16f16(*((float16x4 *)b),
                                               *((float16x4 *)a), *c, 0, 0, 0);
  }
};

// Specialization for bfloat16_t
template <> struct MfmaTraits<bfloat16_t> {
  template <typename AccType>
  static TL_DEVICE void mfma_op(const bfloat16_t *b, const bfloat16_t *a,
                                AccType *c) {
    bfloat16x4_vec b_vec, a_vec;

    // Reinterpret the pointers
    short *b_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(b));
    short *a_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(a));

    // Copy the data
    for (int i = 0; i < 4; ++i) {
      b_vec[i] = b_short[i];
      a_vec[i] = a_short[i];
    }

    // Call the intrinsic and store the result directly to c
    *c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(b_vec, a_vec, *c, 0, 0, 0);
  }
};

#if defined(HIP_FP8_ENABLED)
// Specialization for fp8_e4_t
template <> struct MfmaTraits<fp8_e4_t> {
  template <typename AccType>
  static TL_DEVICE void mfma_op(const fp8_e4_t *b, const fp8_e4_t *a,
                                AccType *c) {
    int64_t a_val = *reinterpret_cast<const int64_t *>(a);
    int64_t b_val = *reinterpret_cast<const int64_t *>(b);
    *c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b_val, a_val, *c, 0, 0, 0);
  }
};
#endif

// ref to bitblas/tl/mfma_macro_generator.py::kPack
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type, typename AccDataType = float>
class GemmTensorOp {
public:
  // Note: clear_accum=true is not fully supported in HIP implementation
  // but we'll handle it by manually clearing the accumulator
  // static_assert(!clear_accum, "clear_accum=true is not supported yet");

  static constexpr int micro_size_x = 16;
  static constexpr int micro_size_y = 16;
  static constexpr int micro_size_k = 32 / sizeof(A_type);
  static constexpr int vec_size = 8 / sizeof(A_type);

  // This part comes from the Codegen
  static constexpr int M_Tile = M;
  static constexpr int N_Tile = N;
  static constexpr int K_Tile = K;

  static constexpr int block_row_warps = num_warp_m;
  static constexpr int block_col_warps = num_warp_n;

  static constexpr int inner_k = K_Tile / (micro_size_k * kPack);
  static constexpr int warp_rows = M_Tile / (block_row_warps * micro_size_x);
  static constexpr int warp_cols = N_Tile / (block_col_warps * micro_size_y);

  // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen
  // part.
  static constexpr bool kPadA = true;
  static constexpr bool kPadB = true;
  static constexpr bool kPadC = true;

  static constexpr int BANK_SIZE_BYTES = 128;

  static constexpr int warp_size = 64;

  TL_DEVICE static constexpr auto reverse_index_map(int thread_id,
                                                    int local_id) {
    return std::make_pair(thread_id % 16,
                          (thread_id / 16) * (vec_size * kPack) + local_id);
  }

  TL_DEVICE static constexpr auto reverse_index_map_transposed(int thread_id,
                                                               int local_id) {
    return std::make_pair((thread_id / 16) * (vec_size * kPack) + local_id,
                          thread_id % 16);
  }

  /*
   * Detailed Implementation please
   * checkout bitblas/tl/utils.py:get_swizzle_layout
   */
  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static auto make_mfma_swizzle_layout(const int row, const int col) {
    const auto dtype_bits = element_size * 8;

    const int numBanks = 32;
    const int bankBitWidth = 32;
    const int SIMDWidth = 16;
    const int vecSize = vec_size * kPack;
    const int innerDimLength = continuous;
    const int typeWidthInBit = dtype_bits;

    const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;
    const int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
    const int maxPhase =
        std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

    const int phase = (row / perPhase) % maxPhase;
    const int colOffSwizzled = (((col / vecSize) ^ phase) * vecSize);
    const int colOffOrdered = col % vecSize;
    const int colOff = colOffSwizzled + colOffOrdered;

    return std::make_pair(row, colOff);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_layout_padded(const int row,
                                                     const int col) {
    return std::make_pair(row, col);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_swizzle_layout(const int row,
                                                      const int col) {
    auto [n_row, n_col] =
        make_mfma_swizzle_layout<continuous, element_size>(row, col);
    return n_row * continuous + n_col;
  }

  static TL_DEVICE void body(A_type *A_shared, B_type *B_shared,
                             C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_n = warp_id / block_row_warps;
    auto warp_m = warp_id % block_row_warps;
    auto warp_row_tiles = warp_rows * micro_size_x;
    auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    A_type A_local[warp_rows * kPack * local_size_a];
    B_type B_local[warp_cols * kPack * local_size_b];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        const auto l = warp_m * warp_row_tiles + i * micro_size_x;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_a); local_id++) {
          if constexpr (TransposeA) {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    r + row, l + col)];
          } else {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    l + row, r + col)];
          }
        }
      }
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_b); local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) + (i * kPack + kp) * vec_size;

            // Use the trait to select the correct MFMA instruction, either fp8,
            // fp16 or bf16 currently
            MfmaTraits<A_type>::mfma_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_rs(A_type *A_local, B_type *B_shared,
                                C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_n = warp_id / block_row_warps;
    auto warp_m = warp_id % block_row_warps;
    auto warp_row_tiles = warp_rows * micro_size_x;
    auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    B_type B_local[warp_cols * kPack * local_size_b];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        const auto r = ki * kPack * micro_size_k;
        for (int local_id = 0; local_id < kPack * local_size_b; local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }

      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) +
                         (ki * warp_rows * kPack + i * kPack + kp) * vec_size;

            // Use the trait to select the correct MFMA instruction, either fp8,
            // fp16 or bf16 currently
            MfmaTraits<A_type>::mfma_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }
};

} // namespace tl

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type>;
  Compute::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type>;
  Compute::body_rs(pA, pB, accum);
}

} // namespace tl
