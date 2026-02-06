#ifndef TVM_TL_OP_TCGEN5_META_H_
#define TVM_TL_OP_TCGEN5_META_H_

#include <cstdint>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using runtime::DataType;

struct TCGEN5MMAMeta {
  int atom_m, atom_n, atom_k;
  bool enable_ws, enable_2cta;
};

inline std::pair<bool, TCGEN5MMAMeta>
GetTCGEN5MMAMeta(int M, int N, int K, DataType ab_dtype, DataType c_dtype) {
// TODO (lei) Currently not all shapes / dtypes are supported for TCGEN5MMA.
#define FAIL                                                                   \
  return {                                                                     \
    false, TCGEN5MMAMeta { 0, 0, 0, false, false }                             \
  }
#define SUCCESS(atom_m, atom_n, atom_k, use_ws, use_2cta)                      \
  return {                                                                     \
    true, TCGEN5MMAMeta { atom_m, atom_n, atom_k, use_ws, use_2cta }           \
  }
  std::vector<int> ws_valid_atom_ns = {256, 128, 64};
  if ((ab_dtype.is_bfloat16() || ab_dtype.is_float16()) &&
      (c_dtype.is_float() && c_dtype.bits() == 32)) {
    if (K % 16 != 0)
      FAIL;
    if (M % 128 == 0) {
      for (int atom_n = 256; atom_n >= 16; atom_n -= 16)
        if (N % atom_n == 0)
          SUCCESS(128, atom_n, 16, false, false);
      FAIL;
    } else if (M % 64 == 0) {
      for (int atom_n : ws_valid_atom_ns)
        if (N % atom_n == 0)
          SUCCESS(64, atom_n, 16, true, false);
      FAIL;
    } else if (M % 32 == 0) {
      for (int atom_n : ws_valid_atom_ns)
        if (N % atom_n == 0)
          SUCCESS(32, atom_n, 16, true, false);
      FAIL;
    } else {
      FAIL;
    }
  } else if ((ab_dtype.is_float8() || ab_dtype.is_float6_e2m3fn() ||
              ab_dtype.is_float6_e3m2fn() || ab_dtype.is_float4_e2m1fn()) &&
             ((c_dtype.is_float() && c_dtype.bits() == 32) ||
              (c_dtype.is_float16() && c_dtype.bits() == 16))) {
    if (K % 32 != 0)
      FAIL;
    if (M % 128 == 0) {
      for (int atom_n : ws_valid_atom_ns)
        if (N % atom_n == 0)
          SUCCESS(128, atom_n, 32, true, false);
      for (int atom_n = 256; atom_n >= 16; atom_n -= 16)
        if (N % atom_n == 0)
          SUCCESS(128, atom_n, 32, false, true);
      for (int atom_n = 256; atom_n >= 8; atom_n -= 8)
        if (N % atom_n == 0)
          SUCCESS(128, atom_n, 32, false, false);
      FAIL;
    } else if (M % 64 == 0) {
      for (int atom_n : ws_valid_atom_ns)
        if (N % atom_n == 0)
          SUCCESS(64, atom_n, 32, true, false);
      for (int atom_n = 256; atom_n >= 8; atom_n -= 8)
        if (N % atom_n == 0)
          SUCCESS(64, atom_n, 32, false, false);
      FAIL;
    } else if (M % 32 == 0) {
      for (int atom_n : ws_valid_atom_ns)
        if (N % atom_n == 0)
          SUCCESS(32, atom_n, 32, true, false);
      FAIL;
    } else {
      FAIL;
    }
  }
  FAIL;
#undef FAIL
#undef SUCCESS
}

inline uint32_t GetTCGEN5InstrDesc(int atom_m, int atom_n, int atom_k,
                                   DataType ab_dtype, DataType c_dtype,
                                   bool a_is_k_major, bool b_is_k_major,
                                   int scale_in_a, int scale_in_b) {
  ICHECK(atom_m % 16 == 0) << "atom_m must be divisible by 16";
  ICHECK(atom_n % 8 == 0) << "atom_n must be divisible by 8";
  ICHECK(atom_k == 16 || atom_k == 32)
      << "Unsupported atom_k for TCGEN5MMA descriptor: " << atom_k;
  ICHECK(scale_in_a == 1 || scale_in_a == -1)
      << "scale_in_a must be +/-1 for TCGEN5MMA";
  ICHECK(scale_in_b == 1 || scale_in_b == -1)
      << "scale_in_b must be +/-1 for TCGEN5MMA";

  auto encode_dtype = [&](DataType dtype) -> uint32_t {
    if (dtype.is_float16()) {
      return static_cast<uint32_t>(0);
    } else if (dtype.is_bfloat16()) {
      return static_cast<uint32_t>(1);
    } else if (dtype.is_float8_e4m3fn() || dtype.is_float8_e4m3fnuz() ||
               dtype.is_float8_e4m3()) {
      return static_cast<uint32_t>(0);
    } else if (dtype.is_float8_e5m2fnuz() || dtype.is_float8_e5m2()) {
      return static_cast<uint32_t>(1);
    }
    LOG(FATAL) << "Unsupported dtype for TCGEN5MMA descriptor: " << dtype;
    return 0u;
  };

  uint32_t a_format = encode_dtype(ab_dtype);
  uint32_t b_format = a_format;

  uint32_t c_format = 0;
  if (c_dtype.is_float16()) {
    c_format = 0;
  } else if (c_dtype.is_float()) {
    c_format = 1;
  } else if (c_dtype.is_int()) {
    c_format = 2;
  } else {
    LOG(FATAL) << "Unsupported accumulator dtype for TCGEN5MMA descriptor: "
               << c_dtype;
  }

  auto set_bits = [](uint32_t value, int start, int width) -> uint32_t {
    uint32_t mask = (width == 32) ? 0xFFFFFFFFu : ((1u << width) - 1);
    return (value & mask) << start;
  };

  uint32_t desc = 0;
  desc |= set_bits(0, 0, 2); // sparse_id2
  desc |= set_bits(0, 2, 1); // sparse_flag
  desc |= set_bits(0, 3, 1); // saturate
  desc |= set_bits(c_format, 4, 2);

  desc |= set_bits(a_format, 7, 3);
  desc |= set_bits(b_format, 10, 3);

  uint32_t a_neg = (scale_in_a == -1) ? 1u : 0u;
  uint32_t b_neg = (scale_in_b == -1) ? 1u : 0u;
  desc |= set_bits(a_neg, 13, 1);
  desc |= set_bits(b_neg, 14, 1);

  uint32_t a_major = a_is_k_major ? 0u : 1u;
  uint32_t b_major = b_is_k_major ? 0u : 1u;
  desc |= set_bits(a_major, 15, 1);
  desc |= set_bits(b_major, 16, 1);

  uint32_t n_dim = static_cast<uint32_t>(atom_n >> 3);
  uint32_t m_dim = static_cast<uint32_t>(atom_m >> 4);
  desc |= set_bits(n_dim, 17, 6);
  desc |= set_bits(0, 23, 1);
  desc |= set_bits(m_dim, 24, 5);
  desc |= set_bits(0, 29, 1);

  uint32_t max_shift = 0u;
  desc |= set_bits(max_shift, 30, 2);

  return desc;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_TCGEN5_META_H_
