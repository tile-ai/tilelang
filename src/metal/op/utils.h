/*!
 * \file tl/metal/op/utils.h
 * \brief Metal-specific operator helpers.
 */

#ifndef TVM_TL_BACKEND_METAL_OP_UTILS_H_
#define TVM_TL_BACKEND_METAL_OP_UTILS_H_

#include <cmath>
#include <limits>
#include <utility>

#include "op/utils.h"

namespace tvm {
namespace tl {
namespace metal {

inline bool IsSIMDGroupBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "metal.simdgroup";
}

inline bool IsRegisterBuffer(const Buffer &buffer) {
  return IsFragmentBuffer(buffer) || IsSIMDGroupBuffer(buffer);
}

inline std::pair<int, int> ComputeSquareWarpPartition(int num_warps, int M,
                                                      int N, int kMPerWarp,
                                                      int kNPerWarp) {
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;
  float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

  int best_m = 1, best_n = 1;
  float best_balance = std::numeric_limits<float>::max();
  for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
    if (num_warps % m != 0)
      continue;
    int n = num_warps / m;
    if (n > max_n)
      continue;

    float m_per = static_cast<float>(M) / (m * kMPerWarp);
    float n_per = static_cast<float>(N) / (n * kNPerWarp);
    float balance = std::abs(m_per / n_per - ideal_ratio);
    if (balance < best_balance) {
      best_balance = balance;
      best_m = m;
      best_n = n;
    }
  }
  return {best_m, best_n};
}

} // namespace metal
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_METAL_OP_UTILS_H_
