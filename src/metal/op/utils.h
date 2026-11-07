/*!
 * \file tl/metal/op/utils.h
 * \brief Metal-specific operator helpers.
 */

#ifndef TVM_TL_BACKEND_METAL_OP_UTILS_H_
#define TVM_TL_BACKEND_METAL_OP_UTILS_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "metal/target_utils.h"
#include "op/operator.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace metal {

inline bool IsSIMDGroupBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "metal.simdgroup";
}

inline bool IsCooperativeTensorBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "metal.cooperative_tensor";
}

// Matrix numbering for a compact 2D buffer owned by one complete simdgroup.
struct SIMDGroupRegion {
  int64_t min[2];
  int64_t extent[2];
  int64_t col_tiles;

  int64_t Index(int64_t row, int64_t col) const {
    return (min[0] + row) * col_tiles + min[1] + col;
  }
};

inline bool IsWholeBufferRegion(const Buffer &buffer,
                                const ffi::Array<Range> &region,
                                arith::Analyzer *analyzer) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), region.size());
  for (size_t d = 0; d < region.size(); ++d) {
    if (!analyzer->CanProveEqual(region[d]->min, 0) ||
        !analyzer->CanProveEqual(region[d]->extent, buffer->shape[d])) {
      return false;
    }
  }
  return true;
}

inline SIMDGroupRegion
GetSIMDGroupPartialRegion(const Buffer &buffer, const ffi::Array<Range> &region,
                          const LowerArgs &args, arith::Analyzer *analyzer,
                          const char *operation) {
  TVM_FFI_ICHECK(buffer->shape.size() == 2 && region.size() == 2);
  CHECK(buffer->strides.empty() &&
            analyzer->CanProveEqual(buffer->elem_offset, 0),
        ValueError)
      << operation << ": requires a compact buffer with zero elem_offset for "
      << buffer->name;
  const int warp_size = TargetMetalGetWarpSize(args.target);
  CHECK(analyzer->CanProveEqual(args.block_num_threads, warp_size), ValueError)
      << operation << ": requires exactly one simdgroup (" << warp_size
      << " threads) in the kernel launch, got " << args.block_num_threads;
  CHECK(analyzer->CanProveEqual(args.thread_bounds->min, 0) &&
            analyzer->CanProveEqual(args.thread_bounds->extent, warp_size),
        ValueError)
      << operation << ": requires a complete simdgroup execution range [0, "
      << warp_size << "), got " << args.thread_bounds;

  SIMDGroupRegion tiles;
  for (size_t d = 0; d < 2; ++d) {
    const auto *shape = buffer->shape[d].as<IntImmNode>();
    const auto *min = region[d]->min.as<IntImmNode>();
    const auto *extent = region[d]->extent.as<IntImmNode>();
    CHECK(shape && min && extent, ValueError)
        << operation << ": requires constant shape, min and extent for "
        << buffer->name;
    CHECK(shape->value % 8 == 0 && min->value % 8 == 0 &&
              extent->value % 8 == 0 && extent->value > 0,
          ValueError)
        << operation
        << ": requires 8-aligned shape, min and positive extent for "
        << buffer->name;
    CHECK(min->value >= 0 && extent->value <= shape->value &&
              min->value <= shape->value - extent->value,
          ValueError)
        << operation << ": region exceeds buffer bounds for " << buffer->name;
    tiles.min[d] = min->value / 8;
    tiles.extent[d] = extent->value / 8;
  }
  tiles.col_tiles = buffer->shape[1].as<IntImmNode>()->value / 8;
  const int64_t last_row = tiles.min[0] + tiles.extent[0] - 1;
  const int64_t last_col = tiles.min[1] + tiles.extent[1] - 1;
  constexpr int64_t kMaxMatrixIndex = std::numeric_limits<int32_t>::max();
  CHECK(last_col <= kMaxMatrixIndex &&
            last_row <= (kMaxMatrixIndex - last_col) / tiles.col_tiles,
        ValueError)
      << operation << ": tile index exceeds the int32 intrinsic range for "
      << buffer->name;
  return tiles;
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
    if (M % (m * kMPerWarp) != 0 || N % (n * kNPerWarp) != 0)
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
