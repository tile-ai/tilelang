/*!
 * \file tl/cuda/op/tma_layout.h
 * \brief Shared CUDA TMA layout analysis helpers.
 */

#ifndef TVM_TL_CUDA_OP_TMA_LAYOUT_H_
#define TVM_TL_CUDA_OP_TMA_LAYOUT_H_

#include "layout/cute_layout.h"
#include "op/operator.h"

#include <optional>
#include <string>

namespace tvm {
namespace tl {
namespace cuda {

struct TMASharedLayoutEncoding {
  cute::ComposedLayout composed;
  cute::ComposedLayout composed_bytes;
  SwizzleMode swizzle_mode;
};

struct TMASharedLayoutAnalysis {
  std::optional<TMASharedLayoutEncoding> encoding;
  std::string reason;
};

// Recover the affine layout and require its byte-address swizzle to be one of
// the four modes representable by a TensorMap descriptor.
TMASharedLayoutAnalysis AnalyzeTMASharedLayout(const Layout &layout,
                                               DataType dtype);

// Record the shared-memory base alignment required by the selected TensorMap
// swizzle so later allocation merging cannot shift the swizzle phase.
void RequireTMASmemAlignment(const LowerArgs &lower_args,
                             const tirx::Buffer &shared_tensor,
                             const SwizzleMode &swizzle_mode);

} // namespace cuda
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_OP_TMA_LAYOUT_H_
