/*!
 * \file tl/cuda/op/tma_layout.cc
 * \brief Shared CUDA TMA layout analysis helpers.
 */

#include "cuda/op/tma_layout.h"

#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace cuda {

TMASharedLayoutAnalysis AnalyzeTMASharedLayout(const Layout &layout,
                                               DataType dtype) {
  ffi::Optional<cute::ComposedLayout> composed =
      cute::ComposedLayoutFromTileLang(layout);
  if (!composed.defined()) {
    return {std::nullopt,
            "layout is not a recoverable XOR swizzle over an affine layout"};
  }

  cute::ComposedLayout composed_bytes =
      composed.value().Recast(dtype.bits(), /*new_bits=*/8);
  const cute::Swizzle &swizzle = composed_bytes->swizzle;
  if (!swizzle->IsTMACompatible()) {
    std::ostringstream reason;
    reason << "byte-address swizzle Sw<" << swizzle->b_bits << ","
           << swizzle->m_base << "," << swizzle->s_shift
           << "> is not representable by a TensorMap descriptor";
    return {std::nullopt, reason.str()};
  }

  TMASharedLayoutEncoding encoding{composed.value(), composed_bytes,
                                   swizzle->ToSwizzleMode()};
  return {std::move(encoding), ""};
}

void RequireTMASmemAlignment(const LowerArgs &lower_args,
                             const tirx::Buffer &shared_tensor,
                             const SwizzleMode &swizzle_mode) {
  if (!lower_args.require_smem_alignment) {
    return;
  }
  lower_args.require_smem_alignment(shared_tensor->data,
                                    swizzle_mode.SmemAlignment());
}

} // namespace cuda
} // namespace tl
} // namespace tvm
