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

using namespace tirx;

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

std::optional<TMASharedLayoutEncoding>
WidenTMASharedLayout(const Layout &layout, DataType dtype,
                     const TMASharedLayoutEncoding &encoding,
                     const SwizzleMode &mode) {
  // TensorMap modes are Sw<b,4,3> on byte addresses with b = the mode
  // ordinal; Recast keeps b, so require the same least width in element
  // space.
  ffi::Optional<cute::ComposedLayout> composed =
      cute::ComposedLayoutFromTileLang(layout, mode.CanonicalOrdinal());
  if (!composed.defined())
    return std::nullopt;
  ICHECK(ffi::StructuralEqual()(composed.value()->layout,
                                encoding.composed->layout))
      << "widening the swizzle of " << layout << " to " << mode
      << " changed the plain layout";
  cute::ComposedLayout composed_bytes =
      composed.value().Recast(dtype.bits(), /*new_bits=*/8);
  return TMASharedLayoutEncoding{composed.value(), composed_bytes,
                                 composed_bytes->swizzle->ToSwizzleMode()};
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

Layout MakeTmaLinearLayout(const ffi::Array<PrimExpr> &shape,
                           const ffi::Array<Range> &region) {
  ICHECK(region.empty() || region.size() == shape.size());

  // Physical order is [fixed slice, repeated boxes, box contents]. A fixed
  // pipeline version therefore owns one contiguous run of complete TMA boxes.
  ffi::Array<PrimExpr> fixed, outer, inner;
  for (size_t i = 0; i < shape.size(); i++) {
    Var v = InputPlaceholder(i);
    if (!region.empty() && is_one(region[i]->extent)) {
      fixed.push_back(v);
      continue;
    }
    const int64_t *s = as_const_int(shape[i]);
    if (s != nullptr && *s > kTmaMaxBoxDim && *s % kTmaMaxBoxDim == 0) {
      outer.push_back(FloorDiv(v, Integer(kTmaMaxBoxDim)));
      inner.push_back(FloorMod(v, Integer(kTmaMaxBoxDim)));
    } else {
      inner.push_back(v);
    }
  }
  ffi::Array<PrimExpr> forward;
  forward.insert(forward.end(), fixed.begin(), fixed.end());
  forward.insert(forward.end(), outer.begin(), outer.end());
  forward.insert(forward.end(), inner.begin(), inner.end());
  return Layout(shape, forward);
}

} // namespace cuda
} // namespace tl
} // namespace tvm
