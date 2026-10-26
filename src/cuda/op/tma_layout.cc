#include "compat/analyzer.h"
/*!
 * \file tl/cuda/op/tma_layout.cc
 * \brief Shared CUDA TMA layout analysis helpers.
 */

#include "cuda/op/tma_layout.h"
#include "op/prim.h"

#include <numeric>
#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tirx;

TMASharedLayoutAnalysis AnalyzeTMASharedLayout(const Layout &layout,
                                               PrimType dtype) {
  ffi::Optional<cute::ComposedLayout> composed =
      cute::ComposedLayoutFromTileLang(layout);
  if (!composed.has_value()) {
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
                             const tirx::TensorVar &shared_tensor,
                             const SwizzleMode &swizzle_mode) {
  if (!lower_args.require_smem_alignment) {
    return;
  }
  lower_args.require_smem_alignment(shared_tensor,
                                    swizzle_mode.SmemAlignment());
}

Layout MakeTmaLinearLayout(const ffi::Array<PrimExpr> &shape,
                           const ffi::Array<Range> &region, PrimType dtype) {
  ICHECK(region.empty() || region.size() == shape.size());

  // Physical order is [fixed slice, repeated boxes, box contents]. A fixed
  // pipeline version therefore owns one contiguous run of complete TMA boxes.
  ffi::Array<PrimExpr> fixed, outer, inner;
  for (size_t i = 0; i < shape.size(); i++) {
    PrimVar variable = InputPlaceholder(i);
    if (!region.empty() && IsOne(region[i]->extent)) {
      fixed.push_back(variable);
      continue;
    }
    auto s = GetConstInt(shape[i]);
    if (s.has_value() && *s > kTmaMaxBoxDim && *s % kTmaMaxBoxDim == 0) {
      outer.push_back(
          FloorDiv(variable, tl::MakeConst(variable.ty(), kTmaMaxBoxDim)));
      inner.push_back(
          FloorMod(variable, tl::MakeConst(variable.ty(), kTmaMaxBoxDim)));
    } else {
      inner.push_back(variable);
    }
  }
  ffi::Array<PrimExpr> forward;
  forward.insert(forward.end(), fixed.begin(), fixed.end());
  forward.insert(forward.end(), outer.begin(), outer.end());
  forward.insert(forward.end(), inner.begin(), inner.end());
  Layout layout(shape, forward);
  if (fixed.empty())
    return layout;
  auto physical_shape = layout->OutputShape();
  PrimExpr fixed_offset = 0;
  PrimExpr tile_offset = 0;
  PrimExpr tile_extent = 1;
  for (size_t axis = 0; axis < fixed.size(); ++axis)
    fixed_offset = fixed_offset * physical_shape[axis] + forward[axis];
  for (size_t axis = fixed.size(); axis < forward.size(); ++axis) {
    tile_offset = tile_offset * physical_shape[axis] + forward[axis];
    tile_extent = tile_extent * physical_shape[axis];
  }
  int64_t element_bits = dtype.bits() * dtype.lanes();
  int64_t alignment = std::lcm(int64_t{128 * 8}, element_bits) / element_bits;
  PrimExpr unit = tl::MakeConst(tile_extent.ty(), alignment);
  sym::Analyzer analyzer;
  PrimExpr stride = tl::arith::Simplify(
      analyzer, FloorDiv(tile_extent + unit - 1, unit) * unit);
  if (tl::arith::CanProveEqual(analyzer, stride, tile_extent))
    return layout;
  return Layout(shape, {fixed_offset * stride + tile_offset});
}

} // namespace cuda
} // namespace tl
} // namespace tvm
