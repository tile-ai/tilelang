#include "compat/analyzer.h"
#ifndef TL_TRANSFORM_TIRX_FLATTEN_TENSOR_UTILS_H_
#define TL_TRANSFORM_TIRX_FLATTEN_TENSOR_UTILS_H_
#include "op/prim.h"

#include <tvm/ir/prim/op.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>

#include <cstdint>
#include <limits>

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;

TVM_FFI_INLINE TensorVar
FlattenedTensor(const TensorVar &tensor,
                ffi::Optional<sym::Analyzer> analyzer = std::nullopt,
                bool fold_origin = false) {
  if (!analyzer.has_value())
    analyzer = sym::Analyzer();
  const auto &ana = analyzer.value();
  ffi::Array<PrimExpr> output_shape{1};
  const auto *tile = tensor.ty()->layout.as<TileLayoutNode>();
  bool explicit_strides =
      tile && MatchShapeStridePattern(tensor.ty()).has_value();
  if (explicit_strides) {

    PrimExpr span = 1;
    for (const auto &iter : tile->shard) {
      span = span + (iter->extent - 1) * iter->stride;
    }
    if (ffi::GetRef<TileLayout>(tile).IsContiguous(ana)) {
      for (const auto &extent : tensor.shape())
        output_shape.Set(0, output_shape[0] * extent);
    } else {
      output_shape.Set(
          0, tvm::max(tile->shard[0]->stride * tensor.shape()[0], span));
    }
    if (fold_origin) {

      static const Axis memory_axis = Axis::Get("m");
      if (auto origin = tile->offset.Get(memory_axis)) {
        output_shape.Set(0, output_shape[0] + origin.value());
      }
    }
  } else if (tile) {
    output_shape.Set(
        0, tl::arith::Simplify(
               ana, tensor.ty()->layout.value().GetSpan(ffi::String("m"))));
  } else if (const auto *compose =
                 tensor.ty()->layout.as<ComposeLayoutNode>()) {
    PrimExpr span = compose->tile_layout.GetSpan(ffi::String("m"));
    if (compose->tile_layout.IsTrivial()) {
      span = IntImm(tensor.ty()->DefaultIndexType(), 1);
      for (const auto &extent : tensor.shape())
        span = span * extent;
    }
    if (compose->swizzle_len != 0) {
      int changed_bits = compose->per_element + compose->swizzle_len;
      if (!compose->swizzle_inner)
        changed_bits += compose->atom_len;
      PrimExpr block = IntImm(span.ty(), int64_t{1} << changed_bits);
      span = floordiv(span + block - 1, block) * block;
    }
    output_shape.Set(0, tl::arith::Simplify(ana, span));
  } else {
    for (const auto &extent : tensor.shape())
      output_shape.Set(0, output_shape[0] * extent);
  }

  if (output_shape.size() == tensor.shape().size() &&
      !tensor.ty()->layout.has_value()) {
    return tensor;
  }
  return TensorVar(tensor->name,
                   TensorType(output_shape, tensor.dtype(), tensor.scope(),
                              tensor.ty()->data_alignment, std::nullopt),
                   tensor->loc);
}

template <typename IndexMapper>
ffi::Array<Range> MapTensorRegionBounds(const ffi::Array<Range> &regions,
                                        const sym::Analyzer &analyzer,
                                        IndexMapper map_indices) {
  ffi::Array<PrimExpr> indices;
  ffi::Map<Var, sym::IntSet> domains;
  bool empty = false;
  for (const Range &region : regions) {
    PrimVar index("region_index", region->begin.ty());
    indices.push_back(index);
    domains.Set(index, sym::IntSet::FromRange(region));
    empty |= tl::arith::CanProve(analyzer, region->extent <= 0);
  }
  ffi::Array<Range> mapped;
  if (empty) {
    ffi::Array<PrimExpr> begins;
    for (const Range &region : regions)
      begins.push_back(region->begin);
    for (const PrimExpr &begin : map_indices(begins)) {
      mapped.push_back(
          Range::FromMinExtent(begin, tl::MakeConst(begin.ty(), 0)));
    }
  } else {
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      tl::arith::Bind(analyzer, indices[axis].as_or_throw<Var>(),
                      regions[axis]);
    }
    for (const PrimExpr &index : map_indices(indices)) {
      auto covered = tl::arith::IntSetOf(analyzer, index, domains)
                         .CoverRange(std::nullopt);
      if (!covered.has_value()) {
        auto bounds = tl::arith::ConstIntBoundOf(analyzer, index);
        int64_t minimum = bounds->min_value;
        int64_t maximum = bounds->max_value;
        TVM_FFI_CHECK(minimum != sym::ConstIntBound::kNegInf &&
                          maximum != sym::ConstIntBound::kPosInf &&
                          minimum <= maximum,
                      ValueError)
            << "Cannot flatten a tensor region with an unbounded physical "
               "address range: "
            << regions;
        uint64_t extent =
            static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum) + 1;
        TVM_FFI_CHECK_LE(
            extent, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            ValueError)
            << "Flattened tensor region extent does not fit int64: " << regions;
        covered = Range::FromMinExtent(
            IntImm(PrimType::Int(64), minimum),
            IntImm(PrimType::Int(64), static_cast<int64_t>(extent)));
      }
      TVM_FFI_CHECK(covered.has_value(), ValueError)
          << "Cannot flatten a tensor region with an unbounded physical "
             "address range: "
          << regions;
      mapped.push_back(Range::FromMinExtent(
          tl::arith::Simplify(analyzer, covered.value()->begin),
          tl::arith::Simplify(analyzer, covered.value()->extent)));
    }
  }
  return mapped;
}

} // namespace tvm::tl::tirx_passes

#endif
