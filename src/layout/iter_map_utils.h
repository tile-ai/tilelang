/*!
 * \file layout/iter_map_utils.h
 * \brief Coordinate views accepted by native affine iterator analysis.
 */

#ifndef TVM_TL_LAYOUT_ITER_MAP_UTILS_H_
#define TVM_TL_LAYOUT_ITER_MAP_UTILS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>
#include <unordered_map>

namespace tvm::tl {

struct LayoutIterMapView {
  ffi::Array<PrimExpr> indices;
  ffi::Map<PrimVar, Range> ranges;
  ffi::Map<PrimVar, PrimVar> coordinates;
};

// A layout's scalar placeholders use int32 independently of the physical
// index width. Native affine analysis accepts an iterator Var, but not a
// widening Cast around it. Project only coordinates whose every use is the
// same direct, lossless signed widening cast. Arithmetic inside a cast,
// narrowing casts, mixed widths and bare uses keep their original expression.
// This is an analysis view; no source layout or runtime arithmetic is changed.
class LayoutIterMapCoordinates final : public tirx::StmtExprVisitor {
public:
  explicit LayoutIterMapCoordinates(const ffi::Map<PrimVar, Range> &ranges) {
    for (const auto &[var, range] : ranges)
      uses_.emplace(var.get(), Use{var, std::nullopt, false});
    // Coordinate-dependent bounds require a dependent-domain conversion.
    // Leave those coordinates untouched instead of leaking an old identity
    // into the projected range map.
    for (const auto &[var, range] : ranges)
      Visit(range);
    visiting_domain_ = false;
  }

  using tirx::StmtExprVisitor::Visit;
  using tirx::StmtExprVisitor::Visit_;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarObj *node) final {
    if (auto it = uses_.find(node); it != uses_.end())
      it->second.blocked = true;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::CastObj *node) final {
    if (visiting_domain_)
      return tirx::StmtExprVisitor::Visit_(node);
    if (auto var = node->value.as<PrimVar>()) {
      auto it = uses_.find(var->get());
      PrimType from = var->ty();
      PrimType to = node->ty.as_or_throw<PrimType>();
      if (it != uses_.end() && from.IsScalar() && to.IsScalar() &&
          from.code() == DLDataTypeCode::kDLInt && to.code() == from.code() &&
          to.bits() > from.bits()) {
        auto &use = it->second;
        if (use.widened && use.widened.value() != to)
          use.blocked = true;
        use.widened = to;
        return std::nullopt;
      }
    }
    return tirx::StmtExprVisitor::Visit_(node);
  }

  ffi::Map<PrimVar, PrimVar> Coordinates() const {
    ffi::Map<PrimVar, PrimVar> result;
    for (const auto &[node, use] : uses_) {
      if (!use.blocked && use.widened)
        result.Set(use.original,
                   PrimVar(use.original->name + "_iter", *use.widened));
    }
    return result;
  }

private:
  struct Use {
    PrimVar original;
    ffi::Optional<PrimType> widened;
    bool blocked;
  };
  std::unordered_map<const VarObj *, Use> uses_;
  bool visiting_domain_{true};
};

class LayoutIterMapCastRewriter final : public tirx::StmtExprMutator {
public:
  explicit LayoutIterMapCastRewriter(ffi::Map<PrimVar, PrimVar> coordinates)
      : coordinates_(std::move(coordinates)) {}
  using tirx::StmtExprMutator::Mutate;
  using tirx::StmtExprMutator::Mutate_;

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::CastObj *node,
                                     ffi::InplaceMode mode) final {
    if (auto var = node->value.as<PrimVar>()) {
      if (auto coordinate = coordinates_.Get(*var);
          coordinate && coordinate->ty() == node->ty.as_or_throw<PrimType>())
        return *coordinate;
    }
    return tirx::StmtExprMutator::Mutate_(node, mode);
  }

private:
  ffi::Map<PrimVar, PrimVar> coordinates_;
};

inline LayoutIterMapView
MakeLayoutIterMapView(const ffi::Array<PrimExpr> &indices,
                      const ffi::Map<PrimVar, Range> &ranges) {
  auto collector = ffi::make_object<LayoutIterMapCoordinates>(ranges);
  collector->Visit(indices);
  auto coordinates = collector->Coordinates();
  if (coordinates.empty())
    return {indices, ranges, coordinates};
  auto rewriter = ffi::make_object<LayoutIterMapCastRewriter>(coordinates);
  auto mapped_indices = indices.Map([&](const PrimExpr &index) {
    return rewriter->Mutate(index, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(index);
  });
  auto mapped_ranges = ranges;
  for (const auto &[original, coordinate] : coordinates) {
    mapped_ranges.erase(original);
    mapped_ranges.Set(coordinate, ranges.at(original));
  }
  return {mapped_indices, mapped_ranges, coordinates};
}

} // namespace tvm::tl

#endif // TVM_TL_LAYOUT_ITER_MAP_UTILS_H_
