#include "compat/analyzer.h"
#include "op/prim.h"
#include "transform/common/loop_aware_mutator.h"

#include "compat/expr.h"
#include "support/check.h"

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

namespace {

class NegativeIndexRewriter : public ::tvm::tl::LoopAwareMutator {
public:
  explicit NegativeIndexRewriter(const sym::Analyzer &analyzer)
      : LoopAwareMutator(analyzer) {}

private:
  PrimExpr NormalizeIndex(const PrimExpr &index, const PrimExpr &extent) {
    PrimExpr value = tl::arith::Simplify(analyzer_, index);
    if (value.ty().IsScalar()) {
      return tl::arith::CanProve(analyzer_, value < 0)
                 ? tl::arith::Simplify(analyzer_, extent + value)
                 : index;
    }
    if (const auto *broadcast = value.as<BroadcastNode>()) {
      PrimExpr scalar = NormalizeIndex(broadcast->value, extent);
      return scalar.same_as(broadcast->value)
                 ? index
                 : Broadcast(scalar, broadcast->lanes, broadcast->loc);
    }
    if (const auto *ramp = value.as<RampNode>()) {
      auto lanes = GetConstInt(ramp->lanes);
      if (!lanes) {
        return index;
      }
      PrimExpr last =
          ramp->base +
          tl::MakeConst(ramp->stride.ty(), *lanes - 1) * ramp->stride;
      if (tl::arith::CanProve(analyzer_, Min(ramp->base, last) >= 0)) {
        return index;
      }
      if (tl::arith::CanProve(analyzer_, Max(ramp->base, last) < 0)) {
        return tl::arith::Simplify(analyzer_, extent + value);
      }
      ffi::Array<PrimExpr> values;
      ffi::Array<PrimExpr> positions;
      PrimType element_type = (extent + ramp->base).ty();
      for (int64_t lane = 0; lane < *lanes; ++lane) {
        PrimExpr scalar = tl::arith::Simplify(
            analyzer_,
            ramp->base + tl::MakeConst(ramp->stride.ty(), lane) * ramp->stride);
        scalar = Cast(element_type, scalar);
        PrimExpr wrapped = Cast(element_type, extent + scalar);
        values.push_back(tl::arith::Simplify(
            analyzer_, Select(scalar < 0, wrapped, scalar)));
        positions.push_back(IntImm(PrimType::Int(32), lane));
      }
      return tl::arith::Simplify(analyzer_,
                                 Shuffle(values, positions, value->loc));
    }
    return index;
  }

  ffi::Array<PrimExpr> NormalizeIndices(const Expr &tensor,
                                        const ffi::Array<PrimExpr> &indices) {
    const auto *type = tensor->ty.as<tirx::TensorTypeNode>();
    ICHECK(type);
    ICHECK_EQ(indices.size(), type->shape.size());
    ffi::Array<PrimExpr> normalized;
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      normalized.push_back(NormalizeIndex(indices[axis], type->shape[axis]));
    }
    return normalized;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    auto load = LoopAwareMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<TensorLoad>();
    return TensorLoad(load->source,
                      NormalizeIndices(load->source, load->indices), load->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto store = LoopAwareMutator::Mutate_(node, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    return TensorStore(store->dest,
                       NormalizeIndices(store->dest, store->indices),
                       store->value, store->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const TensorRegionNode *node,
                                 ffi::InplaceMode mode) final {
    Expr original = ffi::GetRef<Expr>(node);
    auto region = LoopAwareMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original)
                      .as_or_throw<TensorRegion>();
    const auto *type = region->source->ty.as<tirx::TensorTypeNode>();
    ICHECK(type);
    ffi::Array<Range> ranges;
    for (size_t axis = 0; axis < region->region.size(); ++axis) {
      const Range &range = region->region[axis];
      PrimExpr begin = NormalizeIndex(range->begin, type->shape[axis]);
      ranges.push_back(Range::FromMinExtent(begin, range->extent));
    }
    return TensorRegion(region->source, ranges, region->ty, region->loc);
  }
};

} // namespace

tvm::transform::Pass LegalizeNegativeIndexPass() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      sym::Analyzer analyzer;
      auto rewriter = ffi::make_object<NegativeIndexRewriter>(analyzer);
      SeqStmt original = function->body.value();
      function.CopyOnWrite()->body =
          rewriter->Mutate(original).ValueOrUnchanged(original);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LegalizeNegativeIndex");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeNegativeIndex",
                                   LegalizeNegativeIndexPass);
}

} // namespace tvm::tl
