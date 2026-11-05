/*!
 * \file legalize_negative_index.cc
 * \brief Retain TileLang negative-index semantics on native tensor accesses.
 */

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/transform.h>

#include "../ir/analyzer.h"

namespace tvm::tl {
namespace {

class NegativeIndexRewriter : public tl::StmtExprMutatorWithAnalyzer {
public:
  explicit NegativeIndexRewriter(const sym::Analyzer &analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;

  SeqStmt Rewrite(const tirx::Function &function) {
    MarkTensorParamShapes(function);
    return Mutate(function->body).ValueOrUnchanged(function->body);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    auto load = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<TensorLoad>(op))
                    .as_or_throw<TensorLoad>();
    auto tensor = load->source.as_or_throw<tirx::TensorVar>();
    return TensorLoad(tensor, RewriteIndices(load->indices, tensor.shape()),
                      load->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    auto store = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();
    auto tensor = store->dest.as_or_throw<tirx::TensorVar>();
    return TensorStore(tensor, RewriteIndices(store->indices, tensor.shape()),
                       store->value, store->loc);
  }

private:
  PrimExpr RewriteScalar(const PrimExpr &index, const PrimExpr &extent) {
    // Preserve the existing contract: a provably negative scalar is wrapped
    // once. An unknown scalar is left to the bounds legalizer; it is not a
    // Python modulo index. Mixed vector lanes are handled separately below.
    if (GetSyncedAnalyzer()->CanProve(index < 0))
      return GetSyncedAnalyzer()->Simplify(extent + index);
    return index;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr> &indices,
                                      const ffi::Array<PrimExpr> &shape) {
    TVM_FFI_CHECK_EQ(indices.size(), shape.size(), ValueError);
    ffi::Array<PrimExpr> result;
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      const PrimExpr &index = indices[axis];
      const PrimExpr &extent = shape[axis];
      PrimExpr simplified = GetSyncedAnalyzer()->Simplify(index);
      if (simplified.ty().IsScalar()) {
        result.push_back(RewriteScalar(index, extent));
      } else if (const auto *ramp = simplified.as<prim::RampObj>()) {
        const auto *lanes = ramp->lanes.as<IntImmObj>();
        TVM_FFI_CHECK(lanes != nullptr, ValueError)
            << "Negative-index legalization requires a fixed lane count";
        // A wrapped negative lane can require the extent's wider type even
        // when the ramp itself uses int32.  All Shuffle inputs, including
        // unchanged non-negative lanes and both Select branches, must share
        // the promoted type.  Do not truncate a large tensor extent back to
        // the ramp's original index type.
        PrimType lane_type = (extent + ramp->base).ty().WithLanes(1);
        ffi::Array<PrimExpr> values;
        ffi::Array<PrimExpr> order;
        for (int lane = 0; lane < lanes->value; ++lane) {
          PrimExpr scalar =
              GetSyncedAnalyzer()->Simplify(ramp->base + ramp->stride * lane);
          PrimExpr value = RewriteScalar(scalar, extent);
          if (value.same_as(scalar) &&
              !GetSyncedAnalyzer()->CanProve(scalar >= 0)) {
            PrimExpr wrapped = prim::cast(extent + scalar, lane_type);
            PrimExpr original = prim::cast(scalar, lane_type);
            value = prim::Select(scalar < 0, wrapped, original);
          }
          values.push_back(prim::cast(value, lane_type));
          order.push_back(IntImm(PrimType::Int(32), lane));
        }
        result.push_back(GetSyncedAnalyzer()->Simplify(
            prim::Shuffle(values, order, index->loc)));
      } else if (const auto *broadcast = simplified.as<prim::BroadcastObj>()) {
        result.push_back(
            prim::Broadcast(RewriteScalar(broadcast->value, extent),
                            broadcast->lanes, index->loc));
      } else {
        result.push_back(index);
      }
    }
    return result;
  }
};

} // namespace

tvm::transform::Pass LegalizeNegativeIndexPass() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto analyzer = CreateTileLangAnalyzer();
    auto rewriter = ffi::make_object<NegativeIndexRewriter>(analyzer);
    function.CopyOnWrite()->body = rewriter->Rewrite(function);
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
