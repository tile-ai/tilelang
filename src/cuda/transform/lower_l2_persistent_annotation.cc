#include "compat/expr.h"
#include "op/utils.h"

#include <tvm/tirx/transform.h>

#include <cmath>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class L2PersistentLowerer : public StmtExprMutator {
private:
  Stmt Hint(const TensorVar &tensor, const PrimExpr &ratio,
            const Location &location) {
    TVM_FFI_CHECK(IsGlobalBuffer(tensor), ValueError)
        << "L2 persistence requires global tensor storage";
    auto constant = ratio.as<FloatImmNode>();
    TVM_FFI_CHECK(constant && std::isfinite(constant->value) &&
                      constant->value >= 0 && constant->value <= 1,
                  ValueError)
        << "L2 hit ratio must be a finite constant between zero and one";
    return Evaluate(
        Call(PrimType::Void(), Op::Get("tl.l2_persistent_hint"),
             {tensor, FloatImm(PrimType::Float(64), constant->value)},
             std::nullopt, {}, location));
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    auto attributes = node->attrs->dict;
    if (auto annotation = attributes.Get("l2_hit_ratio_map")) {
      auto ratios = annotation->cast<Map<Var, PrimExpr>>();
      Array<Stmt> statements;
      for (const auto &[variable, ratio] : ratios) {
        auto tensor = variable.as<TensorVar>();
        TVM_FFI_CHECK(tensor, TypeError)
            << "L2 persistence annotations must name native tensors";
        statements.push_back(Hint(*tensor, ratio, node->loc));
      }
      statements.push_back(body);
      body = SeqStmt(statements, node->loc);
      attributes.erase("l2_hit_ratio_map");
    }
    return RegionStmt(node->op, node->args, node->body_params,
                      DictAttrs(attributes), body, node->result_vars,
                      node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tl.l2_persistent_hint"))) {
      return Hint(call->args[0].as_or_throw<TensorVar>(),
                  call->args[1].as_or_throw<PrimExpr>(), node->loc);
    }
    return StmtExprMutator::Mutate_(node, mode);
  }
};
} // namespace

tvm::transform::Pass LowerL2Persistent() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        if (function->body) {
          auto lowerer = make_object<L2PersistentLowerer>();
          SeqStmt body = *function->body;
          function.CopyOnWrite()->body =
              lowerer->Mutate(body).ValueOrUnchanged(body);
        }
        return function;
      },
      0, "tl.LowerL2Persistent");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.l2_persistent_hint")
      .signature(sig::arg<TensorVar>("tensor", "Global tensor storage"),
                 sig::arg<PrimExpr>("hit_ratio",
                                    "Expected persistent-cache hit ratio"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  reflection::GlobalDef().def("tl.cuda.transform.LowerL2Persistent",
                              LowerL2Persistent);
}
} // namespace tvm::tl
