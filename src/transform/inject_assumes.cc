#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
/*!
 * \file inject_assumes.cc
 * \brief Inject assumes on buffer's shape boundary check. Also convert
 * user-authored assumes to optimizer and runtime-check AttrNodes.
 */

#include "common/assume.h"
#include "common/attr.h"
#include "compat/expr.h"
#include "compat/region.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <unordered_set>

namespace tvm::tl {

namespace {

constexpr const char *kProcessedAssume = "tl.assume_processed";

class AssumeInjector : public tvm::tl::StmtExprMutator {
public:
  ffi::Array<Stmt> TensorFacts(const ffi::Array<tirx::TensorVar> &tensors) {
    ffi::Array<Stmt> facts;
    std::unordered_set<PrimExpr, tl::StructuralHash, tl::StructuralEqual> seen;
    auto add = [&](const PrimExpr &condition) {
      PrimExpr simplified = tl::arith::Simplify(analyzer_, condition);
      if (!tl::arith::CanProve(analyzer_, simplified) &&
          seen.insert(simplified).second) {
        facts.push_back(Evaluate(Assume(simplified)));
      }
    };
    for (const tirx::TensorVar &tensor : tensors) {
      for (const PrimExpr &extent : tensor.shape()) {
        if (!extent.as<IntImmNode>()) {
          add(extent >= tl::MakeConst(extent.ty(), 0));
        }
      }
      int bits = tensor.dtype().bits() * tensor.dtype().lanes();
      if (bits >= 8 || !tensor.ty()->layout ||
          tensor.ty()->HasDefaultLayout(analyzer_)) {
        continue;
      }
      if (auto pattern = tirx::MatchShapeStridePattern(tensor.ty())) {
        const auto &strides = pattern.value().get<1>();
        for (size_t axis = 0; axis + 1 < strides.size(); ++axis) {
          const auto &stride = strides[axis];
          if (!stride.as<IntImmNode>()) {
            add(floormod(stride, tl::MakeConst(stride.ty(), 8 / bits)) ==
                tl::MakeConst(stride.ty(), 0));
          }
        }
      }
    }
    return facts;
  }

private:
  static Call Assume(const PrimExpr &condition) {
    return Call(PrimType::Bool(), prim::assume_op(), {condition},
                DictAttrs({{kProcessedAssume, true}}));
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto binding = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                       .ValueOrUnchanged(original)
                       .as_or_throw<Bind>();
    auto tensor = binding->var.as<tirx::TensorVar>();
    if (!tensor) {
      return binding;
    }
    auto facts = TensorFacts({tensor.value()});
    if (facts.empty()) {
      return binding;
    }
    ffi::Array<Stmt> statements{binding};
    statements.insert(statements.end(), facts.begin(), facts.end());
    return SeqStmt(statements, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                                 ffi::InplaceMode mode) final {
    ffi::Array<Stmt> statements;
    for (const Stmt &statement : node->seq) {
      statements.push_back(Mutate(statement, mode).ValueOrUnchanged(statement));
    }
    ffi::Array<Stmt> suffix;
    for (size_t position = statements.size(); position > 0; --position) {
      const Stmt &statement = statements[position - 1];
      auto condition = GetAssumeExprInEvaluateForm(statement);
      if (condition) {
        const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
        bool processed = call->attrs && call->attrs.value()
                                            .as_or_throw<DictAttrs>()
                                            .GetAttr<bool>(kProcessedAssume)
                                            .value_or(false);
        if (!processed) {
          std::ostringstream message;
          message << "Assume: " << condition.value();
          ffi::Array<Stmt> body{
              Evaluate(Assume(condition.value()), statement->loc)};
          body.insert(body.end(), suffix.begin(), suffix.end());
          suffix = {RegionStmt(
              ::tvm::tl::scope_op(), {}, {},
              DictAttrs({{attr::kAssumeRequiresRuntimeCheck, condition.value()},
                         {"tl.assume_message", ffi::String(message.str())}}),
              SeqStmt(body), {}, statement->loc)};
          continue;
        }
      }
      suffix.insert(suffix.begin(), statement);
    }
    return SeqStmt(suffix, node->loc);
  }

  sym::Analyzer analyzer_;
};

} // namespace

tvm::transform::Pass InjectAssumes() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body) {
      return function;
    }
    auto injector = ffi::make_object<AssumeInjector>();
    SeqStmt original = function->body.value();
    SeqStmt body = injector->Mutate(original).ValueOrUnchanged(original);
    ffi::Array<tirx::TensorVar> tensors;
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<tirx::TensorVar>()) {
        tensors.push_back(tensor.value());
      }
    }
    auto facts = injector->TensorFacts(tensors);
    facts.push_back(body);
    function.CopyOnWrite()->body = SeqStmt(facts);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.InjectAssumes");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.InjectAssumes", InjectAssumes);
}

} // namespace tvm::tl
