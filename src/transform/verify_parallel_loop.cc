/*!
 * \file verify_parallel_loop.cc
 * \brief Diagnose native parallel stores using two logical executions.
 */

#include "common/constr_visitor.h"
#include "op/builtin.h"
#include "op/reducer.h"
#include "span_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <exception>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace prim;

struct ParallelLoopVerifier : public ConstrVisitor {
  struct RaceReport {
    tirx::TensorVar tensor;
    Array<PrimExpr> indices;
    Array<Var> failed_vars;
    String model;
    Location loc;
    Location loop_loc;
  };

  std::vector<Var> parallel_vars_;
  std::vector<Location> parallel_locs_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reducers_;
  std::vector<RaceReport> reports_;

  Result Visit_(const ForObj *op) override {
    if (op->kind != ForKind::kParallel)
      return ConstrVisitor::Visit_(op);
    parallel_vars_.push_back(op->var.as_or_throw<Var>());
    parallel_locs_.push_back(op->loc);
    auto result = ConstrVisitor::Visit_(op);
    parallel_vars_.pop_back();
    parallel_locs_.pop_back();
    return result;
  }

  Result Visit_(const SeqStmtObj *op) override {
    auto saved = reducers_;
    auto result = ConstrVisitor::Visit_(op);
    reducers_ = std::move(saved);
    return result;
  }

  void RecordReducers(const Map<Any, Any> &annotations) {
    if (!annotations.count(attr::kReducerInfo))
      return;
    for (const auto &[tensor, info] :
         annotations.at(attr::kReducerInfo).cast<Map<Any, Any>>()) {
      (void)info;
      reducers_.insert(tensor.cast<Var>());
    }
  }

  Result Visit_(const CallObj *op) override {
    if (auto callee = op->op.as<Op>();
        callee.has_value() && callee.value()->name == "tl.annotation") {
      RecordReducers(
          DecodeAnnotations(op->args[0].as_or_throw<tvm::Tuple>(),
                            op->attrs.value()
                                .as_or_throw<DictAttrs>()
                                .GetAttr<Array<Any>>("annotation_schema")
                                .value())
              .cast<Map<Any, Any>>());
      return std::nullopt;
    }
    return ConstrVisitor::Visit_(op);
  }

  Result Visit_(const RegionStmtObj *op) override {
    auto saved = reducers_;
    if (op->op->name == "tl.attr") {
      auto values =
          DecodeAnnotations(
              op->args[0].as_or_throw<tvm::Tuple>(),
              op->attrs.GetAttr<Array<Any>>("annotation_schema").value())
              .cast<Map<Any, Any>>();
      if (values.at("key").cast<String>() == attr::kReducerInfo)
        RecordReducers({{String(attr::kReducerInfo), values.at("value")}});
    }
    auto result = ConstrVisitor::Visit_(op);
    reducers_ = std::move(saved);
    return result;
  }

  Result Visit_(const TensorStoreObj *op) override {
    auto tensor = op->dest.as<tirx::TensorVar>();
    if (!tensor.has_value() || parallel_vars_.empty())
      return ConstrVisitor::Visit_(op);
    const String scope = tensor.value().scope().name();
    if (reducers_.count(tensor.value()) || scope == "local" ||
        scope == "local.var")
      return ConstrVisitor::Visit_(op);

    // Definitions outside the first parallel loop are shared. Definitions
    // inside it must have a distinct identity in the other logical execution.
    ConstrSet cset{.constrs_ = constr_stack_};
    Map<Var, PrimExpr> subs;
    cset = cset.Merge(cset.RenameFrom("<OTHER>", subs, parallel_vars_.front()));
    for (const PrimExpr &index : op->indices)
      cset.AddConstr(index == SubstituteConstraintExpr(index, subs));
    auto analyzer = CreateTileLangAnalyzer();
    auto context = cset.Populate(analyzer);

    PrimExpr same_iteration = IntImm(PrimType::Bool(), 1);
    Array<Var> pairs;
    for (const Var &var : parallel_vars_) {
      if (auto other = subs.Get(var)) {
        same_iteration =
            And(same_iteration, var.as_or_throw<PrimExpr>() == *other);
        pairs.push_back(var);
      }
    }
    // A direct mutable read, just like a captured Bind, may evaluate to a
    // different value at the other access point. Do not obtain same-value
    // permission by treating two live memory reads as one immutable expression.
    auto first = make_object<FreshenMutableReads>();
    auto second = make_object<FreshenMutableReads>();
    PrimExpr same_value = (*first)(op->value) ==
                          (*second)(SubstituteConstraintExpr(op->value, subs));
    PrimExpr race_free = Or(same_iteration, same_value);
    if (analyzer->CanProve(race_free))
      return ConstrVisitor::Visit_(op);

    Array<Var> failed;
    for (const Var &var : pairs)
      if (!analyzer->CanProve(var.as_or_throw<PrimExpr>() == subs.at(var)))
        failed.push_back(var);
    if (!failed.empty()) {
      String model;
      try {
        model = analyzer->z3_prover.GetModel(race_free);
      } catch (const std::exception &) {
        // Unsupported solver terms must not suppress the diagnostic itself.
      }
      reports_.push_back({tensor.value(), op->indices, failed, model, op->loc,
                          parallel_locs_.empty() ? Location(UnknownLoc())
                                                 : parallel_locs_.back()});
    }
    return ConstrVisitor::Visit_(op);
  }

  void EmitReport() const {
    if (reports_.empty())
      return;
    std::ostringstream os;
    os << "Data race detected: " << reports_.size() << " potential race(s)\n";
    for (size_t k = 0; k < reports_.size(); ++k) {
      const auto &report = reports_[k];
      os << "  [" << k + 1 << "] `" << report.tensor << report.indices
         << "` is written by multiple threads in loop " << report.failed_vars
         << SpanHintSuffix({report.loc, report.loop_loc}) << "\n"
         << "  Example:\n"
         << report.model;
      if (report.model.empty())
        os << "\n";
    }
    os << "If you believe this is a false positive, disable the check by "
          "setting `PassKey.TL_DISABLE_DATA_RACE_CHECK` in the pass config, "
          "or by unsetting the `TILELANG_ENABLE_DATA_RACE_CHECK` environment "
          "variable (the check is disabled by default).";
    LOG(WARNING) << os.str();
  }
};

tvm::transform::Pass VerifyParallelLoop() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext ctx) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) ||
        ctx->GetConfig<bool>(kDisableDataRaceCheck).value_or(false))
      return function;
    auto verifier = make_object<ParallelLoopVerifier>();
    verifier->Visit(function->body);
    verifier->EmitReport();
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.VerifyParallelLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyParallelLoop",
                                   VerifyParallelLoop);
}
} // namespace
} // namespace tvm::tl
