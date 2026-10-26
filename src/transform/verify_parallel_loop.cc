#include "common/constr_visitor.h"
#include "compat/analyzer.h"
#include "op/reducer.h"
#include "op/utils.h"
#include "span_utils.h"

#include <tvm/tirx/transform.h>

#include <sstream>

namespace tvm::tl {

namespace {

class ParallelLoopVerifier : public ConstrVisitor {
public:
  void EmitReport() const {
    if (reports_.empty()) {
      return;
    }
    std::ostringstream message;
    message << "Data race detected: " << reports_.size()
            << " potential race(s)\n";
    for (size_t index = 0; index < reports_.size(); ++index) {
      const auto &report = reports_[index];
      message << "  [" << index + 1 << "] `" << report.tensor << report.indices
              << "` is written by multiple threads in loop "
              << report.failed_variables
              << SpanHintSuffix({report.location, report.loop_location})
              << "\n  Example:\n"
              << report.model << "\n";
    }
    message << "If this is a false positive, disable the check with "
               "PassKey.TL_DISABLE_DATA_RACE_CHECK or unset "
               "TILELANG_ENABLE_DATA_RACE_CHECK.";
    LOG(WARNING) << message.str();
  }

private:
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  struct RaceReport {
    tirx::TensorVar tensor;
    ffi::Array<PrimExpr> indices;
    ffi::Array<PrimVar> failed_variables;
    ffi::String model;
    Location location;
    Location loop_location;
  };

  Result Visit_(const ForNode *node) final {
    bool parallel = node->kind == ForKind::kParallel;
    if (parallel) {
      parallel_variables_.push_back(node->var);
      parallel_locations_.push_back(node->loc);
    }
    auto result = ConstrVisitor::Visit_(node);
    if (parallel) {
      parallel_variables_.pop_back();
      parallel_locations_.pop_back();
    }
    return result;
  }

  Result Visit_(const TensorStoreNode *node) final {
    auto tensor = node->dest.as_or_throw<tirx::TensorVar>();
    if (parallel_variables_.empty() || IsLocalBuffer(tensor, true) ||
        IsReducerV2Buffer(tensor) || reducers_.count(tensor)) {
      return ConstrVisitor::Visit_(node);
    }
    ConstrSet constraints = GetConstrSet();
    ffi::Map<PrimVar, PrimExpr> replacements;
    constraints = constraints.Merge(constraints.RenameFrom(
        "<OTHER>", replacements, parallel_variables_.front()));
    for (const PrimExpr &index : node->indices) {
      FreshenMutableReads first, second;
      constraints.AddConstr(first(index) ==
                            second(Substitute(index, replacements)));
    }
    sym::Analyzer analyzer;
    auto context = constraints.Populate(analyzer);
    PrimExpr same_iteration = IntImm::Bool(true);
    ffi::Array<PrimVar> iteration_variables;
    for (const PrimVar &variable : parallel_variables_) {
      if (auto other = replacements.Get(variable)) {
        same_iteration = same_iteration && (variable == other.value());
        iteration_variables.push_back(variable);
      }
    }
    FreshenMutableReads first_value, second_value;
    PrimExpr same_value = first_value(node->value) ==
                          second_value(Substitute(node->value, replacements));
    PrimExpr race_free = same_iteration || same_value;
    if (!tl::arith::CanProve(analyzer, race_free)) {
      ffi::Array<PrimVar> failed;
      for (const PrimVar &variable : iteration_variables) {
        if (!tl::arith::CanProve(analyzer,
                                 variable == replacements.at(variable))) {
          failed.push_back(variable);
        }
      }
      if (!failed.empty()) {
        reports_.push_back({tensor, node->indices, failed,
                            tl::arith::Z3GetModel(analyzer, race_free),
                            node->loc, parallel_locations_.back()});
      }
    }
    return ConstrVisitor::Visit_(node);
  }

  Result Visit_(const RegionStmtNode *node) final {
    auto saved = reducers_;
    if (auto info = node->attrs->dict.Get(attr::kReducerInfo)) {
      for (const auto &[tensor, metadata] :
           info->cast<ffi::Map<tirx::TensorVar, ffi::Any>>()) {
        reducers_.insert(tensor);
      }
    }
    auto result = ConstrVisitor::Visit_(node);
    reducers_ = std::move(saved);
    return result;
  }

  std::vector<PrimVar> parallel_variables_;
  std::vector<Location> parallel_locations_;
  std::unordered_set<tirx::TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      reducers_;
  std::vector<RaceReport> reports_;
};

} // namespace

tvm::transform::Pass VerifyParallelLoop() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      auto verifier = ffi::make_object<ParallelLoopVerifier>();
      verifier->Visit(function->body.value());
      verifier->EmitReport();
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.VerifyParallelLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyParallelLoop",
                                   VerifyParallelLoop);
}

} // namespace tvm::tl
