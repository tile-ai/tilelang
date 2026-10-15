#include "../op/reducer.h"
#include "../op/utils.h"
#include "common/constr_visitor.h"
#include "span_utils.h"
#include "support/check.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include <sstream>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>
#include <utility>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {
using tvm::tl::ConstrSet;
using tvm::tl::ConstrVisitor;

struct ParallelLoopVerifier : public ConstrVisitor {
  /*! \brief One suspected data race, reported together at the end. */
  struct RaceReport {
    Buffer buffer;
    Array<PrimExpr> indices;
    Array<Var> failed_vars;
    String model;
    Span span;
    // Span of the innermost enclosing parallel loop; fallback location when
    // the store itself carries no span.
    Span loop_span;
  };

  std::vector<Var> parallel_loop_vars_;
  std::vector<Span> parallel_loop_spans_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reducers;
  std::vector<RaceReport> reports_;

  void VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kParallel) {
      parallel_loop_vars_.push_back(op->loop_var);
      parallel_loop_spans_.push_back(op->span);
      ConstrVisitor::VisitStmt_(op);
      parallel_loop_vars_.pop_back();
      parallel_loop_spans_.pop_back();
    } else {
      ConstrVisitor::VisitStmt_(op);
    }
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    if (reducers.count(op->buffer->data) ||
        IsLocalBuffer(op->buffer, /*allow_var=*/true)) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (parallel_loop_vars_.empty()) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    ConstrSet cset{constr_stack_};
    // Model a second logical iteration. Renaming starts at the outermost
    // parallel loop variable: binds before it are outside all parallelism and
    // stay shared, while the loop variables and anything inside the region are
    // private per iteration. Merge, so a shared bind is not populated twice.
    Map<Var, PrimExpr> subs;
    cset = cset.Merge(
        cset.RenameFrom("<OTHER>", subs, parallel_loop_vars_.front()));
    for (const auto &idx : op->indices) {
      cset.AddConstr(idx == tirx::Substitute(idx, subs));
    }
    arith::Analyzer analyzer;
    cset.Populate(analyzer);

    Array<Var> parallel_var_pairs;
    PrimExpr same_iteration = Bool(true);
    for (const auto &var : parallel_loop_vars_) {
      auto it = subs.find(var);
      if (it != subs.end()) {
        same_iteration = And(same_iteration, EQ(var, (*it).second));
        parallel_var_pairs.push_back(var);
      }
    }
    PrimExpr same_value = op->value == tirx::Substitute(op->value, subs);
    PrimExpr race_free = Or(same_iteration, same_value);
    if (analyzer.CanProve(race_free)) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    Array<Var> failed_vars;
    for (const auto &var : parallel_var_pairs) {
      if (!analyzer.CanProve(EQ(var, subs.at(var)))) {
        failed_vars.push_back(var);
      }
    }
    if (!failed_vars.empty()) {
      reports_.push_back({op->buffer, op->indices, failed_vars,
                          analyzer.z3_prover.GetModel(race_free), op->span,
                          parallel_loop_spans_.empty()
                              ? Span()
                              : parallel_loop_spans_.back()});
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) override {
    if (op->annotations.count(attr::kReducerInfo)) {
      auto map = op->annotations.Get(attr::kReducerInfo)
                     ->as<Map<Var, Map<String, String>>>();
      ICHECK(map) << "reducer_replication map is not defined";
      for (const auto &[var, info] : map.value()) {
        reducers.insert(var);
      }
    }
    return StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief Emit all collected races as one aggregated warning. */
  void EmitReport() const {
    if (reports_.empty()) {
      return;
    }
    std::ostringstream os;
    os << "Data race detected: " << reports_.size() << " potential race(s)\n";
    for (size_t k = 0; k < reports_.size(); ++k) {
      const RaceReport &report = reports_[k];
      os << "  [" << k + 1 << "] `" << report.buffer << report.indices
         << "` is written by multiple threads in loop " << report.failed_vars
         << SpanHintSuffix({report.span, report.loop_span}) << "\n"
         << "  Example:\n"
         << report.model;
      if (report.model.empty()) {
        os << "\n";
      }
    }
    os << "If you believe this is a false positive, disable the check by "
          "setting `PassKey.TL_DISABLE_DATA_RACE_CHECK` in the pass config, "
          "or by unsetting the `TILELANG_ENABLE_DATA_RACE_CHECK` environment "
          "variable (the check is disabled by default).";
    LOG(WARNING) << os.str();
  }
};

using namespace tirx::transform;

tvm::transform::Pass VerifyParallelLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ParallelLoopVerifier verifier;
    verifier(f->body);
    verifier.EmitReport();
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyParallelLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyParallelLoop", VerifyParallelLoop);
}

} // namespace

} // namespace tvm::tl
