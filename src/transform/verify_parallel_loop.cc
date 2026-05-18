#include "../op/utils.h"
#include "common/constr_visitor.h"
#include "layout_reducer.h"
#include "support/check.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {
using tvm::tl::ConstrSet;
using tvm::tl::ConstrVisitor;

struct ParallelLoopVerifier : public ConstrVisitor {
  std::vector<Var> parallel_loop_vars_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reducers;

  void VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kParallel) {
      parallel_loop_vars_.push_back(op->loop_var);
      ConstrVisitor::VisitStmt_(op);
      parallel_loop_vars_.pop_back();
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
    ConstrSet cset{constr_stack_};
    std::vector<Var> other_thread_vars_;
    Map<Var, PrimExpr> subs;
    for (const auto &var : parallel_loop_vars_) {
      Var v_other_thread(var->name_hint + "<OTHER>", var->dtype);
      other_thread_vars_.push_back(v_other_thread);
      subs.Set(var, v_other_thread);
    }
    cset.Extend(cset.Substitute(subs));
    for (const auto &idx : op->indices) {
      cset.AddConstr(idx == tirx::Substitute(idx, subs));
    }
    arith::Analyzer analyzer;
    cset.Populate(analyzer);
    // If we can prove the values are the same, then no data race can happen.
    if (analyzer.CanProve(op->value == tirx::Substitute(op->value, subs))) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    Array<Var> failed_vars;
    PrimExpr failed_var_expr;
    for (auto [k, v] : subs) {
      if (!analyzer.CanProve(k == v)) {
        failed_vars.push_back(k);
        failed_var_expr =
            failed_var_expr.defined() ? And(failed_var_expr, k == v) : (k == v);
      }
    }
    if (!failed_vars.empty()) {
      LOG(WARNING) << "Data race detected: `" << op->buffer << op->indices
                   << "`"
                   << "is written by multiple threads in loop " << failed_vars
                   << ", Example:\n"
                   << analyzer.z3_prover.GetModel(failed_var_expr)
                   << "If you believe this is a false positive, pass "
                      "`PassKey.TL_DISABLE_DATA_RACE_CHECK` to pass key to "
                      "disable this check.";
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
};

using namespace tirx::transform;

tvm::transform::Pass VerifyParallelLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ParallelLoopVerifier verifier;
    verifier(f->body);
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
