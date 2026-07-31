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
#include <utility>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {
using tvm::tl::ConstrSet;
using tvm::tl::ConstrVisitor;

// arith::Analyzer only models scalar arithmetic, and its Z3 backend aborts
// compilation on a vector node (Ramp, Broadcast, ...). Reduce a vector
// expression to one lane's scalar expression, or return an undefined Optional
// when that lane is not expressible as a scalar (e.g. a vector BufferLoad).
Optional<PrimExpr> ExtractLane(const PrimExpr &value, int lane) {
  if (value.dtype().is_scalar()) {
    return value;
  }
  if (const auto *broadcast = value.as<BroadcastNode>()) {
    return broadcast->value;
  }
  if (const auto *ramp = value.as<RampNode>()) {
    return ramp->base + make_const(ramp->base.dtype(), lane) * ramp->stride;
  }
  return Optional<PrimExpr>();
}

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
      PrimExpr other_idx = tirx::Substitute(idx, subs);
      if (idx.dtype().is_scalar()) {
        cset.AddConstr(idx == other_idx);
        continue;
      }
      // Constrain lanes one by one to keep the constraint scalar; dropping
      // it would lose injectivity and report a spurious data race.
      if (!idx.dtype().is_fixed_length_vector()) {
        continue;
      }
      for (int lane = 0; lane < idx.dtype().lanes(); ++lane) {
        Optional<PrimExpr> lane_idx = ExtractLane(idx, lane);
        Optional<PrimExpr> other_lane_idx = ExtractLane(other_idx, lane);
        if (lane_idx.defined() && other_lane_idx.defined()) {
          cset.AddConstr(lane_idx.value() == other_lane_idx.value());
        }
      }
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
    PrimExpr other_value = tirx::Substitute(op->value, subs);
    PrimExpr same_value;
    if (op->value.dtype().is_scalar()) {
      same_value = op->value == other_value;
    } else if (op->value.dtype().is_fixed_length_vector()) {
      // Lane by lane: a vector-valued predicate is accepted by neither Or()
      // nor the provers.
      same_value = Bool(true);
      for (int lane = 0; lane < op->value.dtype().lanes(); ++lane) {
        Optional<PrimExpr> lane_value = ExtractLane(op->value, lane);
        Optional<PrimExpr> other_lane_value = ExtractLane(other_value, lane);
        if (!lane_value.defined() || !other_lane_value.defined()) {
          // Not scalarizable: fall back to the same-iteration check.
          same_value = Bool(false);
          break;
        }
        same_value =
            And(same_value, EQ(lane_value.value(), other_lane_value.value()));
      }
    } else {
      // Scalable vector: the lane count is unknown at compile time.
      same_value = Bool(false);
    }
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
      LOG(WARNING) << "Data race detected: `" << op->buffer << op->indices
                   << "` "
                   << "is written by multiple threads in loop " << failed_vars
                   << ", Example:\n"
                   << analyzer.z3_prover.GetModel(race_free)
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
