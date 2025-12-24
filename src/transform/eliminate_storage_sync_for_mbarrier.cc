/*!
 * \file eliminate_storage_sync_for_mbarrier.cc
 */
#include "../op/builtin.h"
#include "./storage_access.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

class Eliminator : public IRMutatorWithAnalyzer {
public:
  static Stmt Substitute(const Stmt &stmt, bool skip_thread_partition = false) {
    arith::Analyzer analyzer;
    Eliminator transformer(&analyzer);
    return transformer.VisitStmt(stmt);
  }

  Eliminator(arith::Analyzer *analyzer) : IRMutatorWithAnalyzer(analyzer) {
    im_mbarrier_for_ = false;
    in_mbarrier_region_ = false;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "thread_extent") {
      if (const auto *var = op->node.as<VarNode>()) {
        if (var->name_hint == "threadIdx.x") {
          thread_extent_ = op;
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = nullptr;
    if (op->value->IsInstance<CallNode>()) {
      call = op->value.as<CallNode>();
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        // Skip storage sync if we're in a region with mbarrier operations
        // and we're not in a for loop with mbarrier operations
        if (in_mbarrier_region_ || im_mbarrier_for_) {
          return Stmt();
        }
      } else if (call->op.same_as(builtin::ptx_arrive_barrier()) ||
                 call->op.same_as(builtin::ptx_wait_barrier())) {
        in_mbarrier_region_ = true;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    bool old_in_mbarrier = in_mbarrier_region_;
    Stmt then_case = VisitStmt(op->then_case);

    Stmt ret;
    if (op->else_case.defined()) {
      in_mbarrier_region_ = old_in_mbarrier;
      Stmt else_case = VisitStmt(op->else_case.value());
      in_mbarrier_region_ = old_in_mbarrier || in_mbarrier_region_;
      ret = IfThenElse(VisitExpr(op->condition), then_case, else_case);
    } else {
      in_mbarrier_region_ = old_in_mbarrier || in_mbarrier_region_;
      ret = IfThenElse(VisitExpr(op->condition), then_case, Stmt());
    }
    return ret;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    PostOrderVisit(tvm::ffi::GetRef<For>(op), [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(create_list_of_mbarrier()) ||
            call->op.same_as(mbarrier_wait_parity()) ||
            call->op.same_as(builtin::ptx_arrive_barrier()) ||
            call->op.same_as(builtin::ptx_cp_async_barrier())) {
          im_mbarrier_for_ = true;
        }
      }
    });
    auto stmt = IRMutatorWithAnalyzer::VisitStmt_(op);
    im_mbarrier_for_ = false;
    return stmt;
  }

private:
  bool im_mbarrier_for_;
  bool in_mbarrier_region_;
  const AttrStmtNode *thread_extent_{nullptr};
};
using namespace tir::transform;

namespace transform {

tvm::transform::Pass EliminateStorageSyncForMBarrier() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = Eliminator::Substitute(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.EliminateStorageSyncForMBarrier",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.EliminateStorageSyncForMBarrier",
                        EliminateStorageSyncForMBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
