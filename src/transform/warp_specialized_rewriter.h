/*!
 * \file warp_specialized_rewriter.h
 * \brief tools for warp-specialized-related analysis and transformation
 */

#pragma once

#include "arith/ir_visitor_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/builtin.h"
#include "./common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace runtime;
using arith::IRVisitorWithAnalyzer;

class WarpSpecializedDetector : public IRVisitorWithAnalyzer {
public:
  // return true means this aws will be disabled
  static bool Detect(const Stmt &stmt, bool skip_thread_partition = false) {
    WarpSpecializedDetector detector;
    detector.VisitStmt(stmt);
    if (detector.has_warp_specialization_) {
      LOG(WARNING) << "Auto warp specialization will be disabled because warp "
                      "specialization is manually enabled";
      return true;
    }
    if (detector.has_tma_op_ && detector.has_mbarrier_op_) {
      LOG(WARNING) << "Auto warp specialization will be disabled because TMA "
                      "and mbarrier are both present";
      return true;
    }
    return false;
  }

  WarpSpecializedDetector() {
    has_tma_op_ = false;
    has_mbarrier_op_ = false;
    has_warp_specialization_ = false;
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(create_list_of_mbarrier()) ||
          call->op.same_as(mbarrier_wait_parity()) ||
          call->op.same_as(builtin::ptx_arrive_barrier()) ||
          call->op.same_as(builtin::ptx_cp_async_barrier())) {
        has_mbarrier_op_ = true;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col()) ||
        op->op.same_as(set_max_nreg())) {
      has_tma_op_ = true;
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "warp_specialize" &&
        op->value.as<IntImmNode>()->value == 1) {
      has_warp_specialization_ = true;
    }
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  bool has_tma_op_{false};
  IterVar thread_var_;
  bool has_mbarrier_op_{false};
  bool has_warp_specialization_{false};
};

} // namespace tl
} // namespace tvm
