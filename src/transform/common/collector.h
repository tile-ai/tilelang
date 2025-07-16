/*!
 * \file collector.h
 * \brief Collect information from the IR
 */

#include "arith/ir_visitor_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class ThreadTagChecker : public StmtExprVisitor {
public:
  static bool HasOnlyThreadIdxX(const PrimFunc &f) {
    ThreadTagChecker checker;
    checker(f->body);
    return checker.is_valid_;
  }

  static IterVar GetThreadVar(const Stmt &body) {
    ThreadTagChecker checker;
    checker(body);
    return checker.thread_var_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iter_var = Downcast<IterVar>(op->node);
      String thread_tag = iter_var->thread_tag;
      bool is_y_or_z =
          thread_tag == "threadIdx.y" || thread_tag == "threadIdx.z";

      if (!thread_tag.empty() && is_y_or_z && !is_one(iter_var->dom->extent)) {
        is_valid_ = false;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding) {
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        thread_var_ = Downcast<IterVar>(op->thread_binding);
      }
      bool is_y_or_z =
          thread_tag == "threadIdx.y" || thread_tag == "threadIdx.z";
      if (!thread_tag.empty() && is_y_or_z) {
        auto iter_var = Downcast<IterVar>(op->thread_binding);
        if (iter_var.defined() && iter_var->dom.defined() &&
            !is_one(iter_var->dom->extent)) {
          is_valid_ = false;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  IterVar thread_var_;
  bool is_valid_ = true;
};

} // namespace tl
} // namespace tvm
