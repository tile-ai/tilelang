// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file if_stmt_binding.cc
 * \brief Bind the If Stmt to each Stmt in SeqStmt
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class IfStmtBindingRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = IfStmtBindingRewriter();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  IfStmtBindingRewriter() = default;

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto condition = op->condition;
    auto then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }

    auto bind_if_stmt = [](Optional<Stmt> body,
                           const PrimExpr condition) -> Stmt {
      if (body.defined()) {
        auto stmt = body.value();
        if (auto seq_stmt = stmt.as<SeqStmtNode>()) {
          Array<Stmt> seq_;
          for (auto s : seq_stmt->seq) {
            seq_.push_back(IfThenElse(condition, s, Stmt()));
          }
          return SeqStmt(std::move(seq_));
        } else {
          return IfThenElse(condition, stmt, Stmt());
        }
      } else {
        return Stmt();
      }
    };

    Array<Stmt> new_seq;

    if (then_case.defined()) {
      new_seq.push_back(bind_if_stmt(then_case, condition));
    }
    if (else_case.defined()) {
      new_seq.push_back(bind_if_stmt(else_case, !condition));
    }
    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(std::move(new_seq));
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (auto stmt : op->seq) {
      seq.push_back(VisitStmt(stmt));
    }
    return SeqStmt(std::move(seq));
  }
};

using namespace tir::transform;
tvm::transform::Pass IfStmtBinding() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return IfStmtBindingRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfStmtBinding", {});
}

TVM_REGISTER_GLOBAL("tl.transform.IfStmtBinding").set_body_typed(IfStmtBinding);

} // namespace tl
} // namespace tvm
