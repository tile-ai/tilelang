/*!
 * \file if_stmt_binding.cc
 * \brief Merge the If Stmt in SeqStmt
 */

#include "merge_if_stmt.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class MergeIfStmtRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    f.CopyOnWrite()->body = MergeIfStmtRewriter::Apply(f->body);
    return f;
  }

  static Stmt Apply(Stmt stmt) {
    auto rewriter = MergeIfStmtRewriter();
    return rewriter(stmt);
  }

private:
  MergeIfStmtRewriter() = default;

  void FlattenAppend(const Stmt &s, Array<Stmt> *out) {
    if (const auto *seq = s.as<SeqStmtNode>()) {
      for (const Stmt &e : seq->seq) {
        FlattenAppend(e, out);
      }
    } else {
      out->push_back(s);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // First, recursively flatten nested SeqStmt so that
    //   SeqStmt{ if, SeqStmt{ if, SeqStmt{ if } } }
    // becomes a single-level sequence of [if, if, if].
    Array<Stmt> flat_seq;
    for (const Stmt &stmt : op->seq) {
      Stmt new_stmt = this->VisitStmt(stmt);
      FlattenAppend(new_stmt, &flat_seq);
    }

    // Then, merge consecutive IfThenElse (without else) that share the same
    // condition.
    Array<Stmt> new_seq;
    PrimExpr current_condition;
    Array<Stmt> current_if_bodies;

    for (const Stmt &stmt : flat_seq) {
      if (const auto *if_node = stmt.as<IfThenElseNode>()) {
        if (!if_node->else_case.defined()) {
          if (current_condition.defined() &&
              ExprDeepEqual()(current_condition, if_node->condition)) {
            current_if_bodies.push_back(if_node->then_case);
            continue;
          } else {
            if (!current_if_bodies.empty()) {
              auto if_stmt =
                  IfThenElse(current_condition,
                             current_if_bodies.size() == 1
                                 ? current_if_bodies[0]
                                 : this->VisitStmt(SeqStmt(current_if_bodies)),
                             Stmt());
              new_seq.push_back(if_stmt);
              current_if_bodies.clear();
            }

            current_condition = if_node->condition;
            current_if_bodies.push_back(if_node->then_case);
            continue;
          }
        }
      }

      if (!current_if_bodies.empty()) {
        auto if_stmt =
            IfThenElse(current_condition,
                       current_if_bodies.size() == 1
                           ? current_if_bodies[0]
                           : this->VisitStmt(SeqStmt(current_if_bodies)),
                       Stmt());
        new_seq.push_back(if_stmt);
        current_condition = PrimExpr();
        current_if_bodies.clear();
      }

      new_seq.push_back(stmt);
    }

    if (!current_if_bodies.empty()) {
      auto if_stmt =
          IfThenElse(current_condition,
                     current_if_bodies.size() == 1
                         ? current_if_bodies[0]
                         : this->VisitStmt(SeqStmt(current_if_bodies)),
                     Stmt());
      new_seq.push_back(if_stmt);
    }

    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }
};

PrimFunc MergeIfStmtSubstitute(PrimFunc &f) {
  return MergeIfStmtRewriter::Substitute(f);
}

Stmt ApplyMergeIfStmt(Stmt stmt) { return MergeIfStmtRewriter::Apply(stmt); }

using namespace tir::transform;
tvm::transform::Pass MergeIfStmt() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MergeIfStmtRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeIfStmt", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MergeIfStmt", MergeIfStmt);
}

} // namespace tl
} // namespace tvm
