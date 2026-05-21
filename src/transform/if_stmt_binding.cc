/*!
 * \file if_stmt_binding.cc
 * \brief Bind the If Stmt to each Stmt in SeqStmt
 */

#include "support/check.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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
      return GetRef<Stmt>(op);
    }
    ICHECK(then_case.defined()) << "then_case must be defined";
    ICHECK(!else_case.defined()) << "else_case must be undefined";

    auto make_seq = [](Array<Stmt> seq) -> Stmt {
      ICHECK(!seq.empty());
      return seq.size() == 1 ? seq[0] : SeqStmt(std::move(seq));
    };

    auto bind_if_stmt = [&make_seq](const Optional<Stmt> &body,
                                    const PrimExpr &condition) -> Stmt {
      if (body.defined()) {
        auto stmt = body.value();
        if (auto seq_stmt = stmt.as<SeqStmtNode>()) {
          Array<Stmt> seq;
          const size_t n = seq_stmt->seq.size();
          size_t i = 0;
          for (; i < n && !seq_stmt->seq[i].as<BindNode>(); ++i) {
            seq.push_back(IfThenElse(condition, seq_stmt->seq[i], Stmt()));
          }

          // A direct Bind is emitted as a C/CUDA declaration. Keep it and the
          // following statements in one lexical block, matching old LetStmt
          // scope semantics.
          if (i < n) {
            Array<Stmt> bind_scope;
            for (; i < n; ++i) {
              bind_scope.push_back(seq_stmt->seq[i]);
            }
            seq.push_back(
                IfThenElse(condition, make_seq(std::move(bind_scope)), Stmt()));
          }
          return make_seq(std::move(seq));
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

using namespace tirx::transform;
tvm::transform::Pass IfStmtBinding() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return IfStmtBindingRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfStmtBinding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.IfStmtBinding", IfStmtBinding);
}

} // namespace tl
} // namespace tvm
