/*!
 * \file if_condition_extract.cc
 * \brief Extract if conditions into temporary LetStmt variables, then expand if
 * statements to all branches.
 */

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

class IfConditionExtractor : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = IfConditionExtractor();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  IfConditionExtractor() = default;

  // counter to generate unique name for each IfStmt
  int counter_ = 0;

  //! \brief Check if the expression is a simple variable.
  bool IsSimpleVar(const PrimExpr &expr) {
    return expr.as<VarNode>() != nullptr;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }

    std::string var_name = "__cond_" + std::to_string(counter_++);
    Var cond_var(var_name, DataType::Bool());
    bool is_simple = true;

    if (IsSimpleVar(condition)) {
      // If the condition is already a simple variable, no need to extract it.
      cond_var = Downcast<Var>(condition);
    } else {
      is_simple = false;
    }

    auto bind_cond_var = [](const Stmt &sentence, const Var &cond) -> Stmt {
      if (auto if_sentence = sentence.as<IfThenElseNode>()) {
        PrimExpr new_cond = cond & if_sentence->condition;
        return IfThenElse(new_cond, if_sentence->then_case,
                          if_sentence->else_case);
      } else {
        return IfThenElse(cond, sentence);
      }
    };

    auto bind_cond_var_body = [&](const Optional<Stmt> &body,
                                  const Var &cond) -> Stmt {
      if (!body.defined()) {
        return Stmt();
      }
      if (auto seq = body.as<SeqStmtNode>()) {
        Array<Stmt> new_seq;
        for (auto sentence : seq->seq) {
          new_seq.push_back(bind_cond_var(sentence, cond));
        }
        return SeqStmt(std::move(new_seq));
      } else {
        return bind_cond_var(body.value(), cond);
      }
    };

    Array<Stmt> new_seq;
    new_seq.insert(new_seq.end(), bind_cond_var_body(then_case, cond_var));
    if (else_case.defined())
      new_seq.insert(new_seq.end(), bind_cond_var_body(else_case, cond_var));

    Stmt body =
        new_seq.empty()
            ? Stmt()
            : (new_seq.size() == 1 ? new_seq[0] : SeqStmt(std::move(new_seq)));
    if (is_simple) {
      return body;
    } else {
      return LetStmt(cond_var, condition, body);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (auto stmt : op->seq) {
      auto new_stmt = VisitStmt(stmt);
      if (!new_stmt.defined())
        continue;
      if (auto seq_node = new_stmt.as<SeqStmtNode>()) {
        seq.insert(seq.end(), seq_node->seq.begin(), seq_node->seq.end());
      } else {
        seq.push_back(new_stmt);
      }
    }
    return SeqStmt(std::move(seq));
  }
};

using namespace tir::transform;
tvm::transform::Pass IfConditionExtract() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return IfConditionExtractor::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfConditionExtract", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.IfConditionExtract", IfConditionExtract);
}

} // namespace tl
} // namespace tvm
