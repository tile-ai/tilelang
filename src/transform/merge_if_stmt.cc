/*!
 * \file merge_if_stmt.cc
 * \brief Merge adjacent native If statements with replayable conditions.
 */
#include "merge_if_stmt.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
using namespace tirx;

class MergeIfStmtRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    auto recursive = StmtExprMutator::Mutate_(op, mode);
    const bool recursive_unchanged = recursive.IsUnchanged();
    auto sequence = std::move(recursive)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                        .as_or_throw<SeqStmt>();
    ffi::Array<Stmt> flat;
    Flatten(sequence, &flat);
    ffi::Array<Stmt> result;
    bool merged = false;
    for (const Stmt &statement : flat) {
      const auto *next = statement.as<IfObj>();
      const auto *previous =
          result.empty() ? nullptr : result.back().as<IfObj>();
      if (previous && next && !previous->else_case && !next->else_case &&
          SideEffect(previous->condition) <= CallEffectKind::kPure &&
          prim::ExprDeepEqual()(previous->condition, next->condition)) {
        // Tensor loads and stateful calls cannot be coalesced: the first body
        // may change the value observed by the second condition. Scalar SSA
        // definitions have stable identity and remain safe to evaluate once.
        ffi::Array<Stmt> bodies = previous->then_case->seq;
        for (const Stmt &child : next->then_case->seq)
          bodies.push_back(child);
        SeqStmt body(bodies, previous->then_case->loc);
        body = Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
        result.Set(result.size() - 1, If(previous->condition, std::move(body),
                                         std::nullopt, previous->loc));
        merged = true;
      } else {
        result.push_back(statement);
      }
    }
    if (!merged && flat.size() == sequence->seq.size()) {
      if (recursive_unchanged)
        return ffi::Unchanged();
      return sequence;
    }
    return SeqStmt(result, sequence->loc);
  }

private:
  static void Flatten(const Stmt &statement, ffi::Array<Stmt> *result) {
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        Flatten(child, result);
    } else {
      result->push_back(statement);
    }
  }
};

Stmt ApplyMergeIfStmt(Stmt statement) {
  auto rewriter = ffi::make_object<MergeIfStmtRewriter>();
  return rewriter->Mutate(statement, InplaceMode::kDisallow)
      .ValueOrUnchanged(statement);
}

Function MergeIfStmtSubstitute(Function &function) {
  if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    function.CopyOnWrite()->body =
        ApplyMergeIfStmt(function->body).as_or_throw<SeqStmt>();
  return function;
}

tvm::transform::Pass MergeIfStmt() {
  auto rewrite = [](Function function, IRModule, tvm::transform::PassContext) {
    return MergeIfStmtSubstitute(function);
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0, "tl.MergeIfStmt");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MergeIfStmt", MergeIfStmt);
}

} // namespace tvm::tl
