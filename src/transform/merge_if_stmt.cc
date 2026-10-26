#include "merge_if_stmt.h"
#include "compat/metadata.h"
#include "compat/region.h"

#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tvm::tl {

class MergeIfStmtRewriter : public tvm::tl::StmtExprMutator {
public:
  static Stmt Apply(const Stmt &statement) {
    auto rewriter = ffi::make_object<MergeIfStmtRewriter>();
    return rewriter->Mutate(statement).ValueOrUnchanged(statement);
  }

private:
  static Stmt PreserveBindingScope(const SeqStmt &body) {
    for (const Stmt &statement : body->seq) {
      const auto *region = statement.as<RegionStmtNode>();
      if (statement.as<BindNode>() ||
          (region && !region->result_vars.empty())) {
        return RegionStmt(::tvm::tl::scope_op(), {}, {}, DictAttrs(), body, {},
                          body->loc);
      }
    }
    return body;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *statement,
                                 ffi::InplaceMode inplace_mode) final {
    Stmt original = ffi::GetRef<Stmt>(statement);
    SeqStmt visited = tvm::tl::StmtExprMutator::Mutate_(statement, inplace_mode)
                          .ValueOrUnchanged(original)
                          .as_or_throw<SeqStmt>();
    ffi::Array<Stmt> result;
    bool merged = false;
    for (size_t index = 0; index < visited->seq.size();) {
      const auto *first = visited->seq[index].as<IfNode>();
      if (!first || first->else_case.has_value() ||
          SideEffect(first->condition) > CallEffectKind::kPure) {
        result.push_back(visited->seq[index++]);
        continue;
      }
      size_t end = index + 1;
      for (; end < visited->seq.size(); ++end) {
        const auto *next = visited->seq[end].as<IfNode>();
        if (!next || next->else_case.has_value() ||
            !prim::ExprDeepEqual()(first->condition, next->condition)) {
          break;
        }
      }
      if (end == index + 1) {
        result.push_back(visited->seq[index]);
      } else {
        ffi::Array<Stmt> bodies;
        for (size_t branch = index; branch < end; ++branch) {
          bodies.push_back(PreserveBindingScope(
              visited->seq[branch].as<IfNode>()->then_case));
        }
        SeqStmt body = Apply(SeqStmt(bodies, first->then_case->loc))
                           .as_or_throw<SeqStmt>();
        result.push_back(If(first->condition, body, std::nullopt, first->loc));
        merged = true;
      }
      index = end;
    }
    if (!merged) {
      if (visited.same_as(original)) {
        return ffi::Unchanged();
      }
      return visited;
    }
    return SeqStmt(result, statement->loc);
  }
};

Stmt ApplyMergeIfStmt(Stmt statement) {
  return MergeIfStmtRewriter::Apply(statement);
}

tirx::Function MergeIfStmtSubstitute(tirx::Function &function) {
  if (function->body.has_value()) {
    SeqStmt body =
        ApplyMergeIfStmt(function->body.value()).as_or_throw<SeqStmt>();
    function.CopyOnWrite()->body = body;
  }
  return function;
}

tvm::transform::Pass MergeIfStmt() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    return MergeIfStmtSubstitute(function);
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.MergeIfStmt");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MergeIfStmt", MergeIfStmt);
}

} // namespace tvm::tl
