/*!
 * \file lower_frontend_scopes.cc
 * \brief Lower TileLang lexical allocation regions without losing their scope.
 */

#include "../ir/annotations.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class FrontendScopeLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    const auto fields = ReadAnnotations(op);
    if (fields.at("key").cast<ffi::String>() != "lexical_alloc_scope")
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK(op->body_params.empty() && op->result_vars.empty(),
                  ValueError)
        << "A lexical allocation scope cannot bind or return region values";
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    return MakeLexicalScope(body, op->loc, true);
  }
};
} // namespace

tvm::transform::Pass LowerFrontendScopes() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto mutator = ffi::make_object<FrontendScopeLowerer>();
    auto body = function->body;
    auto rewritten = mutator->Mutate(body).ValueOrUnchanged(body);
    if (!body.same_as(rewritten))
      function.CopyOnWrite()->body = rewritten;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerFrontendScopes");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerFrontendScopes",
                                   LowerFrontendScopes);
}
} // namespace tvm::tl
