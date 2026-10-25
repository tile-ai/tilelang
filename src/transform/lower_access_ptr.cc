#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class AccessPtrLowerer : public tirx::StmtExprMutator {
private:
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *call,
                                 ffi::InplaceMode inplace_mode) final {
    Expr original = ffi::GetRef<Expr>(call);
    Expr visited =
        StmtExprMutator::Mutate_(call, inplace_mode).ValueOrUnchanged(original);
    Call result = visited.as_or_throw<Call>();
    if (result->op.same_as(access_ptr())) {
      return LowerAccessPtrCall(result);
    }
    if (visited.same_as(original)) {
      return ffi::Unchanged();
    }
    return visited;
  }
};

} // namespace

namespace transform {

tvm::transform::Pass LowerAccessPtr() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body.has_value()) {
      auto lowerer = ffi::make_object<AccessPtrLowerer>();
      SeqStmt original = function->body.value();
      SeqStmt body = lowerer->Mutate(original)
                         .ValueOrUnchanged(original)
                         .as_or_throw<SeqStmt>();
      if (!body.same_as(original)) {
        function.CopyOnWrite()->body = std::move(body);
      }
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerAccessPtr");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerAccessPtr",
                                   LowerAccessPtr);
}

} // namespace transform
} // namespace tvm::tl
