#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "transform/common/attr.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {

tvm::transform::Pass MarkCudaSyncCallsPass(bool support_pdl) {
  auto pass =
      [support_pdl](
          tirx::Function function, IRModule,
          tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    bool has_trigger = false;
    bool has_sync = false;
    bool has_ldg = false;
    if (function->body) {
      PostOrderVisit(function->body.value(), [&](const ffi::ObjectRef &object) {
        if (const auto *call = object.as<CallNode>()) {
          has_trigger |= call->op.same_as(pdl_trigger());
          has_sync |= call->op.same_as(pdl_sync());
          has_ldg |= call->op.same_as(__ldg());
        }
      });
    }
    TVM_FFI_CHECK(support_pdl || (!has_trigger && !has_sync), ValueError)
        << "PDL is not supported";
    TVM_FFI_CHECK(!has_sync || !has_ldg, ValueError)
        << "Cannot invoke __ldg function with pdl_sync";
    if (has_trigger) {
      function = WithAttr(std::move(function), attr::kHasTriggerLaunch, true);
    }
    if (has_sync) {
      function = WithAttr(std::move(function), attr::kHasGridSync, true);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.MarkCudaSyncCalls");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.MarkCudaSyncCalls",
                                   MarkCudaSyncCallsPass);
}

} // namespace tvm::tl
