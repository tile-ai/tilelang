#include "compat/visit.h"
#include "cuda/op/builtin.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {

tvm::transform::Pass PersistThreadblock() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    bool has_grid_sync = false;
    if (function->body) {
      PostOrderVisit(function->body.value(), [&](const ffi::ObjectRef &object) {
        if (const auto *call = object.as<CallNode>()) {
          has_grid_sync |= call->op.same_as(sync_grid());
        }
      });
    }
    if (has_grid_sync) {
      function = WithAttr(std::move(function), "use_cooperative_groups", true);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.PersistThreadblock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.PersistThreadblock",
                                   PersistThreadblock);
}

} // namespace tvm::tl
