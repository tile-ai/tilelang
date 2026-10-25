/*!
 * \brief Inject eligible global->shared copies into ROCm async-copy intrinsics.
 * \file rocm/transform/async_copy_injector.cc
 */
#include "rocm/transform/async_copy_injector.h"

#include "cuda/transform/ptx_async_copy_injector.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

ROCmAsyncCopyInjectResult InjectROCmAsyncCopy(const Stmt &body,
                                              bool explicit_async) {
  auto result = InjectPTXAsyncCopy(body, explicit_async);
  return {result.stmt, result.injected_ptx_async_copy};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.rocm.transform.InjectROCmAsyncCopy", [](bool explicit_async) {
        return tirx::transform::CreateFunctionPass(
            [explicit_async](
                tirx::Function function, IRModule,
                tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
              if (function->body) {
                function.CopyOnWrite()->body = SeqStmt(
                    InjectROCmAsyncCopy(*function->body, explicit_async).stmt);
              }
              return function;
            },
            0, "tl.rocm.InjectROCmAsyncCopy");
      });
}

} // namespace tvm::tl
