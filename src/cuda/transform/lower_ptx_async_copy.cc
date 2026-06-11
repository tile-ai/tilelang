/*!
 * \brief Lower eligible global->shared copies into PTX cp.async.
 * \file cuda/transform/lower_ptx_async_copy.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/transform.h>

#include "cuda/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass LowerPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      return f;
    }
    Target target = target_opt.value();
    if (!TargetIsCuda(target)) {
      return f;
    }

    if (!TargetCudaHasAsyncCopy(target)) {
      // Graceful fallback on older architectures.
      return f;
    }

    bool enable_auto_async_copy =
        ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();

    auto *n = f.CopyOnWrite();
    auto inject_result =
        InjectPTXAsyncCopy(n->body, enable_auto_async_copy,
                           /*async_without_async_commit_wait=*/false);
    n->body = inject_result.stmt;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.cuda.transform.LowerPTXAsyncCopy",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.LowerPTXAsyncCopy",
                        LowerPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
