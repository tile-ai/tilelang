/*!
 * \file tl/transform/persist_threadblock.cc
 * \brief Persist thread blocks with cooperative groups.
 */

#include "op/native_utils.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kUseCooperativeGroups = "use_cooperative_groups";
} // namespace attr

tvm::transform::Pass PersistThreadblock() {
  auto pass_func = [](tirx::Function function, IRModule,
                      tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    bool has_sync_grid = false;
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
      if (const auto *call = object.as<CallObj>())
        has_sync_grid |= call->op.same_as(sync_grid());
    });
    // The instruction is emitted by TileLang CUDA codegen. This pass records
    // its launch requirement; run it before the host launch ABI is fixed.
    if (has_sync_grid)
      function = WithAttr(function, attr::kUseCooperativeGroups, int64_t{1});
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0,
                                             "tl.PersistThreadblock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.PersistThreadblock",
                        PersistThreadblock);
}

} // namespace tl
} // namespace tvm
