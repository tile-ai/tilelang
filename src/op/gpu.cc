#include "gpu.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {

const Op &StorageSyncOp() {
  static const Op operation = Op::Get("tl.gpu_storage_sync");
  return operation;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.gpu_storage_sync")
      .signature(sig::arg<StringImm>("storage_scope", "The storage scope."),
                 sig::var_args<IntExpr>("participants",
                                        "Warp mask or named barrier operands."))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

} // namespace tvm::tl
