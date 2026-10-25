#include "op/builtin.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl::transform {

tvm::transform::Pass StorageRewrite() {
  auto pass = [](IRModule module, tvm::transform::PassContext context) {
    bool detect_inplace =
        context->GetConfig<bool>(kStorageRewriteDetectInplace).value_or(false);
    return tirx::transform::StorageRewrite(false, detect_inplace, true,
                                           detect_inplace)(module);
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.StorageRewrite");
}

tvm::transform::Pass PointerValueTypeRewrite() {
  return tvm::transform::Sequential(
      {tirx::transform::PointerValueTypeRewrite()},
      "tl.PointerValueTypeRewrite");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.StorageRewrite", StorageRewrite)
      .def("tl.transform.PointerValueTypeRewrite", PointerValueTypeRewrite);
}
} // namespace tvm::tl::transform
