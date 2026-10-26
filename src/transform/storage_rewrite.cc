#include "transform/tirx_passes.h"

#include "compat/visit.h"
#include "op/builtin.h"
#include "tirx_compat/transform.h"
#include "transform/common/pointer_value_type.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

bool HasStorageMetadata(const tirx::Function &function) {
  bool found = false;
  auto inspect = [&](ffi::AnyView metadata) {
    if (found)
      return;
    VisitMetadata(
        metadata,
        [&](ffi::AnyView value) -> ffi::Optional<ffi::VisitInterrupt> {
          if (auto expression = value.as<Expr>()) {
            tl::PostOrderVisit(*expression, [&](const ffi::ObjectRef &object) {
              if (const auto *variable = object.as<VarNode>())
                found |= variable->ty.as<tirx::TensorTypeNode>() != nullptr ||
                         variable->ty.as<PtrTypeNode>() != nullptr;
            });
          }
          return std::nullopt;
        });
  };
  inspect(function->attrs);
  if (function->body) {
    tl::PostOrderVisit(*function->body, [&](const ffi::ObjectRef &object) {
      inspect(ExpressionMetadata(object));
    });
  }
  return found;
}

} // namespace

tirx::Function RewritePointerValueTypes(tirx::Function function,
                                        bool storage_only) {
  bool rewrite_storage_types = !HasStorageMetadata(function);
  return tirx::PointerValueTypeRewrite(
      std::move(function), storage_only, !storage_only && rewrite_storage_types,
      !storage_only && rewrite_storage_types, rewrite_storage_types, true,
      rewrite_storage_types, !storage_only);
}

tvm::transform::Pass RewritePointerValueTypes() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    return RewritePointerValueTypes(std::move(function), false);
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.PointerValueTypeRewrite");
}

} // namespace tvm::tl

namespace tvm::tl::transform {

tvm::transform::Pass StorageRewrite() {
  auto pass = [](IRModule module, tvm::transform::PassContext context) {
    bool detect_inplace =
        context->GetConfig<bool>(kStorageRewriteDetectInplace).value_or(false);
    return tirx_passes::StorageRewrite(false, detect_inplace, true,
                                       detect_inplace)(module);
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.StorageRewrite");
}

tvm::transform::Pass PointerValueTypeRewrite() {
  return tl::RewritePointerValueTypes();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.StorageRewrite", StorageRewrite)
      .def("tl.transform.PointerValueTypeRewrite", PointerValueTypeRewrite);
}
} // namespace tvm::tl::transform
