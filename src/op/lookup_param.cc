/*!
 * \file tl/op/lookup_param.cc
 * \brief Named C storage lookup retained by the TileLang language facade.
 */
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>

namespace tvm::tl {
namespace {

void ValidateLookupParam(const CallObj *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError);
  TVM_FFI_CHECK(call->args[0].as<StringImm>().has_value(), TypeError)
      << "lookup_param requires a literal parameter name";
  TVM_FFI_CHECK(call->ty_args.empty() && !call->attrs.has_value(), ValueError);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, PtrType::VoidPtrType()),
                TypeError)
      << "lookup_param returns an untyped pointer";
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.lookup_param")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateLookupParam>())
      .signature(sig::arg<StringImm>("param_name"))
      .set_attr(tvm::op_attr::kFixedReturnType, PtrType::VoidPtrType())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kUpdateState);
}

} // namespace tvm::tl
