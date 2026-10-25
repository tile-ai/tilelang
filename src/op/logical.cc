/*!
 * \file tl/op/logical.cc
 * \brief Logical operations.
 *
 */

#include "builtin_registry.h"
#include "support/check.h"
#include <tvm/tirx/op/abi.h>

namespace tvm::tl {

PrimExpr LowerLogicalReduction(PrimExpr expression, const char *symbol) {
  const CallNode *call = expression.as<CallNode>();
  ICHECK(call);
  ICHECK_EQ(call->args.size(), 2);
  TVM_FFI_CHECK(call->args[0]->ty.as<PtrTypeNode>(), TypeError)
      << "Logical reduction requires a typed tensor pointer";
  TVM_FFI_CHECK(call->args[1]->ty.as<PrimTypeNode>(), TypeError)
      << "Logical reduction requires a scalar element count";
  return Call(PrimType::Bool(), tirx::call_extern_op(),
              {StringImm(symbol), call->args[0], call->args[1]}, std::nullopt,
              {}, call->loc)
      .as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name : {"tl.any_of", "tl.all_of"}) {
    RegisterBuiltin<2>(name, CallEffectKind::kReadState);
    OpDef(name).set_attr(tvm::op_attr::kFixedReturnType, PrimType::Bool());
  }
  for (const char *backend : {"cuda", "hip", "rocm"}) {
    OpDef("tl.any_of")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", [](PrimExpr expression) {
              return LowerLogicalReduction(expression, "tl::Any");
            });
    OpDef("tl.all_of")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", [](PrimExpr expression) {
              return LowerLogicalReduction(expression, "tl::All");
            });
  }
}

} // namespace tvm::tl
