/*!
 * \file tl/op/logical.cc
 * \brief Native pointer-based logical reductions and backend lowering.
 */
#include "native_intrinsic.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl {
namespace {

PrimExpr LowerLogicalReduction(PrimExpr expression, const char *symbol) {
  const Call call = expression.as_or_throw<Call>();
  const auto arguments = NativeIntrinsicOperands(call.get());
  TVM_FFI_CHECK_EQ(arguments.size(), 2U, ValueError);
  arguments[0]->ty.as_or_throw<PtrType>();
  const IntExpr count = arguments[1].as_or_throw<IntExpr>();
  TVM_FFI_CHECK(count.ty().IsScalar(), TypeError);
  TVM_FFI_CHECK(expression.ty() == PrimType::Bool(), TypeError);
  return Call(PrimType::Bool(), tirx::call_extern_op(),
              {StringImm(symbol), arguments[0], count}, {}, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

PrimExpr LowerAny(PrimExpr expression) {
  return LowerLogicalReduction(expression, "tl::Any");
}
PrimExpr LowerAll(PrimExpr expression) {
  return LowerLogicalReduction(expression, "tl::All");
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef any("tl.any_of");
  // These operations inspect pointee memory. Address expressions are pure, but
  // the result cannot be hoisted across a write to that memory.
  DefineNativeIntrinsic(any, 2, 2, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kPrimitive);
  any.set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerAny)
      .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic", LowerAny);
  OpDef all("tl.all_of");
  DefineNativeIntrinsic(all, 2, 2, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kPrimitive);
  all.set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerAll)
      .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic", LowerAll);
}
} // namespace tvm::tl
