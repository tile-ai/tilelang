/*!
 * \file tl/op/logical.cc
 * \brief Native pointer-based logical reductions and backend lowering.
 */
#include "ir/annotations.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl {
namespace {

void ValidateLogicalReduction(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
      << "Logical reduction expects pointer and count runtime operands";
  args[0]->ty.as_or_throw<PtrType>();
  TVM_FFI_CHECK(args[1].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
  TVM_FFI_CHECK_EQ(call->ty.as_or_throw<PrimType>(), PrimType::Bool(),
                   TypeError);
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK_EQ(call->ty_args[0].as_or_throw<PrimType>(), PrimType::Bool(),
                     TypeError)
        << "Logical reduction result type disagrees with its type argument";
}

void DefineLogicalReduction(OpDef &def) {
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(
             const CallObj *)>::FromNative<&ValidateLogicalReduction>())
      .signature(sig::arg<Expr>("pointer"), sig::arg<IntExpr>("count"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Bool())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

PrimExpr LowerLogicalReduction(PrimExpr expression, const char *symbol) {
  const Call call = expression.as_or_throw<Call>();
  const auto arguments = DecodeCallOperands(call.get());
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
  DefineLogicalReduction(any);
  any.set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerAny)
      .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic", LowerAny);
  OpDef all("tl.all_of");
  DefineLogicalReduction(all);
  all.set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerAll)
      .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic", LowerAll);
}
} // namespace tvm::tl
