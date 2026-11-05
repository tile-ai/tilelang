/*!
 * \file tl/metal/op/builtin.cc
 * \brief Native Metal cooperative tensor intrinsic contracts.
 */
#include "builtin.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include "op/native_intrinsic.h"

namespace tvm::tl {
namespace {

void ValidateIndex(const Expr &expr) {
  const auto value = expr.as_or_throw<IntExpr>();
  TVM_FFI_CHECK(value.ty().IsScalar(), TypeError)
      << "Metal fragment index must be a scalar integer";
}

void ValidateDimension(const Expr &expr) {
  const auto value = expr.as_or_throw<IntImm>()->value.as<int>();
  TVM_FFI_CHECK(value && *value > 0, ValueError)
      << "Metal cooperative tensor dimensions must be positive static integers";
}

void ValidateFragment(const Expr &expr) {
  Expr storage = expr;
  if (const auto *call = expr.as<CallObj>()) {
    TVM_FFI_CHECK(call->op.same_as(tirx::tensor_data_ptr_op()) &&
                      call->args.size() == 1 &&
                      call->args[0]->ty.as<tirx::TensorTypeObj>(),
                  TypeError)
        << "Metal fragment operand must name storage or its tensor_data_ptr";
    storage = call->args[0];
  }
  TVM_FFI_CHECK(storage.as<VarObj>(), TypeError)
      << "Metal fragment operand must retain its named allocation identity";
  const Type type = storage->ty;
  if (const auto *tensor = type.as<tirx::TensorTypeObj>()) {
    TVM_FFI_CHECK_EQ(tensor->scope.name(), "metal.cooperative_tensor",
                     TypeError);
  } else {
    const auto pointer = type.as_or_throw<PtrType>();
    TVM_FFI_CHECK_EQ(pointer->scope.name(), "metal.cooperative_tensor",
                     TypeError);
    pointer->element_type.as_or_throw<PrimType>();
  }
}

void ValidateFill(Call call) {
  const auto args = NativeIntrinsicOperands(call.get());
  ValidateFragment(args[0]);
  ValidateIndex(args[1]);
  args[2].as_or_throw<PrimExpr>();
  ValidateDimension(args[3]);
  ValidateDimension(args[4]);
}

void ValidateTransfer(Call call) {
  const auto args = NativeIntrinsicOperands(call.get());
  ValidateFragment(args[0]);
  ValidateIndex(args[1]);
  const auto pointer = args[2]->ty.as_or_throw<PtrType>();
  TVM_FFI_CHECK(!pointer->element_type.as_or_throw<PrimType>().IsVoid(),
                TypeError)
      << "Metal transfer pointer requires an element type";
  ValidateIndex(args[3]);
  ValidateDimension(args[4]);
  ValidateDimension(args[5]);
  TVM_FFI_CHECK(args[6].as_or_throw<PrimExpr>().ty().IsScalar(), TypeError);
  for (int i = 7; i < 10; ++i)
    ValidateDimension(args[i]);
  ValidateIndex(args[10]);
}

void ValidateMMA(Call call) {
  const auto args = NativeIntrinsicOperands(call.get());
  for (int i = 0; i < 8; i += 2) {
    ValidateFragment(args[i]);
    ValidateIndex(args[i + 1]);
  }
  for (int i = 8; i < 11; ++i)
    ValidateDimension(args[i]);
  TVM_FFI_CHECK(args[11].as_or_throw<PrimExpr>().ty().IsScalar() &&
                    args[12].as_or_throw<PrimExpr>().ty().IsScalar(),
                TypeError);
}

} // namespace

const Op &cooperative_tensor_fill() {
  static const Op op = Op::Get("tl.cooperative_tensor_fill");
  return op;
}
const Op &cooperative_tensor_load() {
  static const Op op = Op::Get("tl.cooperative_tensor_load");
  return op;
}
const Op &cooperative_tensor_store() {
  static const Op op = Op::Get("tl.cooperative_tensor_store");
  return op;
}
const Op &cooperative_tensor_multiply_accumulate() {
  static const Op op = Op::Get("tl.cooperative_tensor_multiply_accumulate");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  {
    OpDef def("tl.cooperative_tensor_fill");
    DefineNativeIntrinsic(def, 5, 5, CallEffectKind::kOpaque);
    def.set_attr(kNativeIntrinsicExtraValidator,
                 NativeIntrinsicExtraValidator(ValidateFill));
  }
  for (const char *name :
       {"tl.cooperative_tensor_load", "tl.cooperative_tensor_store"}) {
    OpDef def(name);
    DefineNativeIntrinsic(def, 11, 11, CallEffectKind::kOpaque);
    def.set_attr(kNativeIntrinsicExtraValidator,
                 NativeIntrinsicExtraValidator(ValidateTransfer));
  }
  {
    OpDef def("tl.cooperative_tensor_multiply_accumulate");
    DefineNativeIntrinsic(def, 13, 13, CallEffectKind::kOpaque);
    def.set_attr(kNativeIntrinsicExtraValidator,
                 NativeIntrinsicExtraValidator(ValidateMMA));
  }
}
} // namespace tvm::tl
