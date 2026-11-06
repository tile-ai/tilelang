/*!
 * \file tl/metal/op/builtin.cc
 * \brief Native Metal cooperative tensor intrinsic contracts.
 */
#include "builtin.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include "ir/annotations.h"

namespace tvm::tl {
namespace {

// Cooperative instructions mutate fragment storage. Native rewrites use void;
// source wrappers also carry an ignored primitive or handle result.
Type InferCooperativeInstruction(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty()           ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>() ? call->ty
                                                       : Type(PrimType::Void());
  TVM_FFI_CHECK(result.as<PrimType>() || result.as<PtrType>(), TypeError)
      << "Metal cooperative instruction requires a primitive or pointer result";
  return result;
}

void ValidateCooperativeResult(const CallObj *call) {
  TVM_FFI_CHECK(
      ffi::StructuralEqual()(InferCooperativeInstruction(call), call->ty),
      TypeError)
      << "Cooperative result type disagrees with its type argument";
}

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

void ValidateFill(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 5U, ValueError)
      << "Metal cooperative instruction has the wrong number of runtime "
         "operands";
  ValidateCooperativeResult(call);
  ValidateFragment(args[0]);
  ValidateIndex(args[1]);
  args[2].as_or_throw<PrimExpr>();
  ValidateDimension(args[3]);
  ValidateDimension(args[4]);
}

void ValidateTransfer(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 11U, ValueError)
      << "Metal cooperative instruction has the wrong number of runtime "
         "operands";
  ValidateCooperativeResult(call);
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

void ValidateMMA(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 13U, ValueError)
      << "Metal cooperative instruction has the wrong number of runtime "
         "operands";
  ValidateCooperativeResult(call);
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
  OpDef("tl.cooperative_tensor_fill")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateFill>())
      .signature(sig::arg<Expr>("fragment"),
                 sig::arg<IntExpr>("fragment_index"),
                 sig::arg<PrimExpr>("value"), sig::arg<IntImm>("rows"),
                 sig::arg<IntImm>("columns"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferCooperativeInstruction>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  for (const char *name :
       {"tl.cooperative_tensor_load", "tl.cooperative_tensor_store"}) {
    OpDef(name)
        .set_validator(ffi::reflection::NativeFunctionView<void(
                           const CallObj *)>::FromNative<&ValidateTransfer>())
        .signature(
            sig::arg<Expr>("fragment"), sig::arg<IntExpr>("fragment_index"),
            sig::arg<Expr>("pointer"), sig::arg<IntExpr>("stride"),
            sig::arg<IntImm>("rows"), sig::arg<IntImm>("columns"),
            sig::arg<PrimExpr>("transpose"), sig::arg<IntImm>("simd_groups"),
            sig::arg<IntImm>("simd_rows"), sig::arg<IntImm>("simd_columns"),
            sig::arg<IntExpr>("simd_group_index"),
            sig::var_args<tvm::Tuple>("annotation_payload"),
            sig::var_ty_args<Type>("statement_type"))
        .set_native_fn_attr<&InferCooperativeInstruction>(
            tvm::op_attr::kInferType)
        .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
        .set_attr(tirx::op_attr::kVectorizable, false);
  }
  OpDef("tl.cooperative_tensor_multiply_accumulate")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateMMA>())
      .signature(sig::arg<Expr>("d"), sig::arg<IntExpr>("d_index"),
                 sig::arg<Expr>("a"), sig::arg<IntExpr>("a_index"),
                 sig::arg<Expr>("b"), sig::arg<IntExpr>("b_index"),
                 sig::arg<Expr>("c"), sig::arg<IntExpr>("c_index"),
                 sig::arg<IntImm>("m"), sig::arg<IntImm>("n"),
                 sig::arg<IntImm>("k"), sig::arg<PrimExpr>("transpose_a"),
                 sig::arg<PrimExpr>("transpose_b"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferCooperativeInstruction>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}
} // namespace tvm::tl
