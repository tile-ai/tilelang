/*!
 * \file tl/op/native_intrinsic.h
 * \brief Native result-type and annotation contracts for scalar intrinsics.
 */
#ifndef TILELANG_OP_NATIVE_INTRINSIC_H_
#define TILELANG_OP_NATIVE_INTRINSIC_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op/attr.h>

#include "ir/annotations.h"

namespace tvm::tl {

// Intrinsics with historical ignored handle/int results use kAnyScalar. Only
// operations whose producers promise a real void result use kVoid.
enum class NativeIntrinsicResult : int64_t {
  kAnyScalar,
  kPrimitive,
  kPointer,
  kVoid
};

inline constexpr ffi::reflection::AttrKey<int64_t> kNativeIntrinsicMinArgs{
    "tl.native_intrinsic_min_args"};
inline constexpr ffi::reflection::AttrKey<int64_t> kNativeIntrinsicMaxArgs{
    "tl.native_intrinsic_max_args"};
inline constexpr ffi::reflection::AttrKey<int64_t> kNativeIntrinsicResult{
    "tl.native_intrinsic_result"};
using NativeIntrinsicExtraValidator = ffi::TypedFunction<void(Call)>;
inline constexpr ffi::reflection::AttrKey<NativeIntrinsicExtraValidator>
    kNativeIntrinsicExtraValidator{"tl.native_intrinsic_extra_validator"};

inline ffi::Array<Expr> NativeIntrinsicOperands(const CallObj *call) {
  if (!call->attrs)
    return call->args;
  const auto *dict = call->attrs.value().as<DictAttrsObj>();
  if (!dict)
    return call->args;
  const auto attrs = ffi::GetRef<DictAttrs>(dict);
  const auto count = attrs.GetAttr<int64_t>("operand_count");
  const auto schema = attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema");
  if (!count && !schema)
    return call->args;
  TVM_FFI_CHECK(count && schema, ValueError)
      << "Intrinsic metadata requires operand_count and annotation_schema "
         "together";
  TVM_FFI_CHECK_EQ(attrs.GetAttr<int64_t>("schema_version").value_or(0), 1,
                   ValueError);
  TVM_FFI_CHECK_GE(count.value(), 0, ValueError);
  TVM_FFI_CHECK_EQ(static_cast<size_t>(count.value()) + 1, call->args.size(),
                   ValueError)
      << "Intrinsic metadata must occupy exactly one final Tuple operand";
  const auto payload = call->args.back().as_or_throw<tvm::Tuple>();
  // Resolve the schema as well: a valid Tuple with an out-of-range variable
  // reference must not slip through and disappear at code generation.
  DecodeAnnotations(payload, schema.value());
  ffi::Array<Expr> result;
  result.reserve(count.value());
  for (int64_t i = 0; i < count.value(); ++i)
    result.push_back(call->args[i]);
  return result;
}

inline Type InferNativeIntrinsicResult(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1, TypeError)
      << "TileLang scalar intrinsic accepts at most one result type argument";
  if (!call->ty_args.empty())
    return call->ty_args[0];
  // Native C++ callers and VectorizeLoop reconstruct calls with an explicit
  // new result type and no type-argument list. Keep this supported native form.
  if (!call->ty.as<MissingTypeObj>())
    return call->ty;
  auto kind =
      Op::GetAttrMap(kNativeIntrinsicResult)[call->op.as_or_throw<Op>()];
  TVM_FFI_CHECK(kind == static_cast<int64_t>(NativeIntrinsicResult::kVoid),
                TypeError)
      << "TileLang scalar intrinsic requires an explicit result type";
  return PrimType::Void();
}

inline void ValidateNativeIntrinsic(const CallObj *call) {
  const auto op = call->op.as_or_throw<Op>();
  const auto args = NativeIntrinsicOperands(call);
  const auto minimum = Op::GetAttrMap(kNativeIntrinsicMinArgs)[op];
  const auto maximum = Op::GetAttrMap(kNativeIntrinsicMaxArgs)[op];
  TVM_FFI_CHECK_GE(static_cast<int64_t>(args.size()), minimum, ValueError)
      << op->name << " has too few runtime operands";
  TVM_FFI_CHECK(maximum < 0 || static_cast<int64_t>(args.size()) <= maximum,
                ValueError)
      << op->name << " has too many runtime operands";
  Type result = InferNativeIntrinsicResult(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(result, call->ty), TypeError)
      << "Intrinsic explicit result type disagrees with its type argument";
  auto primitive = result.as<PrimType>();
  auto pointer = result.as<PtrType>();
  auto kind = static_cast<NativeIntrinsicResult>(
      Op::GetAttrMap(kNativeIntrinsicResult)[op]);
  bool valid = false;
  switch (kind) {
  case NativeIntrinsicResult::kAnyScalar:
    valid = primitive.has_value() || pointer.has_value();
    break;
  case NativeIntrinsicResult::kPrimitive:
    valid = primitive && !primitive.value().IsVoid();
    break;
  case NativeIntrinsicResult::kPointer:
    valid = pointer.has_value();
    break;
  case NativeIntrinsicResult::kVoid:
    valid = primitive && primitive.value().IsVoid();
    break;
  }
  TVM_FFI_CHECK(valid, TypeError)
      << "Invalid result category for " << op->name << ": " << result;
  if (Op::HasAttrMap(kNativeIntrinsicExtraValidator)) {
    const auto extra = Op::GetAttrMap(kNativeIntrinsicExtraValidator);
    if (extra.count(op))
      extra[op](ffi::GetRef<Call>(call));
  }
}

inline void DefineNativeIntrinsic(
    OpDef &def, int minimum_args, int maximum_args, CallEffectKind effect,
    NativeIntrinsicResult result = NativeIntrinsicResult::kAnyScalar,
    bool vectorizable = false) {
  TVM_FFI_CHECK(minimum_args >= 0 &&
                    (maximum_args < 0 || maximum_args >= minimum_args),
                ValueError);
  def.set_attr(kNativeIntrinsicMinArgs, static_cast<int64_t>(minimum_args))
      .set_attr(kNativeIntrinsicMaxArgs, static_cast<int64_t>(maximum_args))
      .set_attr(kNativeIntrinsicResult, static_cast<int64_t>(result))
      .set_attr(tvm::op_attr::kCallEffectKind, effect)
      .set_attr(tirx::op_attr::kVectorizable, vectorizable)
      .set_native_fn_attr<&InferNativeIntrinsicResult>(tvm::op_attr::kInferType)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateNativeIntrinsic>())
      .signature(
          sig::var_args<Expr>(
              "operands",
              "Runtime operands, optionally followed by an annotation Tuple."),
          sig::var_ty_args<Type>(
              "result_type",
              "Optional explicit result type; validated against Call.ty."));
}

inline void DefineNativeIntrinsic(
    OpDef &&def, int minimum_args, int maximum_args, CallEffectKind effect,
    NativeIntrinsicResult result = NativeIntrinsicResult::kAnyScalar,
    bool vectorizable = false) {
  DefineNativeIntrinsic(def, minimum_args, maximum_args, effect, result,
                        vectorizable);
}

} // namespace tvm::tl
#endif // TILELANG_OP_NATIVE_INTRINSIC_H_
