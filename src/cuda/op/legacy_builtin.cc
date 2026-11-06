/*!
 * \file tl/cuda/op/legacy_builtin.cc
 * \brief TileLang-owned contracts for source spellings absent from native TIRx.
 */

#include "builtin.h"
#include "cuda/op/builtin.h"
#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl {
namespace {

using PointerExpr = TypedExpr<PtrType>;

// These source spellings are statement intrinsics with a historical ignored
// handle/accumulator result. Native C++ producers use void.
Type InferLegacyStatement(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty()           ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>() ? call->ty
                                                       : Type(PrimType::Void());
  TVM_FFI_CHECK(result.as<PrimType>() || result.as<PtrType>(), TypeError);
  return result;
}

void ValidateLegacyResult(const CallObj *call) {
  TVM_FFI_CHECK(ffi::StructuralEqual()(InferLegacyStatement(call), call->ty),
                TypeError)
      << "Legacy statement result type disagrees with its type argument";
}

void ValidateBarrierInstruction(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  const auto op = call->op.as_or_throw<Op>();
  const bool remote =
      op.same_as(tl::ptx_arrive_barrier_expect_tx()) && args.size() == 4;
  TVM_FFI_CHECK(args.size() == op->args_info.size() || remote, ValueError)
      << "Legacy barrier instruction has the wrong number of runtime operands";
  for (const Expr &arg : args) {
    const auto primitive = arg->ty.as<PrimType>();
    TVM_FFI_CHECK(arg->ty.as<PtrType>() ||
                      (primitive && primitive->IsScalar() &&
                       primitive->MatchesCode(kDLInt, kDLUInt, kDLBool)),
                  TypeError)
        << "Barrier operands must be pointers or scalar integers";
  }
  if (op.same_as(tl::create_barriers())) {
    const auto *count = args[0].as<IntImmObj>();
    TVM_FFI_CHECK(count && count->value > 0 && count->value <= INT32_MAX,
                  ValueError)
        << "create_barriers requires a positive constant barrier count";
  } else if (op.same_as(tl::ptx_wait_group())) {
    const auto *count = args[0].as<IntImmObj>();
    TVM_FFI_CHECK(count && count->value >= 0 && count->value < 8, ValueError)
        << "ptx_wait_group requires an immediate in [0, 7]";
  }
  ValidateLegacyResult(call);
}

void ValidateAsyncBytes(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 3 || args.size() == 4, ValueError)
      << "ptx_cp_async_bytes expects three or four runtime operands";
  args[0].as_or_throw<PointerExpr>();
  args[1].as_or_throw<PointerExpr>();
  const auto *count = args[2].as<IntImmObj>();
  TVM_FFI_CHECK(
      count && (count->value == 4 || count->value == 8 || count->value == 16),
      ValueError)
      << "ptx_cp_async_bytes requires a constant byte width in {4, 8, 16}";
  if (args.size() == 4) {
    const auto dtype = args[3].as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(dtype.IsScalar() &&
                      dtype.MatchesCode(kDLBool, kDLInt, kDLUInt),
                  TypeError);
  }
  ValidateLegacyResult(call);
}

void ValidateBulkCopy(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 6U, ValueError)
      << "ptx_cp_async_bulk expects six runtime operands";
  args[0].as_or_throw<PointerExpr>();
  args[2].as_or_throw<PointerExpr>();
  for (int i : {1, 3, 4, 5})
    TVM_FFI_CHECK(args[i].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
  args[5].as_or_throw<IntImm>();
  ValidateLegacyResult(call);
}

void ValidateLegacySparseMMA(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 16U, ValueError)
      << "ptx_mma_sp expects sixteen runtime operands";
  for (int i = 0; i < 6; ++i)
    args[i].as_or_throw<StringImm>();
  for (int i : {6, 8, 10, 12})
    args[i].as_or_throw<PointerExpr>();
  for (int i : {7, 9, 11, 13, 14})
    TVM_FFI_CHECK(args[i].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
  const auto saturate = args[15].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(saturate.IsScalar() &&
                    saturate.MatchesCode(kDLBool, kDLInt, kDLUInt),
                TypeError);
  ValidateLegacyResult(call);
}

void ValidateLegacyLoad(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 4U, ValueError)
      << "ptx_ldg32 expects four runtime operands";
  args[0].as_or_throw<PointerExpr>();
  args[2].as_or_throw<TensorLoad>();
  TVM_FFI_CHECK(args[3].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
  const auto predicate = args[1].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(predicate.IsScalar() &&
                    predicate.MatchesCode(kDLBool, kDLInt, kDLUInt),
                TypeError);
  ValidateLegacyResult(call);
}

} // namespace

#define TL_LEGACY_CUDA_GETTER(Name)                                            \
  const Op &Name() {                                                           \
    static const Op op = Op::Get("tl." #Name);                                 \
    return op;                                                                 \
  }

TL_LEGACY_CUDA_GETTER(ptx_mma_sp)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_bulk)
TL_LEGACY_CUDA_GETTER(ptx_wait_barrier)
TL_LEGACY_CUDA_GETTER(create_barriers)
TL_LEGACY_CUDA_GETTER(ptx_arrive_barrier)
TL_LEGACY_CUDA_GETTER(ptx_init_barrier_thread_count)
TL_LEGACY_CUDA_GETTER(ptx_arrive_barrier_expect_tx)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_barrier)
TL_LEGACY_CUDA_GETTER(ptx_commit_group)
TL_LEGACY_CUDA_GETTER(ptx_wait_group)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_bytes)
TL_LEGACY_CUDA_GETTER(ptx_ldg32)

#undef TL_LEGACY_CUDA_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ptx_mma_sp")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateLegacySparseMMA>())
      .signature(
          sig::arg<StringImm>("shape"), sig::arg<StringImm>("a_layout"),
          sig::arg<StringImm>("b_layout"), sig::arg<StringImm>("a_dtype"),
          sig::arg<StringImm>("b_dtype"), sig::arg<StringImm>("c_dtype"),
          sig::arg<PointerExpr>("a"), sig::arg<IntExpr>("a_index"),
          sig::arg<PointerExpr>("b"), sig::arg<IntExpr>("b_index"),
          sig::arg<PointerExpr>("accumulator"), sig::arg<IntExpr>("c_index"),
          sig::arg<PointerExpr>("metadata"),
          sig::arg<IntExpr>("metadata_index"),
          sig::arg<IntExpr>("sparse_selector"), sig::arg<PrimExpr>("saturate"),
          sig::var_args<tvm::Tuple>("annotation_payload"),
          sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_cp_async_bulk")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateBulkCopy>())
      .signature(sig::arg<PointerExpr>("shared_pointer"),
                 sig::arg<IntExpr>("shared_offset"),
                 sig::arg<PointerExpr>("global_pointer"),
                 sig::arg<IntExpr>("global_offset"), sig::arg<IntExpr>("bytes"),
                 sig::arg<IntImm>("barrier_id"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_wait_barrier")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<Expr>("barrier"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.create_barriers")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<IntImm>("count"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_arrive_barrier")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<Expr>("barrier"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_init_barrier_thread_count")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<Expr>("barrier"), sig::arg<IntExpr>("thread_count"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_arrive_barrier_expect_tx")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<Expr>("barrier"), sig::arg<IntExpr>("bytes"),
                 sig::var_args<Expr>("optional_controls_and_annotations"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_cp_async_barrier")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<Expr>("barrier"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_commit_group")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_wait_group")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateBarrierInstruction>())
      .signature(sig::arg<IntImm>("pending_groups"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_cp_async_bytes")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateAsyncBytes>())
      .signature(sig::arg<PointerExpr>("destination"),
                 sig::arg<PointerExpr>("source"), sig::arg<IntImm>("bytes"),
                 sig::var_args<Expr>("optional_controls_and_annotations"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  OpDef("tl.ptx_ldg32")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateLegacyLoad>())
      .signature(sig::arg<PointerExpr>("registers"),
                 sig::arg<PrimExpr>("predicate"),
                 sig::arg<TensorLoad>("global_element"),
                 sig::arg<IntExpr>("register_index"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .set_native_fn_attr<&InferLegacyStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

} // namespace tvm::tl
