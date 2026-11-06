/*!
 * \file tl/rocm/op/builtin.cc
 * \brief Registration of ROCm-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op/attr.h>

namespace tvm {
namespace tl {

using namespace tirx;

const Op &ds_read_tr16_b64() {
  static const Op op = Op::Get("tl.ds_read_tr16_b64");
  return op;
}

const Op &ds_read_tr8_b64() {
  static const Op op = Op::Get("tl.ds_read_tr8_b64");
  return op;
}

const Op &tvm_mfma() {
  static const Op op = Op::Get("tl.tvm_mfma");
  return op;
}

const Op &tvm_mfma_store() {
  static const Op op = Op::Get("tl.tvm_mfma_store");
  return op;
}

const Op &tvm_rdna_wmma() {
  static const Op op = Op::Get("tl.tvm_rdna_wmma");
  return op;
}

const Op &tvm_rdna_wmma_store() {
  static const Op op = Op::Get("tl.tvm_rdna_wmma_store");
  return op;
}

namespace {
void ValidateTransposeRead(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "LDS transpose read expects one runtime operand";
  args[0]->ty.as_or_throw<PtrType>();
  TVM_FFI_CHECK_EQ(call->ty.as_or_throw<PrimType>(), PrimType::UInt(32, 2),
                   TypeError);
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK_EQ(call->ty_args[0].as_or_throw<PrimType>(),
                     call->ty.as_or_throw<PrimType>(), TypeError)
        << "LDS transpose read result type disagrees with its type argument";
}

// AMD matrix wrappers historically spell their ignored result as the
// accumulator dtype or handle; lowered native calls use void.
Type InferMatrixInstruction(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty()           ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>() ? call->ty
                                                       : Type(PrimType::Void());
  TVM_FFI_CHECK(result.as<PrimType>() || result.as<PtrType>(), TypeError);
  return result;
}

template <bool Store> void ValidateMatrixInstruction(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), Store ? 6U : 12U, ValueError)
      << "AMD matrix instruction has the wrong number of runtime operands";
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, InferMatrixInstruction(call)),
                TypeError)
      << "Matrix result type disagrees with its type argument";
  if constexpr (Store) {
    for (int i : {0, 1, 4, 5})
      TVM_FFI_CHECK(args[i].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
    args[2]->ty.as_or_throw<PtrType>();
    args[3]->ty.as_or_throw<PtrType>();
  } else {
    for (int i = 0; i < 6; ++i)
      args[i].as_or_throw<StringImm>();
    for (int i : {6, 8, 10}) {
      args[i]->ty.as_or_throw<PtrType>();
      TVM_FFI_CHECK(args[i + 1].as_or_throw<IntExpr>().ty().IsScalar(),
                    TypeError);
    }
  }
}
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name : {"tl.ds_read_tr16_b64", "tl.ds_read_tr8_b64"})
    OpDef(name)
        .set_validator(
            ffi::reflection::NativeFunctionView<void(
                const CallObj *)>::FromNative<&ValidateTransposeRead>())
        .signature(sig::arg<Expr>("shared_pointer"),
                   sig::var_args<tvm::Tuple>("annotation_payload"),
                   sig::var_ty_args<PrimType>("result_type"))
        .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32, 2))
        .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
        .set_attr(tirx::op_attr::kVectorizable, false);
  for (const char *name : {"tl.tvm_mfma", "tl.tvm_rdna_wmma"})
    OpDef(name)
        .set_validator(
            ffi::reflection::NativeFunctionView<void(const CallObj *)>::
                FromNative<&ValidateMatrixInstruction<false>>())
        .signature(
            sig::arg<StringImm>("shape"), sig::arg<StringImm>("a_layout"),
            sig::arg<StringImm>("b_layout"), sig::arg<StringImm>("a_dtype"),
            sig::arg<StringImm>("b_dtype"), sig::arg<StringImm>("c_dtype"),
            sig::arg<Expr>("a"), sig::arg<IntExpr>("a_index"),
            sig::arg<Expr>("b"), sig::arg<IntExpr>("b_index"),
            sig::arg<Expr>("c"), sig::arg<IntExpr>("c_index"),
            sig::var_args<tvm::Tuple>("annotation_payload"),
            sig::var_ty_args<Type>("statement_type"))
        .set_native_fn_attr<&InferMatrixInstruction>(tvm::op_attr::kInferType)
        .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
        .set_attr(tirx::op_attr::kVectorizable, false);
  for (const char *name : {"tl.tvm_mfma_store", "tl.tvm_rdna_wmma_store"})
    OpDef(name)
        .set_validator(
            ffi::reflection::NativeFunctionView<void(const CallObj *)>::
                FromNative<&ValidateMatrixInstruction<true>>())
        .signature(sig::arg<IntExpr>("m"), sig::arg<IntExpr>("n"),
                   sig::arg<Expr>("destination"), sig::arg<Expr>("source"),
                   sig::arg<IntExpr>("source_offset"),
                   sig::arg<IntExpr>("stride"),
                   sig::var_args<tvm::Tuple>("annotation_payload"),
                   sig::var_ty_args<Type>("statement_type"))
        .set_native_fn_attr<&InferMatrixInstruction>(tvm::op_attr::kInferType)
        .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
        .set_attr(tirx::op_attr::kVectorizable, false);
}

} // namespace tl
} // namespace tvm
