/*!
 * \file tl/rocm/op/builtin.cc
 * \brief Registration of ROCm-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "op/native_intrinsic.h"

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

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineNativeIntrinsic(OpDef("tl.ds_read_tr16_b64"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ds_read_tr8_b64"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tvm_mfma"), 12, 12, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tvm_mfma_store"), 6, 6,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tvm_rdna_wmma"), 12, 12,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tvm_rdna_wmma_store"), 6, 6,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
}

} // namespace tl
} // namespace tvm
