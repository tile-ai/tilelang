/*!
 * \file tl/op/utils.h
 * \brief Common utilities for TL ops.
 */

#ifndef TVM_TL_OP_UTILS_H_
#define TVM_TL_OP_UTILS_H_

#include "dtype.h"
#include "operator.h"
#include "support/check.h"
#include <tvm/tirx/op/memory.h>

namespace tvm::tl {

TILELANG_DLL bool IsZeroBitPattern(const PrimExpr &expression);
TILELANG_DLL int to_CUtensorMapDataType(PrimType type);

template <typename Element>
ffi::Array<Element> ReverseArray(ffi::Array<Element> array) {
  return ffi::Array<Element>{array.rbegin(), array.rend()};
}

TILELANG_DLL bool IsBufferLikeExpr(const Expr &expression);
TILELANG_DLL TensorRegion NormalizeToBufferRegion(const Expr &expression);
TILELANG_DLL AccessRegion NormalizeToAccessRegion(
    const Expr &expression, int default_access_mask = kAccessReadWrite);
TILELANG_DLL Expr MakeAccessPtrFromRegion(const TensorRegion &region,
                                          int rw_mask, bool require_2d = false);
TILELANG_DLL Expr MakeAccessPtrFromBufferLoad(const TensorLoad &load,
                                              int rw_mask);
TILELANG_DLL Expr LowerAccessPtrCall(const Call &call);

inline bool IsFragmentBuffer(const tirx::TensorVar &tensor) {
  return tensor.scope().name() == "local.fragment";
}

inline Layout ExpandLayoutToMatchBuffer(const Layout &layout,
                                        const tirx::TensorVar &tensor) {
  if (!layout.defined() || tensor.shape().size() <= layout->InputDim()) {
    return layout;
  }
  ffi::Array<PrimExpr> leading_shape;
  for (size_t axis = 0; axis < tensor.shape().size() - layout->InputDim();
       ++axis) {
    leading_shape.push_back(tensor.shape()[axis]);
  }
  return layout->Expand(leading_shape);
}

inline bool IsSharedBuffer(const tirx::TensorVar &tensor,
                           bool allow_dynamic = true) {
  return tensor.scope().name() == "shared" ||
         (allow_dynamic && tensor.scope().name() == "shared.dyn");
}

inline bool IsTmemBuffer(const tirx::TensorVar &tensor) {
  return tensor.scope().name() == "shared.tmem";
}

inline bool IsGlobalBuffer(const tirx::TensorVar &tensor) {
  return tensor.scope().name() == "global";
}

inline bool IsValidCPAsyncTransferBytes(int bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

inline bool IsLocalBuffer(const tirx::TensorVar &tensor,
                          bool allow_var = false) {
  return tensor.scope().name() == "local" ||
         (allow_var && tensor.scope().name() == "local.var");
}

inline bool IsLocalVarBuffer(const tirx::TensorVar &tensor) {
  return tensor.scope().name() == "local.var";
}

inline bool IsFP4PackedToUnpackedStorageCopy(PrimType global_type,
                                             PrimType shared_type) {
  return global_type.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
         IsFP4UnpackedType(shared_type);
}

inline bool IsValidTMALoadDtypePair(PrimType global_type,
                                    PrimType shared_type) {
  if (IsFP4UnpackedType(global_type) || IsFP4UnpackedType(shared_type)) {
    return IsFP4PackedToUnpackedStorageCopy(global_type, shared_type);
  }
  return global_type == shared_type;
}

inline bool IsValidTMAStoreDtypePair(PrimType global_type,
                                     PrimType shared_type) {
  return global_type == shared_type && !IsFP4UnpackedType(shared_type);
}

inline bool IsValidTMADtypePair(bool is_load, PrimType global_type,
                                PrimType shared_type) {
  return is_load ? IsValidTMALoadDtypePair(global_type, shared_type)
                 : IsValidTMAStoreDtypePair(global_type, shared_type);
}

inline bool IsValidTMACopyDtypePair(PrimType global_type,
                                    PrimType shared_type) {
  return IsValidTMALoadDtypePair(global_type, shared_type);
}

inline bool IsFP4UnpackLoad(const tirx::TensorVar &source,
                            const tirx::TensorVar &destination) {
  return IsGlobalBuffer(source) && IsSharedBuffer(destination) &&
         IsFP4PackedToUnpackedStorageCopy(source.dtype(), destination.dtype());
}

} // namespace tvm::tl

#endif // TVM_TL_OP_UTILS_H_
