#ifndef TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
#define TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_

#include "compat/expr.h"
#include <tvm/tirx/expr.h>

#include <cstdint>
#include <limits>

namespace tvm::tl {

inline bool IsPacked4BitStorage(PrimType type) {
  return type.bits() == 4 &&
         type.MatchesCode(kDLFloat4_e2m1fn, kDLInt, kDLUInt);
}

inline PrimExpr GetBufferStorageSizeBytes(const PrimExpr &elements,
                                          PrimType type) {
  TVM_FFI_CHECK(!type.IsScalableVector(), ValueError)
      << "Shared allocations require fixed-width elements";
  PrimExpr count = Cast(PrimType::Int(64), elements);
  int64_t bits = static_cast<int64_t>(type.bits()) * type.lanes();
  if (IsPacked4BitStorage(type))
    return FloorDiv(count * IntImm::Int64(bits) + 7, IntImm::Int64(8));
  return count * IntImm::Int64(type.StorageBytes());
}

inline int64_t GetBufferStorageSizeBytes(int64_t elements, PrimType type) {
  TVM_FFI_CHECK(elements >= 0, ValueError) << "Storage size cannot be negative";
  int64_t bits = static_cast<int64_t>(type.bits()) * type.lanes();
  int64_t scale = IsPacked4BitStorage(type) ? bits : type.StorageBytes();
  TVM_FFI_CHECK(scale > 0 &&
                    elements <=
                        (std::numeric_limits<int64_t>::max() - 7) / scale,
                ValueError)
      << "Storage size overflows int64";
  return IsPacked4BitStorage(type) ? (elements * bits + 7) / 8
                                   : elements * scale;
}

inline PrimExpr GetBufferStorageSizeBytes(const tirx::TensorVar &tensor) {
  PrimExpr elements = IntImm::Int64(1);
  if (auto layout = tensor.ty()->layout) {
    elements =
        Cast(PrimType::Int(64), layout.value().GetSpan(ffi::String("m")));
  } else {
    for (const PrimExpr &extent : tensor.shape())
      elements = elements * Cast(PrimType::Int(64), extent);
  }
  return GetBufferStorageSizeBytes(elements, tensor.dtype());
}

} // namespace tvm::tl

#endif
