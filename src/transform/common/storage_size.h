/*!
 * \file tl/transform/common/storage_size.h
 * \brief Physical storage sizes for native TileLang tensor allocations.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
#define TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/type.h>

#include <cstdint>
#include <limits>

namespace tvm::tl {

inline bool IsPacked4BitStorage(const PrimType &dtype) {
  return dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) ||
         (dtype.bits() == 4 &&
          (dtype.code() == kDLInt || dtype.code() == kDLUInt));
}

inline int64_t GetBufferStorageSizeBytes(int64_t elements,
                                         const PrimType &dtype) {
  TVM_FFI_CHECK(elements >= 0 && !dtype.IsScalableVector(), ValueError)
      << "Storage size needs a nonnegative element count and fixed lanes";
  const int64_t units = IsPacked4BitStorage(dtype)
                            ? int64_t{dtype.bits()} * dtype.lanes()
                            : ((int64_t{dtype.bits()} + 7) / 8) * dtype.lanes();
  const int64_t rounding = IsPacked4BitStorage(dtype) ? 7 : 0;
  TVM_FFI_CHECK(
      units > 0 &&
          elements <= (std::numeric_limits<int64_t>::max() - rounding) / units,
      ValueError)
      << "Tensor physical storage size overflows int64";
  const int64_t total = elements * units;
  return IsPacked4BitStorage(dtype) ? (total + 7) / 8 : total;
}

inline PrimExpr GetBufferStorageSizeBytes(PrimExpr elements,
                                          const PrimType &dtype) {
  TVM_FFI_CHECK(!dtype.IsScalableVector(), ValueError)
      << "Storage size requires a fixed vector lane count";
  if (auto count = elements.as<IntImmObj>())
    return IntImm(
        PrimType::Int(64),
        GetBufferStorageSizeBytes(static_cast<int64_t>(count->value), dtype));
  if (elements.ty() != PrimType::Int(64))
    elements = prim::cast(elements, PrimType::Int(64));
  if (IsPacked4BitStorage(dtype)) {
    PrimExpr bits =
        elements * IntImm(elements.ty(), int64_t{dtype.bits()} * dtype.lanes());
    return prim::FloorDiv(bits + IntImm(bits.ty(), 7), IntImm(bits.ty(), 8));
  }
  return elements * IntImm(elements.ty(),
                           ((int64_t{dtype.bits()} + 7) / 8) * dtype.lanes());
}

inline PrimExpr GetBufferStorageSizeBytes(const tirx::TensorType &tensor) {
  TVM_FFI_CHECK(tensor->HasDefaultLayout(), ValueError)
      << "Physical storage sizing requires a flattened, compact allocation";
  PrimExpr elements = IntImm(PrimType::Int(64), 1);
  int64_t constant = 1;
  bool all_constant = true;
  for (const PrimExpr &extent : tensor->shape) {
    if (const auto *value = extent.as<IntImmObj>()) {
      TVM_FFI_CHECK_GE(value->value, 0, ValueError);
      TVM_FFI_CHECK(value->value == 0 ||
                        constant <= std::numeric_limits<int64_t>::max() /
                                        static_cast<int64_t>(value->value),
                    ValueError)
          << "Tensor element count overflows int64";
      constant *= static_cast<int64_t>(value->value);
    } else {
      all_constant = false;
    }
    PrimExpr dimension = extent;
    if (dimension.ty() != PrimType::Int(64))
      dimension = prim::cast(dimension, PrimType::Int(64));
    elements = elements * dimension;
  }
  if (all_constant)
    return IntImm(PrimType::Int(64),
                  GetBufferStorageSizeBytes(constant, tensor->dtype));
  return GetBufferStorageSizeBytes(elements, tensor->dtype);
}

inline PrimExpr GetBufferStorageSizeBytes(const tirx::TensorVar &tensor) {
  return GetBufferStorageSizeBytes(tensor.ty());
}

} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
