/*!
 * \file tl/ir/dtype.h
 * \brief TileLang storage types expressed through the native DLPack type
 * contract.
 */

#ifndef TILELANG_IR_DTYPE_H_
#define TILELANG_IR_DTYPE_H_

#include <tvm/ir/type.h>

namespace tvm::tl {

// Keep the type codes used by TileLang artifacts.  TF32 uses float32 storage;
// unpacked FP4 is an eight-bit shared-memory storage tag, not an arithmetic
// type.
inline constexpr int kTensorFloat32 = 130;
inline constexpr int kFloat4E2M1Unpacked = 131;

inline bool IsTensorFloat32(const PrimType &type) {
  return static_cast<int>(type.code()) == kTensorFloat32 && type.bits() == 32;
}

inline bool IsFloat4E2M1Unpacked(const PrimType &type) {
  return static_cast<int>(type.code()) == kFloat4E2M1Unpacked &&
         type.bits() == 8;
}

} // namespace tvm::tl

#endif // TILELANG_IR_DTYPE_H_
