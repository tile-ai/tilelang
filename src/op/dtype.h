#ifndef TILELANG_OP_DTYPE_H_
#define TILELANG_OP_DTYPE_H_

#include <tvm/ir/type.h>

namespace tvm::tl {

inline constexpr DLDataTypeCode kTensorFloat32 =
    static_cast<DLDataTypeCode>(130);
inline constexpr DLDataTypeCode kFloat4E2M1Unpacked =
    static_cast<DLDataTypeCode>(131);

inline bool IsTensorFloat32(PrimType type) {
  return type.MatchesElementType(kTensorFloat32, 32);
}

inline bool IsFP4UnpackedType(PrimType type) {
  return type.MatchesElementType(kFloat4E2M1Unpacked, 8);
}

} // namespace tvm::tl

#endif
