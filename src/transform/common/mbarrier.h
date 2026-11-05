#ifndef TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
#define TVM_TL_TRANSFORM_COMMON_MBARRIER_H_

#include "ir/tensor_utils.h"
#include <tvm/tirx/op/memory.h>

namespace tvm::tl {

inline tirx::TensorVar CreateMBarrierBuffer(const std::string &name,
                                            int count) {
  TVM_FFI_CHECK_GT(count, 0, ValueError);
  return tirx::TensorVar(
      name,
      tirx::TensorType({IntImm(PrimType::Int(32), count)}, PrimType::UInt(64),
                       runtime::MemoryScope::Get("shared.barrier")));
}

inline PrimExpr MakeBarrierRef(const tirx::TensorVar &buffer, PrimExpr index) {
  return TensorLoad(buffer, {std::move(index)});
}

inline const std::string injected_mbarrier_name_ = "mbarrier";

} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
