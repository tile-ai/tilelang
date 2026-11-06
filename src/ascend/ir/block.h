/*!
 * \file ascend/ir/block.h
 * \brief Ascend execution domains over native TIRx statements.
 */
#ifndef TILELANG_ASCEND_IR_BLOCK_H_
#define TILELANG_ASCEND_IR_BLOCK_H_

#include <tvm/ir/stmt.h>

namespace tvm::tl::ascend {

TVM_TIRX_DLL const Op &ascend_block_op();

inline bool IsAscendBlock(const RegionStmtObj *region) {
  return region && region->op.same_as(ascend_block_op());
}

enum class VFKind { kNone, kSimd, kSimt };

inline VFKind GetVFKind(const RegionStmtObj *region) {
  if (!IsAscendBlock(region))
    return VFKind::kNone;
  const auto name = region->attrs.GetAttr<ffi::String>("name").value();
  return name == "SIMD_VF"   ? VFKind::kSimd
         : name == "SIMT_VF" ? VFKind::kSimt
                             : VFKind::kNone;
}

} // namespace tvm::tl::ascend
#endif // TILELANG_ASCEND_IR_BLOCK_H_
