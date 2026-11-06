/*!
 * \file ascend/ir/block.h
 * \brief Ascend execution regions with directly accessible static metadata.
 */
#ifndef TILELANG_ASCEND_IR_BLOCK_H_
#define TILELANG_ASCEND_IR_BLOCK_H_

#include <tvm/ir/stmt.h>

namespace tvm::tl::ascend {

TVM_TIRX_DLL bool IsAscendBlockName(const ffi::String &name);
TVM_TIRX_DLL bool IsAscendBlock(const RegionStmtObj *region);
TVM_TIRX_DLL ffi::String AscendBlockName(const RegionStmtObj *region);

enum class VFKind { kNone, kSimd, kSimt };

// Recognize execution blocks and source VF scope annotations before source
// lowering. Ordinary regions remain outside a VF, regardless of variable names.
TVM_TIRX_DLL VFKind GetVFKind(const RegionStmtObj *region);

// Static metadata lives in attrs. Any expression-dependent annotation remains
// an explicit operand, so native substitution and use-def analysis see it.
// The decoded map is an analysis view; reconstruct with MakeAscendBlock.
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
AscendBlockAnnotations(const RegionStmtObj *region);

TVM_TIRX_DLL RegionStmt MakeAscendBlock(
    ffi::String name, const ffi::Map<ffi::String, ffi::Any> &annotations,
    SeqStmt body, Location loc = UnknownLoc());

} // namespace tvm::tl::ascend
#endif // TILELANG_ASCEND_IR_BLOCK_H_
