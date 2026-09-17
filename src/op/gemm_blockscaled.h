/*!
 * \file tl/op/gemm_blockscaled.h
 * \brief Define the block-scaled GEMM tile operator.
 */

#ifndef TVM_TL_OP_GEMM_BLOCKSCALED_H_
#define TVM_TL_OP_GEMM_BLOCKSCALED_H_

#include "gemm.h"

namespace tvm {
namespace tl {

/*!
 * \brief Block-scaled GEMM: C (+)= (A * SFA) @ (B * SFB).
 *
 * Owns the two scale-factor operands and the logical K-axis start offset.
 * Inherits operand layouts, warp partition and scheduling from GemmNode so
 * passes that only need GEMM semantics can keep matching the base type.
 */
class GemmBlockScaledNode : public GemmNode {
public:
  tirx::BufferRegion sfaRegion_, sfbRegion_;
  PrimExpr sfKStart_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmBlockScaled", GemmBlockScaledNode,
                                    GemmNode);

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<GemmBlockScaledNode>()
        .def_ro("sfaRegion", &GemmBlockScaledNode::sfaRegion_)
        .def_ro("sfbRegion", &GemmBlockScaledNode::sfbRegion_)
        .def_ro("sfKStart", &GemmBlockScaledNode::sfKStart_);
  }

  AccessRegions GetAccessRegions() const override;
  ffi::Array<tirx::BufferRegion> GetReadBeforeWriteRegions() const override;
  TileOperator Clone() const override;

  // Calls the backend's dedicated block-scaled instruction selector, or
  // rejects the operation if that selector is not registered.
  ffi::String GetGemmInstructionKey(int block_size,
                                    Target target) const override;

protected:
  const char *InferLayoutGlobalFunc() const override {
    return "tl.gemm_blockscaled.infer_layout";
  }
  const char *LowerGlobalFunc() const override {
    return "tl.gemm_blockscaled.lower";
  }
};

/*! \brief Block-scaled GEMM with explicit scale-factor operands. */
class GemmBlockScaled : public Gemm {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmBlockScaled, Gemm,
                                             GemmBlockScaledNode);
  TVM_DLL
  GemmBlockScaled(ffi::Array<PrimExpr> args,
                  ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

/*! \brief Inspect scale operands when handling a generic GEMM node. */
inline const GemmBlockScaledNode *AsGemmBlockScaled(const GemmNode &op) {
  return op.IsInstance<GemmBlockScaledNode>()
             ? static_cast<const GemmBlockScaledNode *>(&op)
             : nullptr;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_GEMM_BLOCKSCALED_H_
