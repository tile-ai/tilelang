/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_REGION_H_
#define TVM_TL_OP_REGION_H_

#include "./operator.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>

namespace tvm {
namespace tl {

using namespace tir;

class RegionOpNode : public TileOperatorNode {
public:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;

  static constexpr const char *_type_key = "tl.RegionOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegionOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  const Buffer &GetBuffer() const { return buffer_; }
  const Array<Range> &GetRanges() const { return ranges_; }
  int GetAccessMask() const { return access_mask_; }
  bool IsFullRegion() const;

  TileOperator Clone() const;
};

class RegionOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(RegionOp, TileOperator, RegionOpNode);
  TVM_DLL RegionOp(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REGION_H_
