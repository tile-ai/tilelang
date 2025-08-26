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

class RegionOp : public TileOperator {
public:
  RegionOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();

  std::unique_ptr<TileOperator> Clone() const override {
    return std::make_unique<RegionOp>(*this);
  }

  const Buffer &GetBuffer() const { return buffer_; }
  const Array<Range> &GetRanges() const { return ranges_; }
  int GetAccessMask() const { return access_mask_; }
  bool IsFullRegion() const;

private:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;
};

Var GetVarFromAccessPtr(const PrimExpr &expr);

std::unique_ptr<TileOperator> ParseOperator(Call call, BufferMap vmap);
std::unique_ptr<TileOperator> ParseOperator(Stmt stmt, BufferMap vmap);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REGION_H_
