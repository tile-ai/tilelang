/*!
 * \file tl/op/elem.h
 * \brief Define elment-wise operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "op.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

class Copy : public Operator {
public:
  Copy(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;

  static const Op &Get();

  Copy(const Copy &other)
      : args_(other.args_), src(other.src), dst(other.dst),
        src_range(other.src_range), dst_range(other.dst_range),
        coalesced_width(other.coalesced_width), disable_tma(other.disable_tma) {
    // No clone nullptr
    if (other.par_op_)
      par_op_ = std::unique_ptr<ParallelOp>(
          static_cast<ParallelOp *>(other.par_op_->Clone().release()));
  }
  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Copy>(*this);
  }

protected:
  Stmt LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer) const;
  Stmt LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer) const;

  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  Array<IterVar> MakeIterVars() const;

  // ivs: itervars returned by MakeIterVars()
  // src_dst: 0 for src_indices, 1 for dst_indices
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;

  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;

  Array<PrimExpr> args_;

  Buffer src, dst;
  Array<Range> src_range, dst_range;
  IntImm coalesced_width;
  Bool disable_tma = Bool(false);

  std::unique_ptr<ParallelOp> par_op_;
};

class Fill : public Operator {
public:
  Fill(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Fill>(*this);
  }

private:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  tir::Buffer dst;
  PrimExpr value;
  Array<Range> region;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_