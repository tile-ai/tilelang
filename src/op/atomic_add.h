/*!
 * \file tl/op/atomic_add.h
 * \brief Define atomic add operator.
 *
 */

#ifndef TVM_TL_OP_ATOMIC_ADD_H_
#define TVM_TL_OP_ATOMIC_ADD_H_

#include "op.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

class AtomicAdd : public Operator {
public:
  AtomicAdd(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;

  static const Op &Get();

protected:
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

  std::unique_ptr<ParallelOp> par_op_;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ATOMIC_ADD_H_