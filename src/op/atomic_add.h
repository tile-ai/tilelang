/*!
 * \file tl/op/atomic_add.h
 * \brief Define atomic add operator.
 *
 */

#ifndef TVM_TL_OP_ATOMIC_ADD_H_
#define TVM_TL_OP_ATOMIC_ADD_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

class AtomicAddNode : public TileOperatorNode {
public:
  Array<PrimExpr> args_;

  Buffer src, dst;
  Array<Range> src_range, dst_range;
  IntImm coalesced_width;

  mutable ParallelOp par_op_;
  static constexpr const char *_type_key = "tl.AtomicAdd";
  TVM_DECLARE_FINAL_OBJECT_INFO(AtomicAddNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

  static const Op &Get();
  TileOperator Clone() const;

protected:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  Array<IterVar> MakeIterVars() const;

  // ivs: itervars returned by MakeIterVars()
  // src_dst: 0 for src_indices, 1 for dst_indices
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;

  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;
};

class AtomicAdd : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(AtomicAdd, TileOperator, AtomicAddNode);
  TVM_DLL AtomicAdd(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ATOMIC_ADD_H_