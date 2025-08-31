/*!
 * \file tl/op/atomic_add.h
 * \brief Define atomic add operator.
 *
 */

#ifndef TVM_TL_OP_ATOMIC_ADD_H_
#define TVM_TL_OP_ATOMIC_ADD_H_

#include "operator.h"
#include "parallel.h"

/**
 * Lower this tile operator into a TIR statement for the given lowering context.
 *
 * @param T Lowering context containing mapped buffers and iteration
 * information.
 * @param analyzer Arithmetic analyzer used to simplify and reason about
 * expressions.
 * @return A TIR Stmt that implements the atomic-add tile operation for the
 * provided context.
 */
/**
 * Infer memory/layout mapping for tensors and buffers used by this operator.
 *
 * @param T Layout inference context providing buffer and shape information.
 * @param level Inference aggressiveness level; higher levels may perform more
 * speculative decisions.
 * @return A LayoutMap describing inferred layouts for the operator's inputs and
 * outputs.
 */
/**
 * Get the Op registration that identifies this tile operator.
 *
 * @return A reference to the registered Op representing this operator.
 */
/**
 * Create a deep copy of this tile operator node wrapped as a TileOperator.
 *
 * @return A TileOperator handle owning a cloned AtomicAddNode.
 */
/**
 * Construct a SIMT-style For loop nest (thread/block mapping) appropriate for
 * the operator.
 *
 * @param analyzer Arithmetic analyzer used to simplify loop bounds and
 * predicates.
 * @return A For loop node representing the SIMT-parallel loop structure.
 */
/**
 * Create iteration variables used by this operator's loop nest.
 *
 * @return An array of IterVar objects describing the loop iteration axes.
 */
/**
 * Produce index expressions for either source or destination buffer access
 * based on iteration vars.
 *
 * @param ivs IterVars created by MakeIterVars().
 * @param src_dst Selects which indices to produce: 0 for source indices, 1 for
 * destination indices.
 * @return An array of PrimExpr index expressions suitable for indexing the
 * selected buffer.
 */
/**
 * Build a predicate expression that guards out-of-bounds or conditional
 * accesses for src or dst.
 *
 * @param analyzer Arithmetic analyzer used to simplify the predicate.
 * @param ivs IterVars created by MakeIterVars().
 * @param extents The loop extents corresponding to the itervars.
 * @param src_dst Selects which side the predicate is for: 0 for source, 1 for
 * destination.
 * @return A PrimExpr boolean predicate that evaluates to true for valid
 * iterations.
 */
/**
 * Construct an AtomicAdd tile operator from operation arguments and a buffer
 * mapping.
 *
 * @param args Operation arguments (e.g., values or indices) specific to the
 * atomic-add semantics.
 * @param vmap Mapping from buffer names to Buffer objects used by this
 * operator.
 */
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