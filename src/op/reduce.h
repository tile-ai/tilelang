/*!
 * \file tl/op/reduce.h
 * \brief Define reduce operator.
 *
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "operator.h"

namespace tvm {
/**
 * Tile operator node that performs a reduction (sum, max, min, etc.) along a
 * single tensor dimension.
 *
 * Represents a per-instance reduce operator with explicit source/destination
 * buffers, target dimension, reduction type, and a flag controlling whether the
 * destination is cleared before reduction.
 */

/**
 * Lower this ReduceOpNode into a Tir Stmt suitable for code generation.
 *
 * Produces the TIR statement(s) that implement the configured reduction.
 *
 * @return A TIR `Stmt` implementing the reduce operation.
 */

/**
 * Infer input/output layouts for this reduce operator.
 *
 * Returns a LayoutMap describing how input and output buffer layouts relate
 * for the configured reduction dimension.
 *
 * @param level Inference detail level that may affect how aggressively layouts
 * are inferred.
 * @return A LayoutMap mapping operator arguments to inferred layouts.
 */

/**
 * Retrieve the global operator descriptor for the reduce operator.
 *
 * @return A reference to the Op descriptor corresponding to this operator type.
 */

/**
 * Create a copy of this reduce operator as a TileOperator handle.
 *
 * The returned TileOperator preserves the node's configuration (buffers, dim,
 * type, clear).
 *
 * @return A TileOperator wrapping a cloned ReduceOpNode.
 */

/**
 * Construct the initial value used by the reduction (e.g., 0 for sum, -inf for
 * max).
 *
 * @return A PrimExpr representing the reduction's identity/init value.
 */

/**
 * Combine two partial values according to the configured reduction.
 *
 * Implements the binary reducer (for example, `a + b` for sum or `max(a, b)`
 * for max).
 *
 * @return A PrimExpr representing the reduced result of `a` and `b`.
 */

/**
 * Generate a string snippet suitable for code generation of the reducer
 * expression.
 *
 * The returned code fragment should implement the binary reduction operation in
 * the target backend's code string form.
 *
 * @return A std::string containing the codegen expression for the reducer.
 */

/**
 * Reference wrapper for ReduceOpNode as a TileOperator.
 *
 * Construct a ReduceOp from explicit arguments and a buffer map.
 */

/**
 * Construct a ReduceOp TileOperator from operator arguments and a buffer
 * mapping.
 *
 * @param args Operator arguments (typically shapes, axes, or other prim exprs).
 * @param vmap Mapping from argument names to tir::Buffer instances used by the
 * operator.
 */

/**
 * Tile operator node that computes a cumulative sum along a single tensor
 * dimension.
 *
 * Contains source/destination buffers, the target dimension, and a flag to
 * compute the cumulative sum in reverse order.
 */

/**
 * Lower this CumSumOpNode into a Tir Stmt suitable for code generation.
 *
 * Produces the TIR statement(s) that implement the configured cumulative-sum.
 *
 * @return A TIR `Stmt` implementing the cum-sum operation.
 */

/**
 * Infer input/output layouts for this cumulative-sum operator.
 *
 * Returns a LayoutMap describing how input and output buffer layouts relate
 * for the configured cumulative-sum dimension.
 *
 * @param level Inference detail level that may affect how aggressively layouts
 * are inferred.
 * @return A LayoutMap mapping operator arguments to inferred layouts.
 */

/**
 * Retrieve the global operator descriptor for the cumulative-sum operator.
 *
 * @return A reference to the Op descriptor corresponding to this operator type.
 */

/**
 * Create a copy of this cum-sum operator as a TileOperator handle.
 *
 * The returned TileOperator preserves the node's configuration (buffers, dim,
 * reverse).
 *
 * @return A TileOperator wrapping a cloned CumSumOpNode.
 */

/**
 * Reference wrapper for CumSumOpNode as a TileOperator.
 *
 * Construct a CumSumOp from explicit arguments and a buffer map.
 */

/**
 * Construct a CumSumOp TileOperator from operator arguments and a buffer
 * mapping.
 *
 * @param args Operator arguments (typically shapes, axes, or other prim exprs).
 * @param vmap Mapping from argument names to tir::Buffer instances used by the
 * operator.
 */
namespace tl {

using namespace tir;

enum class ReduceType {
  kSum,
  kAbsSum,
  kMax,
  kMin,
  kAbsMax,
};

class ReduceOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst;
  int dim;
  ReduceType type;
  bool clear;

  static constexpr const char *_type_key = "tl.ReduceOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReduceOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;

private:
  PrimExpr MakeInitValue() const;
  PrimExpr MakeReduce(const PrimExpr &a, const PrimExpr &b) const;
  std::string MakeCodegenReducer() const;
};

class ReduceOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(ReduceOp, TileOperator, ReduceOpNode);
  TVM_DLL ReduceOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

class CumSumOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst;
  int dim;
  bool reverse;
  static constexpr const char *_type_key = "tl.CumSumOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(CumSumOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

class CumSumOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(CumSumOp, TileOperator, CumSumOpNode);
  TVM_DLL CumSumOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_