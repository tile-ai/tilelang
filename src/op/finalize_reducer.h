// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/op/finalize_reducer.h
 * \brief Define finalize_reducer operator.
 */

#ifndef TVM_TL_OP_FINALIZE_REDUCER_H_
#define TVM_TL_OP_FINALIZE_REDUCER_H_

#include "../transform/layout_reducer.h"
#include "./operator.h"

/**
 * FinalizeReducer operator node for Tile IR.
 *
 * Represents a TL-level operator that finalizes a reducer buffer into a
 * result using a specified reducer operation.
 *
 * Public members:
 * - reducer: the tir::Buffer that holds the intermediate reduction values.
 * - op: the reducer operation to apply when finalizing values.
 */

/**
 * Lower this operator to a TIR statement.
 *
 * @param T Lowering arguments (buffers, indices, and other lowering context).
 * @param analyzer Arithmetic analyzer used to simplify expressions during
 * lowering.
 * @return A tir::Stmt that implements the finalize-reducer semantics for the
 * provided lowering context.
 */

/**
 * Infer layout mapping for this operator.
 *
 * Determines how input and output buffer layouts relate for the
 * finalize-reducer operator at the given inference level.
 *
 * @param T Layout inference arguments (including operand layouts and shapes).
 * @param level Inference precision level.
 * @return A LayoutMap describing the inferred layouts.
 */

/**
 * Get the singleton Op object representing this operator.
 *
 * @return A reference to the Op describing FinalizeReducer.
 */

/**
 * Create a deep copy of this operator node as a TileOperator.
 *
 * @return A TileOperator handle that is an independent clone of this node.
 */

/**
 * Public wrapper for FinalizeReducerOpNode.
 *
 * Provides the reference semantics and construction API used by callers.
 */

/**
 * Construct a FinalizeReducerOp from TL-level arguments.
 *
 * @param args Positional primitive expressions that parameterize the operator
 *             (e.g., shapes, axis indices). Documented where their meaning is
 *             not obvious from name or type in call sites.
 * @param vmap Mapping from operand names to tir::Buffer instances used by this
 * operator.
 */

/**
 * Get the Op singleton for the public FinalizeReducerOp handle.
 *
 * @return A reference to the Op describing FinalizeReducer.
 */
namespace tvm {
namespace tl {

using namespace tir;

class FinalizeReducerOpNode : public TileOperatorNode {
public:
  tir::Buffer reducer;
  ReducerOpType op;

  static constexpr const char *_type_key = "tl.FinalizeReducerOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(FinalizeReducerOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

class FinalizeReducerOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(FinalizeReducerOp, TileOperator,
                                FinalizeReducerOpNode);
  TVM_DLL FinalizeReducerOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_FINALIZE_REDUCER_H_