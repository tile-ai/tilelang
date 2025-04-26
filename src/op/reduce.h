// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/reduce.h
 * \brief Define reduce operator.
 *
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class ReduceOp : public Operator {
public:
  ReduceOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();

private:
  tir::Buffer src, dst;
  int dim;
  enum class ReduceType {
    kSum,
    kAbsSum,
    kMax,
    kMin,
    kAbsMax,
  } type;
  bool clear;

  PrimExpr MakeInitValue() const;
  PrimExpr MakeReduce(const PrimExpr &a, const PrimExpr &b) const;
  std::string MakeCodegenReducer() const;
};

class CumSumOp : public Operator {
public:
  CumSumOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();

private:
  tir::Buffer src, dst;
  int dim;
  bool reverse;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_