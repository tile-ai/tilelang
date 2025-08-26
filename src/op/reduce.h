/*!
 * \file tl/op/reduce.h
 * \brief Define reduce operator.
 *
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class ReduceOp : public TileOperator {
public:
  ReduceOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();

  std::unique_ptr<TileOperator> Clone() const override {
    return std::make_unique<ReduceOp>(*this);
  }

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

class CumSumOp : public TileOperator {
public:
  CumSumOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();

  std::unique_ptr<TileOperator> Clone() const override {
    return std::make_unique<CumSumOp>(*this);
  }

private:
  tir::Buffer src, dst;
  int dim;
  bool reverse;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_